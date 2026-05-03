from typing import List, Optional

import importlib
import math

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.baselines import index_residual_baseline, linear_baseline
from src.predictor import JSONPredictor


app = FastAPI(title="Intrahour Time Series Reconstruction API")

with open("config/config_api.yaml", "r") as f:
    config = yaml.safe_load(f)["api"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_module = importlib.import_module(config["model"]["module"])
model_class = getattr(model_module, config["model"]["class_name"])
model_kwargs = config["model"].get("kwargs", {})

model = model_class(**model_kwargs).to(device)
model.load_state_dict(torch.load(config["model"]["model_path"], map_location=device))
model.eval()

predictor = JSONPredictor(model, device)


class OHLC(BaseModel):
    open: float
    high: float
    low: float
    close: float


class ReconstructionFeatures(BaseModel):
    corr_30: float
    corr_60: float


class ReconstructRequest(BaseModel):
    target_ohlc: OHLC
    target_sparse: Optional[List[Optional[float]]] = None
    index_series: List[float]
    features: ReconstructionFeatures
    method: str = "model"
    return_normalized: bool = False
    clip: bool = True


def ensure_finite(name: str, values: list[float]) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise HTTPException(status_code=422, detail=f"{name} must contain only finite values.")


def minmax_normalize(values: list[float]) -> tuple[list[float], float, float]:
    arr = np.asarray(values, dtype=np.float32)
    low = float(np.min(arr))
    high = float(np.max(arr))
    if high <= low:
        return [0.5] * len(values), low, high
    normalized = (arr - low) / (high - low)
    return normalized.astype(float).tolist(), low, high


def target_position(value: float, low: float, high: float) -> float:
    return float((value - low) / (high - low))


def build_model_input(request: ReconstructRequest) -> tuple[dict, float, float, int]:
    if len(request.index_series) != 60:
        raise HTTPException(status_code=422, detail="index_series must contain exactly 60 values.")
    ensure_finite("index_series", request.index_series)

    target = request.target_ohlc
    target_values = [target.open, target.high, target.low, target.close]
    ensure_finite("target_ohlc", target_values)
    if target.high <= target.low:
        raise HTTPException(status_code=422, detail="target_ohlc.high must be greater than target_ohlc.low.")

    target_sparse = request.target_sparse
    if target_sparse is None:
        target_sparse = [None] * 60
    if len(target_sparse) != 60:
        raise HTTPException(status_code=422, detail="target_sparse must contain exactly 60 values.")

    target_range = target.high - target.low
    x_ts: list[Optional[float]] = []
    for value in target_sparse:
        if value is None:
            x_ts.append(None)
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise HTTPException(status_code=422, detail="target_sparse must contain finite numbers or null.")
        if numeric < target.low or numeric > target.high:
            raise HTTPException(status_code=422, detail="target_sparse values must be inside target_ohlc low/high range.")
        x_ts.append((numeric - target.low) / target_range)

    # Hourly open and close are hard anchors for the reconstruction.
    x_ts[0] = target_position(target.open, target.low, target.high)
    x_ts[-1] = target_position(target.close, target.low, target.high)

    x_index, index_low, index_high = minmax_normalize(request.index_series)
    if index_high > index_low:
        index_open_pos = (request.index_series[0] - index_low) / (index_high - index_low)
        index_close_pos = (request.index_series[-1] - index_low) / (index_high - index_low)
        index_body_to_range = abs(request.index_series[-1] - request.index_series[0]) / (index_high - index_low)
        index_direction = 1.0 if request.index_series[-1] >= request.index_series[0] else -1.0
    else:
        index_open_pos = 0.5
        index_close_pos = 0.5
        index_body_to_range = 0.0
        index_direction = 0.0

    direction = 1.0 if target.close >= target.open else -1.0
    x_static = [
        float(request.features.corr_30),
        float(request.features.corr_60),
        target_position(target.open, target.low, target.high),
        target_position(target.close, target.low, target.high),
        abs(target.close - target.open) / target_range,
        direction,
        float(index_open_pos),
        float(index_close_pos),
        float(index_body_to_range),
        float(index_direction),
    ]
    ensure_finite("features", x_static)

    known_points = sum(value is not None for value in target_sparse)
    return {"X_index": x_index, "X_ts": x_ts, "X_static": x_static}, target.low, target.high, known_points


def baseline_prediction(data_json: dict, method: str) -> np.ndarray:
    x_index = torch.tensor(data_json["X_index"], dtype=torch.float32).unsqueeze(0)
    x_ts_raw = torch.tensor(
        [np.nan if value is None else value for value in data_json["X_ts"]],
        dtype=torch.float32,
    ).unsqueeze(0)
    x_ts_mask = torch.isfinite(x_ts_raw).float()
    x_ts = torch.nan_to_num(x_ts_raw, nan=0.0)
    x_static = torch.tensor(data_json["X_static"], dtype=torch.float32).unsqueeze(0)

    if method == "linear":
        pred = linear_baseline(x_ts, x_ts_mask, x_static)
    elif method == "index_residual":
        pred = index_residual_baseline(x_ts, x_ts_mask, x_index, x_static)
    else:
        raise HTTPException(
            status_code=422,
            detail="method must be one of: model, index_residual, linear.",
        )
    return pred.squeeze(0).cpu().numpy()


def reconstruct_payload(request: ReconstructRequest):
    data_json, target_low, target_high, known_points = build_model_input(request)
    method = request.method
    if method == "model":
        pred_norm = predictor.predict(data_json)
        response_method = "prior_correction_model"
    elif method in {"index_residual", "linear"}:
        pred_norm = baseline_prediction(data_json, method)
        response_method = method
    else:
        raise HTTPException(
            status_code=422,
            detail="method must be one of: model, index_residual, linear.",
        )

    clipped = False
    if request.clip:
        clipped_pred = np.clip(pred_norm, 0.0, 1.0)
        clipped = bool(np.any(np.abs(clipped_pred - pred_norm) > 1e-8))
        pred_norm = clipped_pred

    if request.return_normalized:
        reconstructed = pred_norm.astype(float).tolist()
    else:
        reconstructed = (pred_norm * (target_high - target_low) + target_low).astype(float).tolist()

    return {
        "reconstructed": reconstructed,
        "normalized": request.return_normalized,
        "method": response_method,
        "metadata": {
            "known_points": int(known_points),
            "clipped": clipped,
        },
    }


@app.post("/reconstruct")
async def reconstruct(request: ReconstructRequest):
    return reconstruct_payload(request)


@app.post("/predict")
async def predict(request: ReconstructRequest):
    return reconstruct_payload(request)
