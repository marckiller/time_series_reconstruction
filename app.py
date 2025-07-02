from fastapi import FastAPI
from pydantic import BaseModel, create_model
from typing import List, Optional
import torch
import yaml
import importlib
from src.predictor import JSONPredictor

app = FastAPI()

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

fields = {
    config["input"]["index_series_name"]: (List[Optional[float]], ...),
    config["input"]["ts_series_name"]: (List[Optional[float]], ...)
}
for feature in config["input"]["static_features"]:
    fields[feature] = (Optional[float], ...)

PredictRequest = create_model("PredictRequest", **fields)

@app.post("/predict")
async def predict(request: PredictRequest):
    data_json = {
        'X_index': getattr(request, config["input"]["index_series_name"]),
        'X_ts': getattr(request, config["input"]["ts_series_name"]),
        'X_static': [getattr(request, f) for f in config["input"]["static_features"]]
    }
    y_pred = predictor.predict(data_json)
    return {"prediction": y_pred.tolist()}