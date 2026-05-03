# time_series_reconstruction

Reconstruction of missing minute-level point prices from sparse stock observations and a market index trajectory.

The project targets offline historical data enhancement. For each completed one-hour interval, it reconstructs a 60-point target-stock price path using:

- hourly OHLC for the target instrument,
- optional sparse minute-level target observations,
- a dense minute-level index series for the same interval,
- stock-index relationship features such as rolling correlations.

The output is a completed minute-level point-price series. The project does not reconstruct full intrahour OHLC bars and does not forecast future prices.

## Project Status

This repository contains a compact MVP for intrahour price-path reconstruction:

- synthetic data generation,
- reconstruction dataset building,
- model training,
- benchmark evaluation,
- inference/API code.

Real market data used for private benchmarks is not included. The public pipeline is reproducible with synthetic data, while real-data results are reported only as aggregate metrics and anonymized plots.

## Reconstruction Contract

Each sample uses a canonical 60-slot grid:

```text
slot 0  -> first minute of the interval
slot 59 -> last minute of the interval
```

Main inputs:

```text
target_hour_ohlc      open, high, low, close for the target interval
target_sparse_series  length-60 target series, NaN where unknown
index_series          length-60 index series for the same interval
corr_30, corr_60      rolling stock-index correlations
```

Internally, target values are normalized by the target interval range:

```text
normalized = (price - target_low) / (target_high - target_low)
```

Known target observations are preserved in the reconstructed output.

See [docs/reconstruction_contract.md](docs/reconstruction_contract.md) for the full data and model contract.

## Baselines

The model is evaluated against two deterministic baselines:

- `linear`: interpolates between target open, close, and any known target points.
- `index_residual`: starts with the linear target baseline and adds local index deviations from its own linear path, scaled by rolling stock-index correlation.

The index residual baseline is intentionally strong. It encodes the core market intuition behind the project: the index path can provide useful intrahour shape information, but target OHLC and known target observations remain hard constraints.

## Current Benchmark Snapshot

Private real-data benchmark on liquid instruments, out-of-time test split:

```text
method           MSE       MAE       RMSE
linear           0.0160    0.0901    0.1266
index_residual   0.0135    0.0833    0.1160
prior_correction 0.0119    0.0793    0.1092
```

Interpretation:

- the neural model improves over simple linear interpolation and the index-residual baseline,
- the current MVP is deliberately conservative: it learns bounded corrections to a strong deterministic baseline,
- a future iteration can target more reactive return-space behavior and stronger local-extrema reconstruction.

These results are based on private licensed market data and are not directly reproducible from this repository alone.

## Public Synthetic Pipeline

Generate synthetic canonical minute/hour OHLC:

```bash
python scripts/create_synthetic_canonical_ohlc.py \
  --output-dir data/synthetic/canonical_v2 \
  --n-hours 8000 \
  --index-sigma 0.0006 \
  --target-sigmas 0.0006 0.0007 0.0008 0.0009 0.0010 0.0011 0.0012 \
  --correlations 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --trajectory-regimes gbm stochastic_vol jumps mixed
```

Build reconstruction samples:

```bash
python scripts/build_synthetic_reconstruction_dataset.py \
  --canonical-dir data/synthetic/canonical_v2 \
  --output-path data/synthetic/reconstruction_dataset_v2.parquet
```

Split the synthetic dataset:

```bash
python scripts/split_reconstruction_dataset.py \
  --input-path data/synthetic/reconstruction_dataset_v2.parquet \
  --output-dir data/synthetic/splits_v2 \
  --mode random \
  --train-frac 0.8
```

## Training And Evaluation

The current training pipeline uses a materialized index-residual prior. The model learns a bounded correction to that prior instead of reconstructing the whole path from scratch.

Materialize the prior for a reconstruction dataset:

```bash
python scripts/materialize_prior_reconstruction_dataset.py \
  --input-path data/synthetic/splits_v2/train.parquet \
  --output-path data/synthetic/prior_splits/train.parquet \
  --keep-prob 1.0

python scripts/materialize_prior_reconstruction_dataset.py \
  --input-path data/synthetic/splits_v2/val.parquet \
  --output-path data/synthetic/prior_splits/val.parquet \
  --keep-prob 1.0
```

Synthetic pretraining:

```bash
python scripts/train_reconstruction_model.py \
  --train-path data/synthetic/prior_splits/train.parquet \
  --val-path data/synthetic/prior_splits/val.parquet \
  --output-dir work/model_runs \
  --run-name prior_correction_synth \
  --model-module src.models.prior_correction_model \
  --model-class PriorCorrectionModel \
  --epochs 12 \
  --batch-size 1024 \
  --diff-weight 0.6 \
  --pull-weight 0.0 \
  --early-stopping-patience 4 \
  --early-stopping-min-delta 0.00005
```

Real-data finetuning uses private local data in the same reconstruction dataset format.

See [docs/training_report.md](docs/training_report.md) for the current experiment setup and benchmark notes.

## API

The API exposes reconstruction as a service: provide absolute target OHLC, optional sparse absolute target observations, an absolute index series, and correlation features; receive a 60-point reconstructed target series. Normalization and denormalization happen inside the backend.
The request can choose `method: "model"`, `method: "index_residual"`, or `method: "linear"`.

```bash
uvicorn app:app --reload
curl -X POST http://127.0.0.1:8000/reconstruct \
  -H "Content-Type: application/json" \
  --data @example.json
```

See [docs/api.md](docs/api.md) for the endpoint contract.

## Repository Hygiene

The following are intentionally excluded from git:

- private `.prn` market data,
- generated real datasets,
- generated synthetic datasets,
- local experiment runs,
- diagnostic scripts and plots under `work/`.

The public repository contains the reproducible synthetic workflow, benchmark summaries, inference code, and trained model artifacts only.
