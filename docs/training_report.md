# Training And Benchmark Report

This document summarizes the current reconstruction pipeline and benchmark setup.

## Goal

The model reconstructs a 60-point minute-level target price path inside a completed one-hour interval.

Inputs:

- target hourly OHLC,
- sparse target minute observations,
- dense index minute series,
- rolling target-index correlations.

Output:

- completed target minute point-price path.

The task is historical reconstruction/imputation, not live prediction.

## Data

### Synthetic Data

Synthetic data is generated as paired index/target price paths with configurable:

- correlation,
- target volatility,
- trajectory regime.

Supported regimes:

- `gbm`: correlated geometric random walk,
- `stochastic_vol`: block-wise changing volatility,
- `jumps`: common and target-specific jumps,
- `mixed`: stochastic volatility plus jumps.

The synthetic pipeline produces canonical minute OHLC and hourly OHLC, then builds reconstruction samples from those files.

### Real Data

Real market data is private/licensed and is not included in this repository.

Private `.prn` files are converted locally into canonical minute/hour OHLC files. Public scripts then build reconstruction datasets from canonical OHLC. Reported real-data metrics are aggregate benchmark results only.

For the current liquid benchmark, instruments were selected by objective coverage criteria:

- sufficient number of samples,
- high mean observed minute count,
- high lower-quartile observed minute count,
- low share of very sparse intervals.

The current liquid core contains:

```text
PKNORLEN, KGHM, PKOBP, PZU, CDPROJEKT, PEKAO, JSW, PGE
```

## Split Strategy

Synthetic data:

- random train/validation split for pretraining.

Real data:

- train: intervals before 2020-01-01,
- validation: 2020-01-01 to 2022-01-01,
- test: intervals from 2022-01-01 onward.

Final claims use the out-of-time real test split.

## Sparse Target Simulation

During training and evaluation, known target points can be hidden from the model. Metrics are computed on known target points that were not visible as input.

This tests reconstruction quality under partial observability while avoiding evaluation on points already given to the model.

## Current Model

The current MVP model is `PriorCorrectionModel`. It consumes:

- `X_index`,
- `X_index_mask`,
- `X_ts`,
- `X_ts_mask`,
- `X_prior`,
- `X_prior_mask`,
- `X_static`,
- `X_static_mask`.

`X_prior` is the deterministic index-residual baseline materialized before training. The network predicts a bounded residual correction:

```text
prediction = X_prior + 0.25 * tanh(neural_correction)
```

Known target observations are inserted back into the output.

## Loss

Current training loss:

```text
loss = mse + 0.6 * diff_loss
```

Where:

- `mse` measures pointwise reconstruction error on hidden known points,
- `diff_loss` measures first-difference shape error,
- both terms are computed only on target points not visible to the model.

## Baselines

### Linear

Interpolates between:

- target open,
- target close,
- visible target observations.

### Index Residual

For each interval between known target anchors:

```text
target_linear = linear interpolation between target anchors
index_linear  = linear interpolation between index endpoints
index_resid   = index - index_linear
prediction    = target_linear + beta * index_resid
```

Where `beta` is the mean of `corr_30` and `corr_60`, clipped to `[-1, 1]`.

Visible target observations are preserved exactly.

## Current Benchmark

Final benchmark summary:

```text
setting                  method             MSE       MAE       RMSE       points
synthetic validation     prior_correction   0.0176    0.0948    0.1326    5,259,239
synthetic validation     index_residual     0.0250    0.1080    0.1581    5,259,239
synthetic validation     linear             0.0347    0.1307    0.1863    5,259,239

real zero-shot           prior_correction   0.0132    0.0836    0.1151    1,519,629
real zero-shot           index_residual     0.0135    0.0833    0.1161    1,519,629
real zero-shot           linear             0.0160    0.0901    0.1265    1,519,629

real finetuned test      prior_correction   0.0119    0.0793    0.1092    1,519,629
real finetuned test      index_residual     0.0135    0.0833    0.1161    1,519,629
real finetuned test      linear             0.0160    0.0901    0.1265    1,519,629
```

Interpretation:

- the neural residual model improves over linear interpolation and the index-residual prior,
- synthetic pretraining transfers to real data, but real-data finetuning gives the best out-of-time result,
- the improvement is measurable but conservative in plots,
- future work can make the model more reactive to return-space index moves and local extremes.

## Model Direction

The strongest deterministic baseline already captures much of the useful index information. A more natural neural setup is:

```text
baseline = index_residual_baseline(inputs)
correction = PriorCorrectionModel(inputs, baseline)
prediction = baseline + correction
```

This is the current MVP architecture.
