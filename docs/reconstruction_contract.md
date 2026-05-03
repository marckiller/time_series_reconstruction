# Minute Price Reconstruction Contract

This document defines the reconstruction task implemented by this project. It is the reference point for data preparation, model inputs, training, evaluation, and API behavior.

## Task Definition

The project reconstructs missing minute-level point prices for a target stock inside a known historical interval.

The default project mode is offline historical data enhancement. The system is allowed to use information that is known after the historical interval has completed, such as target-stock interval features and the index series inside the same interval.

Given:

- interval-level data for a target stock,
- minute-level index series for the same interval,
- sparse minute-level point-price observations for the target stock,
- historical relationship features between the target stock and the index,

the system produces:

- a complete 60-point minute-level point-price series for the target stock inside that interval.

This is reconstruction/imputation, not forecasting. The output is a plausible completed point-price series conditional on known information, not a claim that the exact hidden historical prices can always be uniquely recovered.

## Time Contract

- `timestamp` denotes the start of the reconstructed interval.
- Each sample represents the half-open interval `[timestamp, timestamp + 1h)`.
- Each sample contains exactly 60 minute slots.
- Minute slot `i` represents `[timestamp + i minutes, timestamp + (i + 1) minutes)`.
- Input series for stock and index must refer to the same interval.

## Canonical Grid and Sparse Observations

The model input format is fixed. All target-stock point-price observations are projected onto the same 60-slot minute grid.

Lower-resolution or irregular target data does not create a different input shape. It creates a sparse vector on the canonical grid plus an observation mask.

Examples:

- 5-minute target data populates slots `0, 5, 10, ..., 55`,
- 10-minute target data populates slots `0, 10, 20, ..., 50`,
- 15-minute target data populates slots `0, 15, 30, 45`,
- irregular data populates any subset of slots,
- missing slots remain unavailable to the model.

The target-stock sparse input is represented by:

- `X_ts`: shape `(60,)`,
- `X_ts_mask`: shape `(60,)`.

Mask convention:

- `1.0`: this slot is observed and available to the model,
- `0.0`: this slot is missing or intentionally hidden.

Unobserved target slots are represented as `NaN` during data preparation. They are converted to `0.0` only after the mask has been created. The zero value itself is never interpreted as an observed price or normalized price unless the mask for that slot is `1.0`.

For example, sparse target observations are encoded after mask construction as:

```text
X_ts      = [x0, 0, 0, ..., x15, 0, 0, ..., x30, 0, 0, ..., x45, 0, ...]
X_ts_mask = [ 1, 0, 0, ...,   1, 0, 0, ...,   1, 0, 0, ...,   1, 0, ...]
```

The same contract applies during training, evaluation, and inference. The model does not need to know whether a sparse vector came from 5-minute bars, 15-minute bars, irregular missingness, or an API request.

## Known Inputs

For each target-stock interval, the system uses:

- stock interval features such as `open`, `high`, `low`, `close`,
- index minute-level series for the same interval,
- index interval features,
- sparse stock minute-level point-price observations inside the interval,
- stock-index relationship features computed from the historical data available for the reconstructed dataset.

The following information must not be used as a feature:

- full target-stock minute series, except as supervised training target or as explicitly visible sparse observations.

## Output Contract

The model or baseline returns one reconstructed target-stock point-price series:

- shape: `(60,)`,
- order: minute slots from `0` to `59`,
- type: floating point values,
- domain: normalized or price-level values, depending on the caller contract.

Known target-stock observations are preserved in the output.

If the output is normalized to the target-stock interval range, the expected convention is:

- `0.0` corresponds to the interval `low`,
- `1.0` corresponds to the interval `high`,
- values can be outside `[0, 1]` during raw model inference; evaluation and production post-processing report whether this happened.

## Model Input Contract

The current neural models use three logical inputs:

- `X_index`: minute-level index point-price series for the interval, shape `(60,)`.
- `X_ts`: sparse target-stock point-price series for the interval, shape `(60,)`.
- `X_static`: interval-level and relationship features, shape `(D,)`.

Each input with missing values must have a corresponding mask:

- mask value `1.0`: observed and available to the model,
- mask value `0.0`: missing or intentionally hidden from the model.

Missing numeric values are represented as `NaN` in data preparation and converted to zero only after masks are created.

## Static Feature Contract

Static features include:

- stock-index rolling correlations,
- target-stock interval features,
- index interval features,
- volatility/range features,
- direction/body/range features.

For offline historical enhancement, rolling relationship features include the current completed interval.

## Training Contract

During supervised training:

- `X_ts` is the sparse target-stock point-price series visible to the model,
- `X_ts_mask` marks target-stock slots visible to the model,
- `y` is the target-stock point-price series used for supervision and can also be sparse,
- `y_mask` marks target-stock slots where the true value is known,
- loss is computed only where `y_mask == 1`,
- reconstruction quality is measured primarily where `y_mask == 1` and `X_ts_mask == 0`.

The target series does not need to be complete. Partially observed historical data can still provide supervision for the observed target slots that were not given to the model as input.

The train/validation/test split is time-aware. Random row-level splitting is used only for quick smoke tests and is not reported as final evidence.

Recommended evaluation splits:

- out-of-time split: train on earlier dates, validate/test on later dates,
- out-of-ticker split: train on some stocks, test on unseen stocks,
- combined split when enough data is available.

## Baseline Contract

Neural models are compared against simple baselines before benchmark claims are made.

Minimum useful baselines:

- linear interpolation/fill,
- index-informed residual fill.

The current preferred index-informed baseline is:

```text
target_linear = interpolation between known target anchors
index_linear  = interpolation between index endpoints over the same segment
index_resid   = index - index_linear
prediction    = target_linear + beta * index_resid
```

Where `beta` is derived from stock-index relationship features, such as the average of `corr_30` and `corr_60`.

This baseline preserves:

- target open,
- target close,
- visible sparse target observations.

A model result is meaningful only if it improves on relevant baselines under the same information constraints.

## Evaluation Contract

Pointwise metrics are reported:

- MSE,
- MAE,
- R2.

They are not sufficient on their own. Reconstruction benchmarks also report:

- error of minimum value,
- error of maximum value,
- error of minimum position,
- error of maximum position,
- correctness of high-before-low ordering,
- error on hidden positions only,
- behavior under different levels of sparse target-stock observations.

For trading-oriented use, evaluation can include scenario metrics such as whether the reconstructed series changes stop-loss/take-profit outcomes compared with the known minute-level series.

## Interpretation Limits

Sparse historical observations do not uniquely determine the missing minute-level point prices. Many completed series can be compatible with the same observed inputs.

Therefore:

- output is interpreted as a plausible imputation,
- low pointwise error does not guarantee correct hidden price dynamics,
- the project is not described as recovering uniquely determined hidden data,
- uncertainty-aware or multi-sample reconstruction is preferable for downstream strategy research.
