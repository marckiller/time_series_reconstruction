# API Contract

This document describes the intended public inference API after the repository cleanup.

## Endpoint

```text
POST /reconstruct
```

The endpoint reconstructs one completed 60-point target price path for a historical one-hour interval.

## Request

```json
{
  "target_ohlc": {
    "open": 101.2,
    "high": 103.0,
    "low": 100.5,
    "close": 102.1
  },
  "target_sparse": [
    101.2, null, null, null, null,
    101.8, null, null, null, null
  ],
  "index_series": [
    2500.1, 2500.4, 2499.8
  ],
  "features": {
    "corr_30": 0.52,
    "corr_60": 0.47
  },
  "return_normalized": false
}
```

Actual `target_sparse` and `index_series` arrays must contain exactly 60 elements. Short arrays above are illustrative only.

Missing target observations should be encoded as `null`.

## Required Inputs

Required:

- `target_ohlc.open`,
- `target_ohlc.high`,
- `target_ohlc.low`,
- `target_ohlc.close`,
- `index_series` with 60 values,
- `corr_30`,
- `corr_60`.

Optional:

- `target_sparse` with 60 values or nulls.

If `target_sparse` is omitted, all target minute slots are treated as missing except the endpoint may still anchor the interval using target open and close.

## Normalization

The service normalizes target values using target hourly low/high:

```text
normalized = (price - low) / (high - low)
```

The model predicts in normalized space. Unless `return_normalized=true`, the response is denormalized back to price units:

```text
price = normalized * (high - low) + low
```

Known target observations should be preserved exactly in the output.

## Response

```json
{
  "reconstructed": [
    101.2,
    101.31,
    101.44
  ],
  "normalized": false,
  "method": "neural_model",
  "metadata": {
    "known_points": 12,
    "clipped": false
  }
}
```

The `reconstructed` array contains exactly 60 values in the final API.

## Methods

The final API may expose multiple methods:

- `linear`,
- `index_residual`,
- `neural_model`.

The default should be chosen after final benchmarking. At the current stage, `index_residual` is the strongest deterministic method and the direct neural model is experimental.

## Validation Rules

The endpoint should reject requests when:

- `high <= low`,
- `index_series` is not length 60,
- `target_sparse` is present but not length 60,
- known target sparse values are outside a clearly invalid range,
- correlation values are missing or not finite.

The endpoint may clip normalized outputs to `[0, 1]` before denormalization, but this behavior should be reported in `metadata.clipped`.
