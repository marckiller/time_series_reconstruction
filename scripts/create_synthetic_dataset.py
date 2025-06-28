import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.utils.syntetic_data_generation as sdg
import src.utils.preprocessing as pre

master_df = None

#parameters of the synthetic data generation
n_intervals = 60*60*100
ticks_per_interval = 10
time_interval="1min"
sigma_index=0.008
sigma_instr=0.012
start_time="2025-01-01 00:00:00"
return_type="log"

correlations = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for corr in correlations:

    print(f"Generating synthetic data with target correlation: {corr}")

    synthetic_index, synthetic_instrument = sdg.generate_synthetic_ohlc(
        n_intervals=n_intervals,
        ticks_per_interval=ticks_per_interval,
        time_interval=time_interval,
        target_corr=corr,
        sigma_index=sigma_index,
        sigma_instr=sigma_instr,
        start_time=start_time,
        return_type=return_type
    )

    minute_synthetic_index = pre.aggregate_ohlc(synthetic_index, '1min')
    minute_synthetic_instrument = pre.aggregate_ohlc(synthetic_instrument, '1min')

    hour_synthetic_index = pre.aggregate_ohlc(minute_synthetic_index, '1h')
    hour_synthetic_instrument = pre.aggregate_ohlc(minute_synthetic_instrument, '1h')

    ts_synthetic_index = pre.build_ts_dataframe(hour_df=hour_synthetic_index, minute_df=minute_synthetic_index)
    ts_synthetic_instrument = pre.build_ts_dataframe(hour_df=hour_synthetic_instrument, minute_df=minute_synthetic_instrument)

    hour_synthetic_index, hour_synthetic_instrument = pre.align_on_common_timestamps(hour_synthetic_index, hour_synthetic_instrument)

    hour_synthetic_index = pre.compute_returns(hour_synthetic_index, output_column='ret_log')
    hour_synthetic_instrument = pre.compute_returns(hour_synthetic_instrument, output_column='ret_log')    

    hour_synthetic_instrument = pre.add_rolling_correlations(hour_synthetic_instrument, hour_synthetic_index, column1='ret_log', column2='ret_log', windows=[30, 60])

    hour_synthetic_instrument_metrics = pre.compute_interval_metrics(hour_synthetic_instrument)
    hour_synthetic_index_metrics = pre.compute_interval_metrics(hour_synthetic_index)

    fake_df = pre.build_summary_dataframe(
        hour_instrument = hour_synthetic_instrument,
        ts_instrument= ts_synthetic_instrument,
        hour_instrument_metrics=hour_synthetic_instrument_metrics,
        hour_index=hour_synthetic_index,
        ts_index=ts_synthetic_index,
        hour_index_metrics=hour_synthetic_index_metrics
    )

    master_df = pre.append_to_master_dataframe(master_df, fake_df)

    print(f"     Rows: {len(fake_df)} added to master dataframe")

master_df.to_parquet("data/synthetic/dataset.parquet", index=False)
print(f"\nMaster dataframe with {len(master_df)} rows saved to 'data/synthetic/dataset.parquet'")