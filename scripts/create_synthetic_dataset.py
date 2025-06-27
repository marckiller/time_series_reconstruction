import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.utils.syntetic_data_generation as sdg

master_df = None

#parameters of the synthetic data generation
n_intervals = 60*60*100
ticks_per_interval = 10
time_interval="1min"
#target_corr = 0.5
sigma_index=0.01
sigma_instr=0.015
start_time="2025-01-01 00:00:00"
return_type="log"

correlations = [-0.9, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9]

for corr in correlations:

    print(f"Generating synthetic data with target correlation: {corr}")

    idx_df, instr_df = sdg.generate_synthetic_ohlc(
        n_intervals=n_intervals,
        ticks_per_interval=ticks_per_interval,
        time_interval=time_interval,
        target_corr=corr,
        sigma_index=sigma_index,
        sigma_instr=sigma_instr,
        start_time=start_time,
        return_type=return_type
    )

    minute_ohlc_index = sdg.aggregate_ohlc(idx_df, '1min')
    hour_ohlc_index = sdg.aggregate_ohlc(minute_ohlc_index, '1h')

    time_series_index = sdg.extract_time_series_per_interval(
        high_df=hour_ohlc_index,
        low_df=minute_ohlc_index,
        price_column='close',
        include_first_open=True,
        fill_value=-1.0,
        high_percentile=90.0,
        low_percentile=10.0,
        add_min_max=True,
        add_dist_to_extremes=True,
        add_percentile_masks=True,
        add_norm_local=True,
        add_norm_global=True,
        add_heat=True
    )

    minute_ohlc_instr = sdg.aggregate_ohlc(instr_df, '1min')
    hour_ohlc_instr = sdg.aggregate_ohlc(instr_df, '1h')

    time_series_instr = sdg.extract_time_series_per_interval(
        high_df=hour_ohlc_instr,
        low_df=minute_ohlc_instr,
        price_column='close',
        include_first_open=True,
        fill_value=-1.0,
        add_min_max=True,
        add_dist_to_extremes=False,
        add_percentile_masks=False,
        add_norm_local=True,
        add_norm_global=True,
        add_heat=False
    )

    index_hourly_returns = sdg.compute_returns(
        df = hour_ohlc_index,
        price_column='close',
        return_type='additive',
        output_column='log_return'
    )

    instrument_hourly_returns = sdg.compute_returns(
        df = hour_ohlc_instr,
        price_column='close',
        return_type='additive',
        output_column='log_return'
    )

    corr30h = sdg.compute_rolling_correlation(
        df1 = index_hourly_returns,
        df2 = instrument_hourly_returns,
        column1='log_return',
        column2='log_return',
        window=30,
        output_column='rolling_corr_h'
    )

    corr60h = sdg.compute_rolling_correlation(
        df1 = index_hourly_returns,
        df2 = instrument_hourly_returns,
        column1='log_return',
        column2='log_return',
        window=60,
        output_column='rolling_corr_60h'
    )

    #small trick to avoid computing long correlations
    corr60d = corr60h[['timestamp']].copy()
    corr60d['rolling_corr_60d'] = target_corr

    hour_ohlc_index_metrics = sdg.compute_interval_metrics(
        df=hour_ohlc_index,
        include=['open_pos', 'close_pos', 'body_to_range', 'ditection'])

    hour_ohlc_instr_metrics = sdg.compute_interval_metrics(
        df=hour_ohlc_instr,
        include=['open_pos', 'close_pos', 'body_to_range', 'ditection'])

    index_min_max_to_ohlc = sdg.compare_series_minmax_to_ohlc(time_series_df=time_series_index, ohlc_df=hour_ohlc_index)

    df = sdg.build_summary_dataframe(
        hour_ohlc_instr=hour_ohlc_instr,
        time_series_instr=time_series_instr,
        corr30h=corr30h,
        corr60h=corr60h,
        corr60d=corr60d,
        hour_ohlc_instr_metrics=hour_ohlc_instr_metrics,
        hour_ohlc_index=hour_ohlc_index,
        time_series_index=time_series_index,
        hour_ohlc_index_metrics=hour_ohlc_index_metrics,
        index_min_max_to_ohlc=index_min_max_to_ohlc
    )

    master_df = sdg.append_to_master_dataframe(master_df, df)

    print(f"Finished generating synthetic data with target correlation: {len(df)} samples")

master_df.to_parquet("data/synthetic/dataset.parquet", index=False)