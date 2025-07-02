import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
with open("config_data_generation.yaml", "r") as f:
    config = yaml.safe_load(f)

import src.utils.syntetic_data_generation as sdg
import src.utils.preprocessing as pre

master_df = None

# Parameters for the synthetic data generation
n_intervals = config["generation"]["n_intervals"]
time_interval = config["generation"]["time_interval"]
start_time = config["generation"]["start_time"]
return_type = config["generation"]["return_type"]
output_path = config["generation"]["output_path"]

tick_values = config["simulation"]["tick_values"]
sigma_pairs = [tuple(pair) for pair in config["simulation"]["sigma_pairs"]]
correlations = config["simulation"]["correlations"]

total_cases = len(tick_values) * len(sigma_pairs) * len(correlations)
case_id = 0

for ticks_per_interval in tick_values:
    for sigma_index, sigma_instr in sigma_pairs:
        for corr in correlations:

            case_id += 1
            print(f"[{case_id}/{total_cases}] ticks: {ticks_per_interval}, sigma_index: {sigma_index}, sigma_instr: {sigma_instr}, corr: {corr}")


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

master_df.to_parquet(output_path, index=False)
print(f"\nMaster dataframe with {len(master_df)} rows saved to '{output_path}'")