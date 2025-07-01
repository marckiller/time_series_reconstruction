import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.utils.syntetic_data_generation as sdg
import src.utils.preprocessing as pre

import os

import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

index_name = config["data"]["index"]
folder_path = config["data"]["raw_data_path"]

prn_files = [
    f[:-4] for f in os.listdir(folder_path)
    if f.endswith(".prn") and f[:-4] != index_name
]

index = pre.load_prn_file(index_name, folder_path)
index["timestamp"] = index["timestamp"] - pd.Timedelta(seconds=1)
minute_index = pre.aggregate_ohlc(index, '1min')
minute_index = pre.fill_missing_intervals(minute_index, interval= '1min')
hour_index = pre.aggregate_ohlc(minute_index, '1h')
ts_index = pre.build_ts_dataframe(hour_df=hour_index, minute_df = minute_index)
hour_index_metrics = pre.compute_interval_metrics(hour_index)

master_df = None

for name in prn_files:

    print(f"Processing real data with target: {name}")

    instrument = pre.load_prn_file(name, folder_path)
    instrument["timestamp"] = instrument["timestamp"] - pd.Timedelta(seconds=1)

    minute_instrument = pre.aggregate_ohlc(instrument, '1min')
    minute_instrument = pre.fill_missing_intervals(minute_instrument, interval='1min')
    hour_instrument = pre.aggregate_ohlc(minute_instrument, '1h')

    ts_instrument = pre.build_ts_dataframe(hour_df=hour_instrument, minute_df=minute_instrument)

    hour_index_dummy, hour_instrument = pre.align_on_common_timestamps(hour_index, hour_instrument)

    hour_index_dummy = pre.compute_returns(hour_index_dummy, output_column='ret_log')
    hour_instrument = pre.compute_returns(hour_instrument, output_column='ret_log')

    hour_instrument = pre.add_rolling_correlations(hour_instrument, hour_index_dummy, column1='ret_log', column2='ret_log', windows = [30, 60])

    hour_instrument_metrics = pre.compute_interval_metrics(hour_instrument)

    real_df = pre.build_summary_dataframe(
        hour_instrument = hour_instrument,
        ts_instrument= ts_instrument,
        hour_instrument_metrics=hour_instrument_metrics,
        hour_index=hour_index,
        ts_index=ts_index,
        hour_index_metrics=hour_index_metrics,
        ticker = name
    )

    master_df = pre.append_to_master_dataframe(master_df, real_df)

    print(f"Master dataframe with {len(master_df)} rows created.")

output_path = config["data"]["real_dataset"]
master_df.to_parquet(output_path, index=False)
print(f"\nMaster dataframe with {len(master_df)} rows saved to '{output_path}'")

    



    