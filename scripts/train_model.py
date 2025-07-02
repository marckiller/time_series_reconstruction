import yaml
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.training.train_model import train_model


with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_name = config["data"]["dataset_to_use"]
dataset_path = config["data"]["real_dataset"] if dataset_name == "real" else config["data"]["synthetic_dataset"]
df = pd.read_parquet(dataset_path).dropna()

if dataset_name == "real":
    start_date = config["data"].get("real_data_start")
    end_date = config["data"].get("real_data_end")
    if start_date and end_date:
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

df = df.sample(n=config["data"]["n_samples"], random_state=config["general"]["seed"])
print(f"Dropped NaN values, now {len(df)} rows")

train_model(df, config)