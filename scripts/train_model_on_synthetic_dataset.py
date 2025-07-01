import yaml
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.train_model import train_model

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

sim_dataset = config["data"]["synthetic_dataset"]
df = pd.read_parquet(sim_dataset).dropna()
df = df.sample(n=config["data"]["n_samples"], random_state=config["general"]["seed"])

print(f"Dropped NaN values, now {len(df)} rows")

train_model(df, config)