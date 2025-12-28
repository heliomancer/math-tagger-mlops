import pandas as pd
from datasets import load_dataset
import os

# Constants
DATASET_PATH = "EleutherAI/hendrycks_math"
CONFIGS = ["algebra", "geometry", "precalculus", "intermediate_algebra", 
           "number_theory", "prealgebra", "counting_and_probability"]
SAVE_DIR = "data/raw"

def download_data():
    print(f"Downloading data from {DATASET_PATH}...")
    train_list = []
    test_list = []

    for config in CONFIGS:
        print(f"Processing {config}...")
        ds = load_dataset(DATASET_PATH, config, trust_remote_code=True)
        train_list.append(pd.DataFrame(ds['train']))
        test_list.append(pd.DataFrame(ds['test']))

    # Combine
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    # Save to disk
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_path = os.path.join(SAVE_DIR, "train.csv")
    test_path = os.path.join(SAVE_DIR, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved {len(train_df)} train and {len(test_df)} test samples to {SAVE_DIR}")

if __name__ == "__main__":
    download_data()
