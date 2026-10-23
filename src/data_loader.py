"""
data_loader.py - Automated acquisition, memory optimization, and loading for M5 Dataset.

Part of CSAI Student Project:
Project 5: Inventory Demand Forecasting via Kalman Filters and DeepAR Global Models
Phase 1: Dataset Collection, Data Quality, and Exploratory Data Analysis (EDA)
"""

import os
import sys
import time
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


BASE_HF_URL = "https://huggingface.co/datasets/denephew/M5_Forecasting/resolve/main"
M5_FILES = {
    "calendar.csv": f"{BASE_HF_URL}/calendar.csv",
    "sales_train_validation.csv": f"{BASE_HF_URL}/sales_train_validation.csv",
    "sell_prices.csv": f"{BASE_HF_URL}/sell_prices.csv",
}

EXPECTED_SIZES = {
    "calendar.csv": 103469,
    "sales_train_validation.csv": 120007726,
    "sell_prices.csv": 203395785,
}


class DownloadProgressBar(tqdm):
    """Custom progress bar hook for urllib urlretrieve."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, expected_size: int = None) -> Path:
    """Download a file with progress bar and size verification."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        actual_size = output_path.stat().st_size
        if expected_size is None or actual_size == expected_size or actual_size > 100000:
            print(f"[CACHE] {output_path.name} already exists ({actual_size / (1024*1024):.2f} MB). Skipping download.")
            return output_path
        else:
            print(f"[RETRY] {output_path.name} size mismatch (found {actual_size}, expected {expected_size}). Re-downloading.")

    print(f"[DOWNLOADING] {output_path.name} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as response:
        total_size = int(response.info().get("Content-Length", 0))
        with open(output_path, "wb") as out_file:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=output_path.name) as pbar:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    pbar.update(len(chunk))

    print(f"[SUCCESS] Saved {output_path.name} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
    return output_path


def download_m5_dataset(raw_dir: str = "data/raw") -> dict:
    """
    Downloads all essential M5 competition files into data/raw directory.
    Returns dict mapping filename to Path.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    downloaded_paths = {}

    for filename, url in M5_FILES.items():
        exp_sz = EXPECTED_SIZES.get(filename)
        dest_file = raw_path / filename
        download_file(url, dest_file, exp_sz)
        downloaded_paths[filename] = dest_file

    return downloaded_paths


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterates through DataFrame columns and downcasts numerical types
    to minimize memory footprint. Crucial for scaling M5 dataset in memory.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        # Skip datetimes and categoricals
        if pd.api.types.is_datetime64_any_dtype(df[col]) or isinstance(col_type, pd.CategoricalDtype) or col_type == object:
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if pd.api.types.is_integer_dtype(col_type):
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        elif pd.api.types.is_float_dtype(col_type):
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")

    return df


def load_calendar(path: str = "data/raw/calendar.csv", downcast: bool = True) -> pd.DataFrame:
    """Load and optimize calendar.csv."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if downcast:
        cat_cols = ["wm_yr_wk", "weekday", "wday", "month", "year", 
                    "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].astype("category")
        df = reduce_mem_usage(df, verbose=False)
    return df


def load_sell_prices(path: str = "data/raw/sell_prices.csv", downcast: bool = True) -> pd.DataFrame:
    """Load and optimize sell_prices.csv."""
    df = pd.read_csv(path)
    if downcast:
        df["store_id"] = df["store_id"].astype("category")
        df["item_id"] = df["item_id"].astype("category")
        df = reduce_mem_usage(df, verbose=False)
    return df


def load_sales_train(path: str = "data/raw/sales_train_validation.csv", downcast: bool = True) -> pd.DataFrame:
    """Load and optimize sales_train_validation.csv."""
    df = pd.read_csv(path)
    if downcast:
        cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
        for c in cat_cols:
            df[c] = df[c].astype("category")
        df = reduce_mem_usage(df, verbose=False)
    return df


if __name__ == "__main__":
    print("=== M5 Dataset Automated Acquisition ===")
    download_m5_dataset("data/raw")
