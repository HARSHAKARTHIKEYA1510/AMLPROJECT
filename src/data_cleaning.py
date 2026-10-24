"""
data_cleaning.py - Direct 1-to-1 Data Cleaning Pipeline for M5 Dataset.

Takes the 5 raw files from data/raw/ and produces exactly 5 cleaned files in data/processed/:
1. calendar.csv                 -> cleaned_calendar.csv
2. sell_prices.csv              -> cleaned_sell_prices.csv
3. sales_train_validation.csv   -> cleaned_sales_train_validation.csv
4. sales_train_evaluation.csv   -> cleaned_sales_train_evaluation.csv
5. sample_submission.csv        -> cleaned_sample_submission.csv
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.data_loader import reduce_mem_usage


class M5DataCleaner:
    """
    Cleans raw M5 retail tables with a strict 1-to-1 mapping
    from data/raw/ to data/processed/.
    """

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def clean_calendar(self) -> pd.DataFrame:
        """Clean calendar.csv -> cleaned_calendar.csv"""
        print("[1/5 CLEANING] Processing calendar.csv...")
        df = pd.read_csv(self.raw_dir / "calendar.csv")
        df["date"] = pd.to_datetime(df["date"])

        # Cyclical temporal features
        df["dayofweek"] = df["date"].dt.dayofweek
        df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)
        df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)

        df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12.0).astype(np.float32)
        df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12.0).astype(np.float32)

        # Event flags
        df["has_event_1"] = df["event_name_1"].notnull().astype(np.int8)
        df["has_event_2"] = df["event_name_2"].notnull().astype(np.int8)
        df["is_christmas"] = ((df["date"].dt.month == 12) & (df["date"].dt.day == 25)).astype(np.int8)

        # Categorical event labels
        df["event_name_1"] = df["event_name_1"].fillna("None")
        df["event_type_1"] = df["event_type_1"].fillna("None")
        df["event_name_2"] = df["event_name_2"].fillna("None")
        df["event_type_2"] = df["event_type_2"].fillna("None")

        out_path = self.processed_dir / "cleaned_calendar.csv"
        df.to_csv(out_path, index=False)
        print(f"[SAVED] {out_path.name} ({len(df):,} rows)")
        return df

    def clean_sell_prices(self) -> pd.DataFrame:
        """Clean sell_prices.csv -> cleaned_sell_prices.csv"""
        print("[2/5 CLEANING] Processing sell_prices.csv...")
        df = pd.read_csv(self.raw_dir / "sell_prices.csv")

        # Regular price baseline & discount percentage
        regular_price = df.groupby(["store_id", "item_id"], observed=True)["sell_price"].transform("max")
        df["regular_price"] = regular_price.round(2)
        discount = (df["regular_price"] - df["sell_price"]) / df["regular_price"]
        df["discount_pct"] = np.clip(discount, 0.0, 1.0).round(4)
        df["is_discounted"] = (df["discount_pct"] > 0.05).astype(np.int8)

        out_path = self.processed_dir / "cleaned_sell_prices.csv"
        df.to_csv(out_path, index=False)
        print(f"[SAVED] {out_path.name} ({len(df):,} rows)")
        return df

    def clean_sales_validation(self) -> pd.DataFrame:
        """Clean sales_train_validation.csv -> cleaned_sales_train_validation.csv"""
        print("[3/5 CLEANING] Processing sales_train_validation.csv...")
        df = pd.read_csv(self.raw_dir / "sales_train_validation.csv")

        # Verify no missing or negative values
        d_cols = [c for c in df.columns if c.startswith("d_")]
        df[d_cols] = df[d_cols].fillna(0).clip(lower=0)

        out_path = self.processed_dir / "cleaned_sales_train_validation.csv"
        df.to_csv(out_path, index=False)
        print(f"[SAVED] {out_path.name} ({len(df):,} rows, {len(d_cols)} days)")
        return df

    def clean_sales_evaluation(self) -> pd.DataFrame:
        """Clean sales_train_evaluation.csv -> cleaned_sales_train_evaluation.csv"""
        print("[4/5 CLEANING] Processing sales_train_evaluation.csv...")
        df = pd.read_csv(self.raw_dir / "sales_train_evaluation.csv")

        d_cols = [c for c in df.columns if c.startswith("d_")]
        df[d_cols] = df[d_cols].fillna(0).clip(lower=0)

        out_path = self.processed_dir / "cleaned_sales_train_evaluation.csv"
        df.to_csv(out_path, index=False)
        print(f"[SAVED] {out_path.name} ({len(df):,} rows, {len(d_cols)} days)")
        return df

    def clean_sample_submission(self) -> pd.DataFrame:
        """Clean sample_submission.csv -> cleaned_sample_submission.csv"""
        print("[5/5 CLEANING] Processing sample_submission.csv...")
        df = pd.read_csv(self.raw_dir / "sample_submission.csv")

        # Fill any null values with 0
        f_cols = [c for c in df.columns if c.startswith("F")]
        df[f_cols] = df[f_cols].fillna(0.0)

        out_path = self.processed_dir / "cleaned_sample_submission.csv"
        df.to_csv(out_path, index=False)
        print(f"[SAVED] {out_path.name} ({len(df):,} rows)")
        return df

    def run_cleaning_pipeline(self):
        """Executes full 1-to-1 cleaning pipeline."""
        start = time.time()
        print("=" * 70)
        print("  M5 DATA CLEANING: 5 RAW FILES -> 5 CLEANED FILES")
        print("=" * 70)

        self.clean_calendar()
        self.clean_sell_prices()
        self.clean_sales_validation()
        self.clean_sales_evaluation()
        self.clean_sample_submission()

        elapsed = time.time() - start
        print(f"\n[DONE] All 5 cleaned files successfully created in {elapsed:.2f} seconds.")
        print(f"Directory: {self.processed_dir.resolve()}")


if __name__ == "__main__":
    cleaner = M5DataCleaner()
    cleaner.run_cleaning_pipeline()
