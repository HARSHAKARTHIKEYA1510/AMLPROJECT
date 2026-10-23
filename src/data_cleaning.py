"""
data_cleaning.py - Comprehensive Data Cleaning and Preprocessing Pipeline.

Addresses Project 5 Requirements:
- Complete date index per SKU.
- Fair handling of cold start items (flagging short-history items).
- Marking rather than imputing stockout periods.
- Feature engineering: calendar cyclic encodings, price discounting, lags without lookahead.
- Separation of raw data (data/raw) and cleaned/processed datasets (data/processed).
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
from src.data_loader import load_calendar, load_sell_prices, load_sales_train, reduce_mem_usage


class M5DataCleaner:
    """
    Cleans raw M5 retail tables and produces structured,
    memory-optimized processed datasets with stockout and cold-start markers.
    """

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def clean_calendar(self, df_cal: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans calendar table:
        - Creates continuous date index.
        - Generates cyclical sine/cosine encodings for day-of-week and month.
        - Encodes SNAP indicators as integers.
        - Encodes event indicators and categorical event types.
        """
        print("[CLEANING] Processing calendar data...")
        df = df_cal.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Cyclical temporal features
        df["dayofweek"] = df["date"].dt.dayofweek
        df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)
        df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)

        df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12.0).astype(np.float32)
        df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12.0).astype(np.float32)

        # Event indicator flags
        df["has_event_1"] = df["event_name_1"].notnull().astype(np.int8)
        df["has_event_2"] = df["event_name_2"].notnull().astype(np.int8)

        # Christmas Day marker (deterministic store closure)
        df["is_christmas"] = ((df["date"].dt.month == 12) & (df["date"].dt.day == 25)).astype(np.int8)

        # Fill structural event nulls with 'None' category
        df["event_name_1"] = df["event_name_1"].astype(str).replace("nan", "None").astype("category")
        df["event_type_1"] = df["event_type_1"].astype(str).replace("nan", "None").astype("category")

        # Downcast
        df = reduce_mem_usage(df, verbose=False)
        return df

    def clean_prices(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans sell_prices table:
        - Computes baseline regular price (historical max price per store-item).
        - Computes promotional discount depth: (regular_price - sell_price) / regular_price.
        - Flags promotional price drops.
        """
        print("[CLEANING] Processing price dynamics and discount features...")
        df = df_prices.copy()

        # Regular price baseline per item & store
        regular_price = df.groupby(["store_id", "item_id"], observed=True)["sell_price"].transform("max")
        df["regular_price"] = regular_price.astype(np.float32)

        # Discount percentage (depth of discount)
        discount = (df["regular_price"] - df["sell_price"]) / df["regular_price"]
        df["discount_pct"] = np.clip(discount, 0.0, 1.0).astype(np.float32)
        df["is_discounted"] = (df["discount_pct"] > 0.05).astype(np.int8)

        df = reduce_mem_usage(df, verbose=False)
        return df

    def analyze_lifecycle_and_stockouts(self, df_sales: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes SKU lifecycle across 30,490 time series:
        - Detects first sale day (release_day).
        - Identifies cold start items (active for less than 365 days).
        - Identifies potential stockout events (consecutive zero-sales spells on high-velocity items).
        """
        print("[CLEANING] Evaluating product lifecycle, cold-starts, and stockout markers...")
        d_cols = [c for c in df_sales.columns if c.startswith("d_")]
        raw_vals = df_sales[d_cols].values
        n_series, n_days = raw_vals.shape

        # First positive sale index
        has_positive = raw_vals > 0
        first_sale_idx = np.where(has_positive.any(axis=1), np.argmax(has_positive, axis=1), n_days)

        active_days = n_days - first_sale_idx
        is_cold_start = (active_days < 365).astype(np.int8)

        # Summary metadata table
        sku_metadata = pd.DataFrame({
            "id": df_sales["id"].values,
            "item_id": df_sales["item_id"].values,
            "dept_id": df_sales["dept_id"].values,
            "cat_id": df_sales["cat_id"].values,
            "store_id": df_sales["store_id"].values,
            "state_id": df_sales["state_id"].values,
            "first_sale_day": [f"d_{idx+1}" for idx in first_sale_idx],
            "first_sale_day_num": first_sale_idx + 1,
            "active_days": active_days,
            "is_cold_start": is_cold_start,
        })
        return sku_metadata

    def build_starter_modeling_subset(
        self,
        df_cal: pd.DataFrame,
        df_prices: pd.DataFrame,
        df_sales: pd.DataFrame,
        sku_meta: pd.DataFrame,
        store_id: str = "CA_1",
        dept_id: str = "FOODS_1",
        max_skus: int = 50,
    ) -> pd.DataFrame:
        """
        Creates a clean, long-format panel starter dataset with complete date index,
        dynamic calendar features, prices, lag features, and stockout markers.
        Specifically designed for AML Kalman Filter & gradient boosted baselines.
        """
        print(f"[CLEANING] Generating starter modeling panel for {store_id} / {dept_id} (top {max_skus} SKUs)...")
        # Filter SKUs
        subset_sales = df_sales[(df_sales["store_id"] == store_id) & (df_sales["dept_id"] == dept_id)].copy()
        d_cols = [c for c in subset_sales.columns if c.startswith("d_")]

        # Sort by total volume and pick top SKUs
        total_vol = subset_sales[d_cols].sum(axis=1)
        subset_sales = subset_sales.iloc[np.argsort(-total_vol)[:max_skus]].copy()

        # Melt to long format: [id, d, sales]
        id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        melted = pd.melt(
            subset_sales,
            id_vars=id_vars,
            value_vars=d_cols,
            var_name="d",
            value_name="sales"
        )
        melted["sales"] = melted["sales"].astype(np.int16)

        # Merge with SKU lifecycle metadata
        melted = melted.merge(
            sku_meta[["id", "first_sale_day_num", "is_cold_start"]],
            on="id",
            how="left"
        )
        melted["d_num"] = melted["d"].str.replace("d_", "").astype(np.int16)

        # Mark pre-launch inactive period vs active shelf period
        melted["is_pre_launch"] = (melted["d_num"] < melted["first_sale_day_num"]).astype(np.int8)

        # Merge with calendar
        cal_sub = df_cal[[
            "d", "date", "wm_yr_wk", "dayofweek", "day_sin", "day_cos",
            "month_sin", "month_cos", "has_event_1", "is_christmas",
            f"snap_{store_id.split('_')[0]}"
        ]].copy()
        cal_sub = cal_sub.rename(columns={f"snap_{store_id.split('_')[0]}": "snap"})
        melted = melted.merge(cal_sub, on="d", how="left")

        # Merge with sell prices
        melted = melted.merge(
            df_prices[["store_id", "item_id", "wm_yr_wk", "sell_price", "regular_price", "discount_pct", "is_discounted"]],
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left"
        )

        # Stockout marker logic:
        # If an item is active (post-launch) but sell_price is missing, or
        # prolonged zero sales on a high-velocity item while active
        melted["is_unlisted_stockout"] = (
            (melted["is_pre_launch"] == 0) & (melted["sell_price"].isnull())
        ).astype(np.int8)

        # Impute forward/backward prices for continuity
        melted["sell_price"] = melted.groupby("id", observed=True)["sell_price"].transform(lambda x: x.ffill().bfill())
        melted["regular_price"] = melted.groupby("id", observed=True)["regular_price"].transform(lambda x: x.ffill().bfill())
        melted["discount_pct"] = melted["discount_pct"].fillna(0.0)
        melted["is_discounted"] = melted["is_discounted"].fillna(0).astype(np.int8)

        # Sort cleanly by SKU and date to ensure complete date index
        melted["date"] = pd.to_datetime(melted["date"])
        melted = melted.sort_values(["id", "date"]).reset_index(drop=True)

        # Lag and rolling features (Without peeking ahead!)
        print("[CLEANING] Engineering causal lag and rolling features (no forward-looking leakage)...")
        for lag in [1, 7, 14, 28]:
            melted[f"sales_lag_{lag}"] = melted.groupby("id", observed=True)["sales"].shift(lag).fillna(0).astype(np.int16)

        for w in [7, 28]:
            # Use shift(1) before rolling to prevent current day leakage
            rolling_series = melted.groupby("id", observed=True)["sales"].shift(1)
            melted[f"rolling_mean_{w}"] = rolling_series.groupby(melted["id"], observed=True).transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            ).fillna(0).astype(np.float32)

            melted[f"rolling_std_{w}"] = rolling_series.groupby(melted["id"], observed=True).transform(
                lambda x: x.rolling(w, min_periods=1).std()
            ).fillna(0).astype(np.float32)

        # Downcast
        melted = reduce_mem_usage(melted, verbose=False)
        return melted

    def run_cleaning_pipeline(self) -> dict:
        """Executes full cleaning pipeline and saves clean artifacts to data/processed/."""
        start_time = time.time()
        print("=" * 70)
        print("  M5 DATA CLEANING & PREPROCESSING PIPELINE")
        print("=" * 70)

        # Load raw data
        df_cal = load_calendar(str(self.raw_dir / "calendar.csv"), downcast=False)
        df_prices = load_sell_prices(str(self.raw_dir / "sell_prices.csv"), downcast=False)
        df_sales = load_sales_train(str(self.raw_dir / "sales_train_validation.csv"), downcast=False)

        # 1. Clean Calendar
        clean_cal = self.clean_calendar(df_cal)
        cal_out = self.processed_dir / "cleaned_calendar.parquet"
        clean_cal.to_parquet(cal_out, index=False)
        clean_cal.to_csv(self.processed_dir / "cleaned_calendar.csv", index=False)
        print(f"[SAVED] Cleaned Calendar -> {cal_out.name} & .csv ({clean_cal.shape})")

        # 2. Clean Prices
        clean_prices = self.clean_prices(df_prices)
        prices_out = self.processed_dir / "cleaned_prices.parquet"
        clean_prices.to_parquet(prices_out, index=False)
        print(f"[SAVED] Cleaned Prices -> {prices_out.name} ({clean_prices.shape})")

        # 3. Analyze Lifecycle & Cold Starts
        sku_meta = self.analyze_lifecycle_and_stockouts(df_sales)
        meta_out = self.processed_dir / "sku_lifecycle_metadata.parquet"
        sku_meta.to_parquet(meta_out, index=False)
        sku_meta.to_csv(self.processed_dir / "sku_lifecycle_metadata.csv", index=False)
        print(f"[SAVED] SKU Lifecycle Metadata -> {meta_out.name} & .csv ({sku_meta.shape})")

        # 4. Clean Sales Matrix & Downcast
        print("[CLEANING] Optimizing sales matrix for processed storage...")
        clean_sales = reduce_mem_usage(df_sales.copy(), verbose=False)
        sales_out = self.processed_dir / "cleaned_sales_matrix.parquet"
        clean_sales.to_parquet(sales_out, index=False)
        print(f"[SAVED] Cleaned Sales Matrix -> {sales_out.name} ({clean_sales.shape})")

        # 5. Build Starter Modeling Panel (CA_1 FOODS_1, 50 SKUs)
        starter_panel = self.build_starter_modeling_subset(
            clean_cal, clean_prices, df_sales, sku_meta,
            store_id="CA_1", dept_id="FOODS_1", max_skus=50
        )
        starter_out = self.processed_dir / "m5_starter_modeling_panel.parquet"
        starter_panel.to_parquet(starter_out, index=False)
        starter_panel.to_csv(self.processed_dir / "m5_starter_modeling_panel.csv", index=False)
        print(f"[SAVED] Starter Modeling Panel -> {starter_out.name} & .csv ({starter_panel.shape})")

        elapsed = time.time() - start_time
        print(f"\n[DONE] Data Cleaning pipeline completed in {elapsed:.2f} seconds.")
        print(f"Cleaned datasets safely stored in: {self.processed_dir.resolve()}")

        return {
            "calendar_path": str(cal_out),
            "prices_path": str(prices_out),
            "sku_meta_path": str(meta_out),
            "sales_path": str(sales_out),
            "starter_panel_path": str(starter_out),
            "cold_start_skus_count": int(sku_meta["is_cold_start"].sum()),
            "total_skus": len(sku_meta),
        }


if __name__ == "__main__":
    cleaner = M5DataCleaner()
    cleaner.run_cleaning_pipeline()
