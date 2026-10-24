"""
feature_engineering.py - Modular Feature Engineering Pipeline for M5 Demand Forecasting.

CSAI Student Project 5:
Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models.

This module provides the M5FeatureEngineer class, which converts wide-format
M5 sales data and calendar records into tidy, leak-free, temporally indexed
feature panels with chronological train/validation splitting.
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader import load_calendar, load_sales_train, reduce_mem_usage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("M5FeatureEngineer")


class M5FeatureEngineer:
    """
    Feature Engineering Engine for M5 Inventory Demand Forecasting.
    
    Transforms wide M5 sales records into tidy panel representations,
    constructs causal lag and rolling statistics without target leakage,
    attaches calendar and policy features, and produces strictly chronological
    train/validation splits.
    """

    def __init__(
        self,
        raw_sales_path: str = "data/raw/sales_train_validation.csv",
        calendar_path: str = "data/raw/calendar.csv",
        processed_dir: str = "data/processed",
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        validation_days: int = 28,
        drop_warmup: bool = True,
    ):
        self.raw_sales_path = Path(raw_sales_path)
        self.calendar_path = Path(calendar_path)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.lags = lags if lags is not None else [1, 7, 28]
        self.rolling_windows = rolling_windows if rolling_windows is not None else [7, 28]
        self.validation_days = validation_days
        self.drop_warmup = drop_warmup

        # Maximum history required for features (warmup horizon)
        self.max_warmup_days = max(max(self.lags), max(self.rolling_windows))

        self.id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    def load_data(
        self,
        sales_nrows: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads sales and calendar data with memory optimization.
        Reuses existing data loader functions from src.data_loader.
        """
        if not self.raw_sales_path.exists():
            raise FileNotFoundError(f"Sales dataset not found at {self.raw_sales_path.resolve()}")
        if not self.calendar_path.exists():
            # Check for cleaned calendar fallback
            cleaned_cal = self.processed_dir / "cleaned_calendar.csv"
            if cleaned_cal.exists():
                self.calendar_path = cleaned_cal
            else:
                raise FileNotFoundError(f"Calendar dataset not found at {self.calendar_path.resolve()}")

        logger.info(f"Loading calendar from {self.calendar_path}...")
        df_calendar = load_calendar(str(self.calendar_path), downcast=False)

        logger.info(f"Loading sales matrix from {self.raw_sales_path} (nrows={sales_nrows})...")
        df_sales = load_sales_train(str(self.raw_sales_path), downcast=False, nrows=sales_nrows)

        logger.info(f"Loaded calendar: {df_calendar.shape}, sales: {df_sales.shape}")
        return df_sales, df_calendar

    def select_series_subset(
        self,
        df_sales: pd.DataFrame,
        n_series: Optional[int] = 50,
        store_ids: Optional[List[str]] = None,
        cat_ids: Optional[List[str]] = None,
        dept_ids: Optional[List[str]] = None,
        series_ids: Optional[List[str]] = None,
        selection_strategy: str = "diverse",
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Selects a configurable, manageable subset of product-store series,
        preserving the complete historical time series (all d_1..d_T columns)
        for each selected series.
        
        Selection strategies:
        - 'diverse': Stratified balanced sampling across categories and departments.
        - 'top_volume': Series with the highest total historical unit demand.
        - 'sample': Uniform random sample of series.
        - 'first': Top n_series records in order.
        """
        logger.info(f"Subsetting series (initial pool: {len(df_sales):,} SKUs)...")
        sub = df_sales.copy()

        # Filter by store, category, department, or exact series_ids
        if store_ids:
            sub = sub[sub["store_id"].isin(store_ids)]
            logger.info(f"Filtered by stores {store_ids}: {len(sub):,} series remaining")
        if cat_ids:
            sub = sub[sub["cat_id"].isin(cat_ids)]
            logger.info(f"Filtered by categories {cat_ids}: {len(sub):,} series remaining")
        if dept_ids:
            sub = sub[sub["dept_id"].isin(dept_ids)]
            logger.info(f"Filtered by departments {dept_ids}: {len(sub):,} series remaining")
        if series_ids:
            sub = sub[sub["id"].isin(series_ids)]
            logger.info(f"Filtered by series_ids ({len(series_ids)} requested): {len(sub):,} series remaining")

        if len(sub) == 0:
            raise ValueError("No series matched the specified filtering criteria.")

        # Limit to n_series if specified and smaller than available pool
        if n_series is not None and n_series < len(sub):
            d_cols = [c for c in sub.columns if c.startswith("d_")]
            if selection_strategy == "top_volume":
                sub["total_demand"] = sub[d_cols].sum(axis=1)
                sub = sub.sort_values(by="total_demand", ascending=False).head(n_series)
                sub = sub.drop(columns=["total_demand"])
            elif selection_strategy == "diverse":
                # Balanced representation across dept_id
                groups = sub.groupby("dept_id", observed=True)
                per_group = max(1, n_series // max(1, len(groups)))
                sampled = []
                for _, grp in groups:
                    k = min(len(grp), per_group)
                    sampled.append(grp.sample(n=k, random_state=random_state))
                sub_sampled = pd.concat(sampled, ignore_index=True)
                if len(sub_sampled) > n_series:
                    sub = sub_sampled.sample(n=n_series, random_state=random_state)
                elif len(sub_sampled) < n_series:
                    remaining = sub[~sub["id"].isin(sub_sampled["id"])]
                    fill_k = min(len(remaining), n_series - len(sub_sampled))
                    sub = pd.concat([sub_sampled, remaining.sample(n=fill_k, random_state=random_state)], ignore_index=True)
                else:
                    sub = sub_sampled
            elif selection_strategy == "sample":
                sub = sub.sample(n=n_series, random_state=random_state)
            elif selection_strategy == "first":
                sub = sub.head(n_series)
            else:
                raise ValueError(f"Unknown selection strategy '{selection_strategy}'.")

        sub = sub.reset_index(drop=True)
        d_cols = [c for c in sub.columns if c.startswith("d_")]
        logger.info(
            f"Selected {len(sub)} series across {sub['store_id'].nunique()} store(s) "
            f"and {sub['cat_id'].nunique()} category(ies), preserving all {len(d_cols)} historical days."
        )
        return sub

    def to_tidy_format(
        self,
        df_selected: pd.DataFrame,
        df_calendar: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Converts selected wide sales matrix into tidy long-format time series.
        Merges daily calendar date attributes onto the series observations.
        """
        logger.info("Reshaping wide sales table into tidy time-series format...")
        d_cols = [c for c in df_selected.columns if c.startswith("d_")]
        
        # Melt wide day columns into rows
        df_tidy = pd.melt(
            df_selected,
            id_vars=self.id_cols,
            value_vars=d_cols,
            var_name="d",
            value_name="demand"
        )

        # Merge with calendar to attach chronological date
        cal_cols = ["d", "date"]
        # Include additional calendar attributes if present
        extra_cal = [
            "wm_yr_wk", "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI"
        ]
        for col in extra_cal:
            if col in df_calendar.columns and col not in cal_cols:
                cal_cols.append(col)

        df_cal_sub = df_calendar[cal_cols].copy()
        df_tidy = df_tidy.merge(df_cal_sub, on="d", how="left")

        # Parse date and sort chronologically per series
        df_tidy["date"] = pd.to_datetime(df_tidy["date"])
        df_tidy["demand"] = pd.to_numeric(df_tidy["demand"], errors="coerce").fillna(0).clip(lower=0)
        df_tidy = df_tidy.sort_values(by=["id", "date"]).reset_index(drop=True)

        logger.info(f"Tidy panel constructed: {len(df_tidy):,} rows across {df_tidy['id'].nunique()} series.")
        return df_tidy

    def validate_date_continuity(
        self,
        df_tidy: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Ensures a complete, continuous daily date index for each series.
        Reports any missing dates without inventing stockout labels.
        """
        logger.info("Auditing temporal continuity and daily date index integrity...")
        min_date = df_tidy["date"].min()
        max_date = df_tidy["date"].max()
        expected_range = pd.date_range(min_date, max_date, freq="D")
        n_expected_days = len(expected_range)

        continuity_report = {
            "min_date": str(min_date.date()),
            "max_date": str(max_date.date()),
            "expected_days_per_series": n_expected_days,
            "total_series": int(df_tidy["id"].nunique()),
            "series_with_gaps": [],
            "total_missing_dates": 0,
        }

        # Check date count per series
        counts_per_series = df_tidy.groupby("id", observed=True)["date"].count()
        gaps_detected = False

        for series_id, actual_count in counts_per_series.items():
            if actual_count != n_expected_days:
                gaps_detected = True
                series_dates = set(df_tidy[df_tidy["id"] == series_id]["date"])
                missing_dates = sorted(list(set(expected_range) - series_dates))
                continuity_report["series_with_gaps"].append({
                    "series_id": str(series_id),
                    "missing_count": len(missing_dates),
                    "missing_dates": [str(d.date()) for d in missing_dates[:5]]
                })
                continuity_report["total_missing_dates"] += len(missing_dates)
                logger.warning(
                    f"Series '{series_id}' has {len(missing_dates)} missing dates in range {min_date.date()} to {max_date.date()}."
                )

        if not gaps_detected:
            logger.info(
                f"[CONTINUITY PASS] All {continuity_report['total_series']} series have unbroken, "
                f"complete daily histories of {n_expected_days} consecutive days (0 missing dates)."
            )
            return df_tidy, continuity_report

        # If gaps are present, reindex per series to ensure complete grid
        logger.info("Reindexing series to complete daily date grid (missing observations preserved as NaN)...")
        all_series = df_tidy["id"].unique()
        full_grid = pd.MultiIndex.from_product(
            [all_series, expected_range],
            names=["id", "date"]
        ).to_frame().reset_index(drop=True)

        merged = full_grid.merge(df_tidy, on=["id", "date"], how="left")
        # Propagate static identifiers
        for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
            merged[col] = merged.groupby("id", observed=True)[col].transform("first")

        merged = merged.sort_values(by=["id", "date"]).reset_index(drop=True)
        return merged, continuity_report

    def create_lag_features(
        self,
        df: pd.DataFrame,
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Creates historical lag features grouped by series.
        All lags represent past observations strictly prior to day t.
        """
        use_lags = lags if lags is not None else self.lags
        logger.info(f"Computing lag features: {use_lags}...")

        # Ensure correct chronological order
        df = df.sort_values(by=["id", "date"]).copy()

        for k in use_lags:
            col_name = f"lag_{k}"
            df[col_name] = df.groupby("id", observed=True)["demand"].shift(k)

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Creates rolling mean and standard deviation features over historical demand.
        
        CRITICAL TARGET LEAKAGE PREVENTION:
        The demand series is explicitly shifted by 1 day prior to computing the
        rolling window statistics. Thus, the feature for day t uses exclusively
        observations from {t-1, t-2, ..., t-W}, guaranteeing zero target leakage.
        """
        use_windows = windows if windows is not None else self.rolling_windows
        logger.info(f"Computing rolling features for windows {use_windows} (pre-shifted by 1 to prevent leakage)...")

        df = df.sort_values(by=["id", "date"]).copy()

        # Shift demand by 1 day per series
        shifted = df.groupby("id", observed=True)["demand"].shift(1)

        for w in use_windows:
            mean_col = f"rolling_mean_{w}"
            std_col = f"rolling_std_{w}"

            df[mean_col] = shifted.groupby(df["id"], observed=True).transform(
                lambda s: s.rolling(window=w, min_periods=w).mean()
            )
            df[std_col] = shifted.groupby(df["id"], observed=True).transform(
                lambda s: s.rolling(window=w, min_periods=w).std()
            )

        return df

    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs calendar, cyclical, event, and SNAP policy indicators.
        """
        logger.info("Computing temporal calendar features, cyclical encodings, and event flags...")
        df = df.copy()

        # Core calendar components
        dow = df["date"].dt.day_of_week if hasattr(df["date"].dt, "day_of_week") else df["date"].dt.dayofweek
        df["day_of_week"] = dow.astype(np.int8)  # 0 = Monday, 6 = Sunday
        df["month"] = df["date"].dt.month.astype(np.int8)
        df["year"] = df["date"].dt.year.astype(np.int16)
        df["day"] = df["date"].dt.day.astype(np.int8)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
        df["quarter"] = df["date"].dt.quarter.astype(np.int8)
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int8)

        # Cyclical temporal features
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0).astype(np.float32)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0).astype(np.float32)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)

        # Holiday & Event Indicators
        for ev_col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            if ev_col in df.columns:
                df[ev_col] = df[ev_col].astype(object).fillna("None").astype(str)
            else:
                df[ev_col] = "None"

        df["has_event_1"] = (df["event_name_1"] != "None").astype(np.int8)
        df["has_event_2"] = (df["event_name_2"] != "None").astype(np.int8)
        df["is_event_day"] = ((df["has_event_1"] == 1) | (df["has_event_2"] == 1)).astype(np.int8)
        df["is_christmas"] = ((df["month"] == 12) & (df["day"] == 25)).astype(np.int8)

        # State-aware SNAP policy indicator
        # Map row's state_id to corresponding SNAP column (snap_CA, snap_TX, snap_WI)
        for snap_c in ["snap_CA", "snap_TX", "snap_WI"]:
            if snap_c in df.columns:
                df[snap_c] = pd.to_numeric(df[snap_c], errors="coerce").fillna(0).astype(np.int8)
            else:
                df[snap_c] = 0

        df["snap_active"] = np.where(
            df["state_id"] == "CA", df["snap_CA"],
            np.where(df["state_id"] == "TX", df["snap_TX"], df["snap_WI"])
        ).astype(np.int8)

        return df

    def verify_no_target_leakage(self, df: pd.DataFrame) -> bool:
        """
        Rigorous programmatic check: guarantees that modifying demand on day t
        does not alter any lag or rolling features on day t.
        """
        logger.info("Executing mathematical verification against target leakage...")
        sample_series = df["id"].iloc[0]
        sub = df[df["id"] == sample_series].copy().sort_values("date").reset_index(drop=True)
        
        # Test row at index 50 (well past initial warmup)
        test_idx = min(50, len(sub) - 1)
        original_demand = sub.loc[test_idx, "demand"]
        
        feature_cols = [c for c in sub.columns if c.startswith("lag_") or c.startswith("rolling_")]
        pre_values = sub.loc[test_idx, feature_cols].to_dict()

        # Perturb test index demand
        sub.loc[test_idx, "demand"] = original_demand + 999.0

        # Recompute features on perturbed sample
        sub_recomputed = self.create_lag_features(sub, self.lags)
        sub_recomputed = self.create_rolling_features(sub_recomputed, self.rolling_windows)
        post_values = sub_recomputed.loc[test_idx, feature_cols].to_dict()

        leaked = []
        for feat in feature_cols:
            val_pre = pre_values[feat]
            val_post = post_values[feat]
            # Handle NaN equality
            if pd.isna(val_pre) and pd.isna(val_post):
                continue
            if not np.isclose(val_pre, val_post, equal_nan=True):
                leaked.append(feat)

        if leaked:
            raise RuntimeError(f"Target leakage detected in features: {leaked}")

        logger.info("[LEAKAGE CHECK PASSED] Features on day t are strictly independent of day t demand.")
        return True

    def handle_missing_features(
        self,
        df: pd.DataFrame,
        drop_warmup: Optional[bool] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Handles initial missing lag and rolling values without modifying original demand.
        
        Due to the backward-looking nature of lag_28 and rolling_28, the first 28
        observations per series lack historical context and evaluate to NaN.
        
        When drop_warmup=True:
        The initial uncomputable warm-up period is trimmed for model training.
        The original demand values for the remaining observations are completely intact.
        Both the full panel (with NaNs preserved) and clean training panel are provided.
        """
        should_drop = drop_warmup if drop_warmup is not None else self.drop_warmup
        logger.info(f"Auditing missing feature values (drop_warmup={should_drop})...")

        df_full = df.copy()
        feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("rolling_")]

        null_counts_before = df_full[feature_cols].isnull().sum().to_dict()
        rows_before = len(df_full)

        if should_drop:
            # Drop initial warmup days where lag/rolling features are NaN
            df_clean = df_full.dropna(subset=feature_cols).reset_index(drop=True)
            rows_after = len(df_clean)
            warmup_dropped = rows_before - rows_after
            null_counts_after = df_clean[feature_cols].isnull().sum().to_dict()
            logger.info(
                f"Trimmed {warmup_dropped:,} initial warmup records across {df['id'].nunique()} series. "
                f"Remaining model-ready observations: {rows_after:,} (0 NaN feature values)."
            )
        else:
            df_clean = df_full
            warmup_dropped = 0
            null_counts_after = null_counts_before

        summary = {
            "drop_warmup_applied": should_drop,
            "max_warmup_days": self.max_warmup_days,
            "rows_before_trimming": rows_before,
            "warmup_rows_dropped": warmup_dropped,
            "rows_after_trimming": len(df_clean),
            "missing_counts_before": {k: int(v) for k, v in null_counts_before.items()},
            "missing_counts_after": {k: int(v) for k, v in null_counts_after.items()},
        }
        return df_full, df_clean, summary

    def chronological_split(
        self,
        df: pd.DataFrame,
        val_days: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Splits data chronologically with a strict validation horizon (default: 28 days).
        
        Guarantees:
        1. max(train_date) < min(val_date)
        2. Exactly val_days observations per series in the validation set.
        3. Never shuffles or randomly samples time series observations.
        """
        horizon = val_days if val_days is not None else self.validation_days
        logger.info(f"Performing strictly chronological train/validation split (horizon: {horizon} days)...")

        df = df.sort_values(by=["id", "date"]).copy()
        max_date = df["date"].max()
        val_start_date = max_date - pd.Timedelta(days=horizon - 1)

        train_df = df[df["date"] < val_start_date].copy().reset_index(drop=True)
        val_df = df[df["date"] >= val_start_date].copy().reset_index(drop=True)

        # Integrity validations
        if len(train_df) == 0:
            raise ValueError("Training split is empty after chronological partitioning.")
        if len(val_df) == 0:
            raise ValueError("Validation split is empty after chronological partitioning.")

        train_max_date = train_df["date"].max()
        val_min_date = val_df["date"].min()
        val_max_date = val_df["date"].max()

        if train_max_date >= val_min_date:
            raise RuntimeError(
                f"Chronological ordering violation! train_max={train_max_date} >= val_min={val_min_date}"
            )

        # Check validation day count per series
        val_series_counts = val_df.groupby("id", observed=True)["date"].count()
        val_series_discrepancies = (val_series_counts != horizon).sum()
        if val_series_discrepancies > 0:
            logger.warning(f"{val_series_discrepancies} series in validation set do not have exactly {horizon} days.")

        split_meta = {
            "validation_horizon_days": horizon,
            "train_start_date": str(train_df["date"].min().date()),
            "train_end_date": str(train_max_date.date()),
            "val_start_date": str(val_min_date.date()),
            "val_end_date": str(val_max_date.date()),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "train_days_count": int((train_max_date - train_df["date"].min()).days + 1),
            "val_days_count": int((val_max_date - val_min_date).days + 1),
            "total_series": int(df["id"].nunique()),
        }

        logger.info(
            f"[SPLIT COMPLETE] Train: {split_meta['train_start_date']} -> {split_meta['train_end_date']} "
            f"({len(train_df):,} rows) | Val: {split_meta['val_start_date']} -> {split_meta['val_end_date']} "
            f"({len(val_df):,} rows, {horizon} days per series)."
        )
        return train_df, val_df, split_meta

    def save_artifacts(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        metadata: Dict[str, Any],
        full_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Path]:
        """
        Saves processed train/validation feature sets and JSON metadata under data/processed/.
        """
        logger.info(f"Saving feature-engineered artifacts to {self.processed_dir.resolve()}...")
        saved_paths = {}

        train_path = self.processed_dir / "train_features.csv"
        val_path = self.processed_dir / "val_features.csv"
        meta_path = self.processed_dir / "feature_metadata.json"

        # Downcast to save disk and memory footprint
        logger.info("Applying memory downcasting prior to saving...")
        train_df_opt = reduce_mem_usage(train_df, verbose=False)
        val_df_opt = reduce_mem_usage(val_df, verbose=False)

        train_df_opt.to_csv(train_path, index=False)
        val_df_opt.to_csv(val_path, index=False)
        saved_paths["train_features"] = train_path
        saved_paths["val_features"] = val_path

        if full_df is not None:
            full_path = self.processed_dir / "selected_features_full.csv"
            full_df_opt = reduce_mem_usage(full_df, verbose=False)
            full_df_opt.to_csv(full_path, index=False)
            saved_paths["full_features"] = full_path
            logger.info(f"[SAVED] {full_path.name} ({len(full_df_opt):,} rows, {full_path.stat().st_size / (1024*1024):.2f} MB)")

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        saved_paths["metadata"] = meta_path

        logger.info(f"[SAVED] {train_path.name} ({len(train_df_opt):,} rows, {train_path.stat().st_size / (1024*1024):.2f} MB)")
        logger.info(f"[SAVED] {val_path.name} ({len(val_df_opt):,} rows, {val_path.stat().st_size / (1024*1024):.2f} MB)")
        logger.info(f"[SAVED] {meta_path.name}")

        return saved_paths

    def run_pipeline(
        self,
        n_series: Optional[int] = 50,
        store_ids: Optional[List[str]] = None,
        cat_ids: Optional[List[str]] = None,
        dept_ids: Optional[List[str]] = None,
        series_ids: Optional[List[str]] = None,
        selection_strategy: str = "diverse",
        sales_nrows: Optional[int] = None,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end feature engineering pipeline:
        1. Load data
        2. Subset series (preserving complete history)
        3. Tidy reshaping
        4. Date continuity audit
        5. Lag feature generation
        6. Target-leak-free rolling statistics
        7. Calendar, event, and SNAP policy feature creation
        8. Leakage mathematical verification
        9. Warmup missing value handling
        10. Chronological train/validation splitting
        11. Artifact and metadata persistence
        """
        start_time = time.time()
        logger.info("=" * 70)
        logger.info("  M5 FEATURE ENGINEERING PIPELINE (NO TARGET LEAKAGE)")
        logger.info("=" * 70)

        # 1. Load Data
        df_sales, df_calendar = self.load_data(sales_nrows=sales_nrows)

        # 2. Select series subset preserving full history
        df_selected = self.select_series_subset(
            df_sales,
            n_series=n_series,
            store_ids=store_ids,
            cat_ids=cat_ids,
            dept_ids=dept_ids,
            series_ids=series_ids,
            selection_strategy=selection_strategy,
            random_state=random_state,
        )

        # 3. Tidy conversion
        df_tidy = self.to_tidy_format(df_selected, df_calendar)

        # 4. Continuity check
        df_continuous, continuity_report = self.validate_date_continuity(df_tidy)

        # 5. Lag features
        df_features = self.create_lag_features(df_continuous, self.lags)

        # 6. Rolling features (shift 1 to prevent leakage)
        df_features = self.create_rolling_features(df_features, self.rolling_windows)

        # 7. Calendar and policy features
        df_features = self.create_calendar_features(df_features)

        # 8. Automated Leakage Check
        self.verify_no_target_leakage(df_features)

        # 9. Handle missing values
        df_full, df_clean, missing_summary = self.handle_missing_features(
            df_features, drop_warmup=self.drop_warmup
        )

        # 10. Chronological split
        train_df, val_df, split_meta = self.chronological_split(
            df_clean, val_days=self.validation_days
        )

        # 11. Compile metadata
        feature_columns = [
            c for c in df_clean.columns
            if c not in self.id_cols and c not in ["date", "d", "demand"]
        ]
        
        metadata = {
            "pipeline_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset_info": {
                "total_series_selected": int(df_selected["id"].nunique()),
                "total_days_historical": int(len([c for c in df_selected.columns if c.startswith("d_")])),
                "selected_stores": sorted(list(df_selected["store_id"].unique())),
                "selected_categories": sorted(list(df_selected["cat_id"].unique())),
                "selected_departments": sorted(list(df_selected["dept_id"].unique())),
                "sample_series_ids": list(df_selected["id"].head(5)),
            },
            "split_info": split_meta,
            "missing_values_audit": missing_summary,
            "temporal_continuity": continuity_report,
            "feature_engineering": {
                "target_column": "demand",
                "id_columns": self.id_cols,
                "date_columns": ["date", "d"],
                "lags_computed": self.lags,
                "rolling_windows_computed": self.rolling_windows,
                "target_leakage_prevented": True,
                "all_feature_columns": feature_columns,
                "total_feature_count": len(feature_columns),
            },
            "data_shapes": {
                "tidy_full_shape": list(df_full.shape),
                "model_ready_shape": list(df_clean.shape),
                "train_shape": list(train_df.shape),
                "val_shape": list(val_df.shape),
            },
            "execution_time_seconds": round(time.time() - start_time, 2),
        }

        # 12. Save artifacts
        saved_paths = self.save_artifacts(train_df, val_df, metadata, full_df=df_full)
        metadata["saved_files"] = {k: str(v.resolve()) for k, v in saved_paths.items()}

        elapsed = time.time() - start_time
        logger.info(f"\n[PIPELINE COMPLETE] Executed successfully in {elapsed:.2f} seconds.")
        logger.info(f"Train Shape: {train_df.shape} | Val Shape: {val_df.shape}")
        logger.info(f"Validation Horizon: {split_meta['val_start_date']} to {split_meta['val_end_date']}")

        return metadata


if __name__ == "__main__":
    engineer = M5FeatureEngineer()
    engineer.run_pipeline(n_series=50, selection_strategy="diverse")
