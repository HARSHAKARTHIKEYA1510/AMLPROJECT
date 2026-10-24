"""
test_feature_engineering.py - Unit and Integration Tests for M5 Feature Engineering.

CSAI Student Project 5:
Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models.
"""

import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import M5FeatureEngineer


@pytest.fixture
def sample_sales_data():
    """Generates synthetic wide M5 sales data matching official schema."""
    n_days = 60
    n_series = 6
    d_cols = [f"d_{i}" for i in range(1, n_days + 1)]
    
    np.random.seed(42)
    data = {
        "id": [
            "FOODS_1_001_CA_1_validation",
            "FOODS_1_002_CA_1_validation",
            "HOBBIES_1_001_TX_1_validation",
            "HOBBIES_1_002_TX_1_validation",
            "HOUSEHOLD_1_001_WI_1_validation",
            "HOUSEHOLD_1_002_WI_1_validation",
        ],
        "item_id": [
            "FOODS_1_001", "FOODS_1_002",
            "HOBBIES_1_001", "HOBBIES_1_002",
            "HOUSEHOLD_1_001", "HOUSEHOLD_1_002",
        ],
        "dept_id": [
            "FOODS_1", "FOODS_1",
            "HOBBIES_1", "HOBBIES_1",
            "HOUSEHOLD_1", "HOUSEHOLD_1",
        ],
        "cat_id": [
            "FOODS", "FOODS",
            "HOBBIES", "HOBBIES",
            "HOUSEHOLD", "HOUSEHOLD",
        ],
        "store_id": [
            "CA_1", "CA_1",
            "TX_1", "TX_1",
            "WI_1", "WI_1",
        ],
        "state_id": [
            "CA", "CA",
            "TX", "TX",
            "WI", "WI",
        ],
    }
    
    for c in d_cols:
        data[c] = np.random.poisson(lam=3, size=n_series)

    return pd.DataFrame(data)


@pytest.fixture
def sample_calendar_data():
    """Generates synthetic calendar data matching official schema."""
    n_days = 60
    dates = pd.date_range("2011-01-29", periods=n_days, freq="D")
    
    df_cal = pd.DataFrame({
        "date": dates,
        "wm_yr_wk": 11101 + (np.arange(n_days) // 7),
        "weekday": dates.day_name(),
        "wday": (((dates.day_of_week if hasattr(dates, "day_of_week") else dates.dayofweek) + 2) % 7) + 1,
        "month": dates.month,
        "year": dates.year,
        "d": [f"d_{i}" for i in range(1, n_days + 1)],
        "event_name_1": [None] * n_days,
        "event_type_1": [None] * n_days,
        "event_name_2": [None] * n_days,
        "event_type_2": [None] * n_days,
        "snap_CA": np.random.choice([0, 1], size=n_days),
        "snap_TX": np.random.choice([0, 1], size=n_days),
        "snap_WI": np.random.choice([0, 1], size=n_days),
    })
    # Set an event for testing
    df_cal.loc[5, "event_name_1"] = "SuperBowl"
    df_cal.loc[5, "event_type_1"] = "Sporting"
    return df_cal


def test_series_selection_preserves_history(sample_sales_data):
    """Verifies that selecting a subset retains the entire history of each series."""
    engineer = M5FeatureEngineer()
    
    # Select 3 series using diverse strategy
    selected = engineer.select_series_subset(sample_sales_data, n_series=3, selection_strategy="first")
    
    assert len(selected) == 3
    # Check that all 60 day columns are preserved
    d_cols = [c for c in selected.columns if c.startswith("d_")]
    assert len(d_cols) == 60
    assert "d_1" in selected.columns
    assert "d_60" in selected.columns


def test_tidy_format_conversion(sample_sales_data, sample_calendar_data):
    """Verifies wide-to-long melting preserves keys and attaches chronological dates."""
    engineer = M5FeatureEngineer()
    selected = sample_sales_data.head(2)
    
    df_tidy = engineer.to_tidy_format(selected, sample_calendar_data)
    
    # 2 series x 60 days = 120 rows
    assert len(df_tidy) == 120
    assert "demand" in df_tidy.columns
    assert "date" in df_tidy.columns
    assert pd.api.types.is_datetime64_any_dtype(df_tidy["date"])
    # Verify all identifiers are preserved
    for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        assert col in df_tidy.columns
        assert df_tidy[col].isnull().sum() == 0


def test_date_continuity_and_missing_reporting(sample_sales_data, sample_calendar_data):
    """Verifies date continuity audit and detection of missing daily dates."""
    engineer = M5FeatureEngineer()
    df_tidy = engineer.to_tidy_format(sample_sales_data.head(2), sample_calendar_data)
    
    # 1. Normal continuous data
    _, report = engineer.validate_date_continuity(df_tidy)
    assert report["total_missing_dates"] == 0
    assert len(report["series_with_gaps"]) == 0
    
    # 2. Artificially drop 2 days from the first series
    tampered = df_tidy.copy()
    first_id = tampered["id"].iloc[0]
    drop_indices = tampered[(tampered["id"] == first_id) & (tampered["d"].isin(["d_5", "d_6"]))].index
    tampered = tampered.drop(drop_indices).reset_index(drop=True)
    
    reindexed, gap_report = engineer.validate_date_continuity(tampered)
    assert gap_report["total_missing_dates"] == 2
    assert len(gap_report["series_with_gaps"]) == 1
    assert gap_report["series_with_gaps"][0]["series_id"] == first_id
    # Reindexing restores the grid without creating stockout labels
    assert len(reindexed) == 120
    assert "is_stockout" not in reindexed.columns


def test_lag_features(sample_sales_data, sample_calendar_data):
    """Verifies lag features correctly reference previous days."""
    engineer = M5FeatureEngineer(lags=[1, 7])
    df_tidy = engineer.to_tidy_format(sample_sales_data.head(1), sample_calendar_data)
    
    df_features = engineer.create_lag_features(df_tidy, lags=[1, 7])
    
    # lag_1 on day 0 should be NaN, day 1 should equal demand of day 0
    assert pd.isna(df_features.loc[0, "lag_1"])
    assert df_features.loc[1, "lag_1"] == df_features.loc[0, "demand"]
    assert df_features.loc[7, "lag_7"] == df_features.loc[0, "demand"]
    assert pd.isna(df_features.loc[6, "lag_7"])


def test_no_target_leakage_in_rolling_features(sample_sales_data, sample_calendar_data):
    """
    MATHEMATICAL LEAKAGE PROOF:
    Verifies that rolling statistics for day t use shift(1) and DO NOT include day t demand.
    """
    engineer = M5FeatureEngineer(rolling_windows=[7])
    df_tidy = engineer.to_tidy_format(sample_sales_data.head(1), sample_calendar_data)
    
    df_features = engineer.create_rolling_features(df_tidy, windows=[7])
    
    # For day index 7 (the 8th day):
    # rolling_mean_7 must equal the average of days 0..6 (strictly previous 7 days)
    expected_mean = df_tidy.loc[0:6, "demand"].mean()
    expected_std = df_tidy.loc[0:6, "demand"].std()
    
    actual_mean = df_features.loc[7, "rolling_mean_7"]
    actual_std = df_features.loc[7, "rolling_std_7"]
    
    assert np.isclose(actual_mean, expected_mean)
    assert np.isclose(actual_std, expected_std)
    
    # Perturb day 7's demand and ensure rolling_mean_7 on day 7 does NOT change
    df_perturbed = df_tidy.copy()
    df_perturbed.loc[7, "demand"] = df_perturbed.loc[7, "demand"] + 1000.0
    df_features_perturbed = engineer.create_rolling_features(df_perturbed, windows=[7])
    
    assert np.isclose(
        df_features_perturbed.loc[7, "rolling_mean_7"],
        actual_mean
    ), "TARGET LEAKAGE DETECTED: Modifying demand at day t altered rolling feature at day t!"


def test_calendar_and_event_features(sample_sales_data, sample_calendar_data):
    """Verifies calendar, event, and SNAP policy feature creation."""
    engineer = M5FeatureEngineer()
    df_tidy = engineer.to_tidy_format(sample_sales_data.head(2), sample_calendar_data)
    
    df_cal = engineer.create_calendar_features(df_tidy)
    
    assert "day_of_week" in df_cal.columns
    assert "is_weekend" in df_cal.columns
    assert "day_sin" in df_cal.columns
    assert "day_cos" in df_cal.columns
    assert "month_sin" in df_cal.columns
    assert "month_cos" in df_cal.columns
    assert "has_event_1" in df_cal.columns
    assert "snap_active" in df_cal.columns
    
    # Weekend verification: dayofweek 5 (Saturday) or 6 (Sunday) -> is_weekend == 1
    weekends = df_cal[df_cal["day_of_week"].isin([5, 6])]
    assert (weekends["is_weekend"] == 1).all()
    weekdays = df_cal[~df_cal["day_of_week"].isin([5, 6])]
    assert (weekdays["is_weekend"] == 0).all()
    
    # Event verification: SuperBowl is on day 5 (index 5)
    event_day = df_cal[df_cal["event_name_1"] == "SuperBowl"]
    assert len(event_day) > 0
    assert (event_day["has_event_1"] == 1).all()
    assert (event_day["is_event_day"] == 1).all()


def test_original_demand_integrity(sample_sales_data, sample_calendar_data):
    """Ensures feature generation does not silently mutate original demand values."""
    engineer = M5FeatureEngineer(lags=[1, 7], rolling_windows=[7])
    df_tidy = engineer.to_tidy_format(sample_sales_data.head(2), sample_calendar_data)
    original_demand = df_tidy["demand"].copy()
    
    df_features = engineer.create_lag_features(df_tidy)
    df_features = engineer.create_rolling_features(df_features)
    df_features = engineer.create_calendar_features(df_features)
    
    pd.testing.assert_series_equal(original_demand, df_features["demand"], check_names=True)


def test_chronological_split(sample_sales_data, sample_calendar_data):
    """Verifies strictly chronological train/validation splitting with a 28-day horizon."""
    engineer = M5FeatureEngineer(lags=[1, 7], rolling_windows=[7], validation_days=28)
    df_tidy = engineer.to_tidy_format(sample_sales_data.head(2), sample_calendar_data)
    df_features = engineer.create_lag_features(df_tidy)
    df_features = engineer.create_rolling_features(df_features)
    df_features = engineer.create_calendar_features(df_features)
    
    _, df_clean, _ = engineer.handle_missing_features(df_features, drop_warmup=True)
    train_df, val_df, split_meta = engineer.chronological_split(df_clean, val_days=28)
    
    # 1. Strict time ordering
    assert train_df["date"].max() < val_df["date"].min()
    
    # 2. Validation horizon is exactly 28 days
    assert split_meta["val_days_count"] == 28
    val_counts = val_df.groupby("id")["date"].count()
    for count in val_counts:
        assert count == 28
        
    # 3. No series lost
    assert train_df["id"].nunique() == val_df["id"].nunique()


def test_end_to_end_pipeline(tmp_path, sample_sales_data, sample_calendar_data):
    """Integration test: validates end-to-end execution and output persistence."""
    raw_sales_file = tmp_path / "sales.csv"
    raw_cal_file = tmp_path / "calendar.csv"
    proc_dir = tmp_path / "processed"
    
    sample_sales_data.to_csv(raw_sales_file, index=False)
    sample_calendar_data.to_csv(raw_cal_file, index=False)
    
    engineer = M5FeatureEngineer(
        raw_sales_path=str(raw_sales_file),
        calendar_path=str(raw_cal_file),
        processed_dir=str(proc_dir),
        lags=[1, 7],
        rolling_windows=[7],
        validation_days=14,
        drop_warmup=True,
    )
    
    metadata = engineer.run_pipeline(n_series=4, selection_strategy="first")
    
    assert (proc_dir / "train_features.csv").exists()
    assert (proc_dir / "val_features.csv").exists()
    assert (proc_dir / "feature_metadata.json").exists()
    assert metadata["feature_engineering"]["target_leakage_prevented"] is True
    assert metadata["split_info"]["val_days_count"] == 14
