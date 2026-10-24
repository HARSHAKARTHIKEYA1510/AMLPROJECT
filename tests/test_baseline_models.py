"""
test_baseline_models.py - Unit and Integration Tests for AML Baseline Forecasters.

CSAI Student Project 5:
Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_models import (
    compute_forecasting_metrics,
    SeasonalNaiveForecaster,
    GBDTForecaster,
    M5BaselinePipeline
)


def test_wape_zero_denominator():
    """Verifies that WAPE safely handles zero-sum actual demand without dividing by zero."""
    # Case 1: All actuals are 0, all predictions are 0 (perfect forecast of zero demand)
    y_true_zeros = np.array([0.0, 0.0, 0.0])
    y_pred_zeros = np.array([0.0, 0.0, 0.0])
    metrics_zero = compute_forecasting_metrics(y_true_zeros, y_pred_zeros)
    assert metrics_zero["wape"] == 0.0
    assert metrics_zero["mae"] == 0.0

    # Case 2: All actuals are 0, but predictions are non-zero
    y_pred_nonzero = np.array([1.0, 2.0, 1.0])
    metrics_nonzero = compute_forecasting_metrics(y_true_zeros, y_pred_nonzero)
    assert metrics_nonzero["wape"] is None  # Handled safely as None/NaN


def test_metrics_accuracy():
    """Verifies numerical precision of MAE, RMSE, and WAPE against known values."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 36.0])

    # abs errors: [2, 2, 3, 4] -> sum = 11, mean = 2.75
    # sq errors:  [4, 4, 9, 16] -> sum = 33, mean = 8.25 -> sqrt = 2.87228
    # sum actual: 100
    # WAPE = 11 / 100 = 0.11 (11.0%)
    metrics = compute_forecasting_metrics(y_true, y_pred)
    assert metrics["mae"] == 2.75
    assert np.isclose(metrics["rmse"], np.sqrt(8.25), atol=1e-4)
    assert metrics["wape"] == 0.11


@pytest.fixture
def synthetic_train_val_data():
    """Generates synthetic train (35 days) and val (14 days) datasets."""
    n_series = 2
    train_days = 35
    val_days = 14

    train_dates = pd.date_range("2016-01-01", periods=train_days, freq="D")
    val_dates = pd.date_range(train_dates[-1] + pd.Timedelta(days=1), periods=val_days, freq="D")

    records_train = []
    for s_idx, sid in enumerate(["FOODS_1_001_CA_1_validation", "HOBBIES_1_001_CA_1_validation"]):
        for d in train_dates:
            dow = d.day_of_week
            # Deterministic demand: higher on weekends
            base = 5.0 if dow >= 5 else 2.0
            records_train.append({
                "id": sid,
                "item_id": sid.split("_")[0] + "_" + sid.split("_")[1],
                "dept_id": sid.split("_")[0],
                "cat_id": sid.split("_")[0],
                "store_id": "CA_1",
                "state_id": "CA",
                "date": d,
                "demand": base + s_idx,
                "day_of_week": dow,
                "month": d.month,
                "year": d.year,
                "day": d.day,
                "is_weekend": int(dow >= 5),
                "quarter": d.quarter,
                "week_of_year": d.isocalendar().week,
                "day_sin": np.sin(2 * np.pi * dow / 7.0),
                "day_cos": np.cos(2 * np.pi * dow / 7.0),
                "month_sin": np.sin(2 * np.pi * d.month / 12.0),
                "month_cos": np.cos(2 * np.pi * d.month / 12.0),
                "has_event_1": 0,
                "has_event_2": 0,
                "is_event_day": 0,
                "is_christmas": 0,
                "snap_active": 1,
                "snap_CA": 1,
                "snap_TX": 0,
                "snap_WI": 0,
                "lag_1": base,
                "lag_7": base,
                "lag_28": base,
                "rolling_mean_7": base,
                "rolling_std_7": 0.5,
                "rolling_mean_28": base,
                "rolling_std_28": 0.5,
            })

    records_val = []
    for s_idx, sid in enumerate(["FOODS_1_001_CA_1_validation", "HOBBIES_1_001_CA_1_validation"]):
        for d in val_dates:
            dow = d.day_of_week
            base = 5.0 if dow >= 5 else 2.0
            records_val.append({
                "id": sid,
                "item_id": sid.split("_")[0] + "_" + sid.split("_")[1],
                "dept_id": sid.split("_")[0],
                "cat_id": sid.split("_")[0],
                "store_id": "CA_1",
                "state_id": "CA",
                "date": d,
                "demand": base + s_idx,
                "day_of_week": dow,
                "month": d.month,
                "year": d.year,
                "day": d.day,
                "is_weekend": int(dow >= 5),
                "quarter": d.quarter,
                "week_of_year": d.isocalendar().week,
                "day_sin": np.sin(2 * np.pi * dow / 7.0),
                "day_cos": np.cos(2 * np.pi * dow / 7.0),
                "month_sin": np.sin(2 * np.pi * d.month / 12.0),
                "month_cos": np.cos(2 * np.pi * d.month / 12.0),
                "has_event_1": 0,
                "has_event_2": 0,
                "is_event_day": 0,
                "is_christmas": 0,
                "snap_active": 1,
                "snap_CA": 1,
                "snap_TX": 0,
                "snap_WI": 0,
                "lag_1": 0,
                "lag_7": 0,
                "lag_28": 0,
                "rolling_mean_7": 0,
                "rolling_std_7": 0,
                "rolling_mean_28": 0,
                "rolling_std_28": 0,
            })

    return pd.DataFrame(records_train), pd.DataFrame(records_val)


def test_seasonal_naive_forecasting(synthetic_train_val_data):
    """Verifies that Seasonal-Naive correctly predicts matching day-of-week demand."""
    train_df, val_df = synthetic_train_val_data
    snaive = SeasonalNaiveForecaster(season_length=7)
    snaive.fit(train_df)

    preds = snaive.predict(val_df)
    assert len(preds) == len(val_df)
    assert (preds >= 0).all()

    # In our deterministic synthetic data, weekday demand is exactly identical to the previous week
    # Hence MAE should be 0.0
    metrics = compute_forecasting_metrics(val_df["demand"], preds)
    assert metrics["mae"] == 0.0


def test_gbdt_recursive_zero_leakage(synthetic_train_val_data):
    """
    CRITICAL ZERO LEAKAGE TEST:
    Modifying the ground-truth demand in val_df MUST NOT change GBDT recursive predictions.
    """
    train_df, val_df = synthetic_train_val_data

    gbdt = GBDTForecaster(n_estimators=30, learning_rate=0.1)
    gbdt.fit(train_df)

    # 1. Normal prediction
    preds_clean = gbdt.predict_recursive(val_df, train_df, horizon=14)

    # 2. Tampered validation data: inject huge spikes into ground-truth demand
    val_tampered = val_df.copy()
    val_tampered["demand"] = val_tampered["demand"] + 9999.0
    preds_after_tamper = gbdt.predict_recursive(val_tampered, train_df, horizon=14)

    # Predictions must be EXACTLY IDENTICAL because validation demand is NEVER consumed!
    np.testing.assert_allclose(
        preds_clean,
        preds_after_tamper,
        err_msg="TARGET LEAKAGE DETECTED: Tampering with validation ground-truth demand altered GBDT predictions!"
    )


def test_baseline_pipeline_integration(tmp_path, synthetic_train_val_data):
    """Integration test: validates end-to-end pipeline execution and output artifact persistence."""
    train_df, val_df = synthetic_train_val_data
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    out_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    pipeline = M5BaselinePipeline(
        train_path=str(train_path),
        val_path=str(val_path),
        output_dir=str(out_dir),
        reports_dir=str(reports_dir),
    )

    results = pipeline.run()

    assert (out_dir / "baseline_predictions.csv").exists()
    assert (reports_dir / "baseline_metrics.json").exists()

    metrics_s = results["metrics_snaive"]
    metrics_g = results["metrics_gbdt"]
    assert "mae" in metrics_s and "rmse" in metrics_s and "wape" in metrics_s
    assert "mae" in metrics_g and "rmse" in metrics_g and "wape" in metrics_g
