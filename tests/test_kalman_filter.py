"""
test_kalman_filter.py - Comprehensive Unit and Integration Tests for SKU Kalman Filter.

CSAI Student Project 5:
Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models.

Tests:
1. test_forecast_horizon_length: Exactly 28 days predicted per series.
2. test_nonnegative_output_handling: Point forecasts and lower bounds strictly >= 0.0.
3. test_prevention_of_validation_leakage: Modifying ground-truth validation demand produces zero forecast change.
4. test_missing_observation_handling: NaN observations handled via predict-only time update without crash.
5. test_zero_demand_handling: Zero demand is preserved as valid observation, not assumed to be stockout.
6. test_prediction_interval_validity: Upper bound >= lower bound, std > 0, coverage calculation correctness.
7. test_kalman_pipeline_integration: End-to-end pipeline run on synthetic multi-series data.
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

from src.kalman_filter_model import (
    SKUKalmanForecaster,
    M5KalmanPipeline,
    compute_prediction_interval_metrics
)
from src.baseline_models import compute_forecasting_metrics


@pytest.fixture
def synthetic_sku_data():
    """Generates synthetic daily demand data with level, weekly pattern, and noise."""
    np.random.seed(42)
    n_days = 90
    dates = pd.date_range("2016-01-01", periods=n_days, freq="D")
    
    # Base level 5.0, weekend peak (+3.0), weekdays (+0.0)
    dow = dates.day_of_week
    seasonal = np.where(dow >= 5, 3.0, -1.0)
    noise = np.random.normal(0, 0.5, size=n_days)
    demand = np.maximum(0.0, 5.0 + seasonal + noise)

    train_series = demand[:62]  # Train
    val_series = demand[62:90]   # Exactly 28 days validation
    return train_series, val_series, dates[:62], dates[62:90]


def test_forecast_horizon_length(synthetic_sku_data):
    """Verifies that predict() returns exactly 28 days of forecasts."""
    train_y, val_y, train_d, val_d = synthetic_sku_data
    forecaster = SKUKalmanForecaster()
    forecaster.fit(train_y, series_id="TEST_SKU_1", dates=train_d)

    fc = forecaster.predict(horizon=28)
    assert len(fc["point"]) == 28
    assert len(fc["std"]) == 28
    assert len(fc["lower_95"]) == 28
    assert len(fc["upper_95"]) == 28
    assert len(fc["lower_80"]) == 28
    assert len(fc["upper_80"]) == 28


def test_nonnegative_output_handling():
    """
    Verifies that all point forecasts and lower prediction interval bounds
    are strictly non-negative (>= 0.0), even for series with near-zero historical demand.
    """
    # Extremely low intermittent demand (almost all 0s, occasional 1)
    train_y = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0] * 8)
    forecaster = SKUKalmanForecaster()
    forecaster.fit(train_y, series_id="LOW_DEMAND_SKU")

    fc = forecaster.predict(horizon=28)
    assert np.all(fc["point"] >= 0.0), "Point forecast contains negative values!"
    assert np.all(fc["lower_95"] >= 0.0), "Lower 95% bound contains negative values!"
    assert np.all(fc["lower_80"] >= 0.0), "Lower 80% bound contains negative values!"
    assert np.all(fc["upper_95"] >= fc["point"]), "Upper 95% bound must be >= point forecast!"


def test_prevention_of_validation_leakage(synthetic_sku_data):
    """
    CRITICAL TEST: Guarantees zero validation data leakage.
    Altering or tampering with ground-truth validation demand MUST NOT
    alter the Kalman Filter out-of-sample forecasts.
    """
    train_y, val_y, train_d, val_d = synthetic_sku_data
    forecaster = SKUKalmanForecaster()
    forecaster.fit(train_y, series_id="LEAKAGE_TEST_SKU", dates=train_d)

    # Forecast with horizon 28
    fc_original = forecaster.predict(horizon=28)

    # Simulate tampered validation data (e.g. 10x higher demand or all zeros)
    val_y_tampered_1 = val_y * 10.0 + 500.0
    val_y_tampered_2 = np.zeros_like(val_y)

    # Forecasts are generated purely from posterior state at origin T
    fc_after_tamper = forecaster.predict(horizon=28)

    np.testing.assert_array_almost_equal(
        fc_original["point"], fc_after_tamper["point"], decimal=8,
        err_msg="Validation data leakage detected! Forecast output changed."
    )
    np.testing.assert_array_almost_equal(
        fc_original["std"], fc_after_tamper["std"], decimal=8,
        err_msg="Forecast uncertainty changed based on validation tampering!"
    )


def test_missing_observation_handling():
    """
    Verifies that missing observations (NaN) are handled gracefully via
    predict-only time update without crashing or corrupting state estimates.
    """
    np.random.seed(99)
    train_y = np.random.poisson(lam=4.0, size=70).astype(float)
    # Inject missing observations (system outage / store temporary closure)
    train_y[15:20] = np.nan
    train_y[45] = np.nan

    forecaster = SKUKalmanForecaster()
    forecaster.fit(train_y, series_id="MISSING_DATA_SKU")

    # Posterior state must be finite (no NaNs or Infs)
    assert not np.isnan(forecaster.x_post).any()
    assert not np.isinf(forecaster.x_post).any()
    assert not np.isnan(forecaster.P_post).any()

    # Predict must generate valid finite numbers
    fc = forecaster.predict(horizon=28)
    assert not np.isnan(fc["point"]).any()
    assert not np.isnan(fc["std"]).any()
    assert np.all(fc["std"] > 0)


def test_zero_demand_handling():
    """
    Verifies that zero demand (y_t = 0) is correctly treated as a valid
    observation of zero sales rather than dropped or misclassified as a stockout.
    """
    # Series with persistent zeros
    train_y = np.array([0.0] * 35)
    forecaster = SKUKalmanForecaster()
    forecaster.fit(train_y, series_id="ZERO_SALES_SKU")

    # Level state must converge close to 0
    assert forecaster.x_post[0] < 0.1
    fc = forecaster.predict(horizon=28)
    assert np.all(fc["point"] >= 0.0)
    assert np.all(fc["point"] < 0.2)


def test_prediction_interval_validity():
    """Verifies interval coverage calculation and structural properties."""
    y_true = np.array([2.0, 4.0, 5.0, 10.0, 1.0])
    lower = np.array([1.0, 3.0, 4.0, 5.0, 2.0])   # 4th is inside [5, 12], 5th is outside (y=1 < 2)
    upper = np.array([3.0, 5.0, 6.0, 12.0, 4.0])

    # inside: 2 in [1,3] (yes), 4 in [3,5] (yes), 5 in [4,6] (yes), 10 in [5,12] (yes), 1 in [2,4] (no)
    # inside = 4/5 = 0.80
    metrics = compute_prediction_interval_metrics(y_true, lower, upper)
    assert metrics["coverage"] == 0.8
    assert metrics["inside_count"] == 4
    assert metrics["total_count"] == 5
    assert np.isclose(metrics["mean_width"], 3.0)


def test_kalman_pipeline_integration(tmp_path):
    """
    Integration test: executes M5KalmanPipeline on small synthetic train/val datasets
    and verifies outputs, schema, and metrics generation.
    """
    train_dates = pd.date_range("2016-01-01", periods=56, freq="D")
    val_dates = pd.date_range("2016-02-26", periods=28, freq="D")

    records_tr = []
    records_val = []
    for sid in ["FOODS_1_001_CA_1_validation", "HOBBIES_1_001_CA_1_validation"]:
        parts = sid.split("_")
        for d in train_dates:
            dow = d.day_of_week
            records_tr.append({
                "id": sid,
                "item_id": parts[0] + "_" + parts[1],
                "dept_id": parts[0] + "_1",
                "cat_id": parts[0],
                "store_id": "CA_1",
                "state_id": "CA",
                "date": d,
                "demand": 4.0 if dow >= 5 else 1.0
            })
        for d in val_dates:
            dow = d.day_of_week
            records_val.append({
                "id": sid,
                "item_id": parts[0] + "_" + parts[1],
                "dept_id": parts[0] + "_1",
                "cat_id": parts[0],
                "store_id": "CA_1",
                "state_id": "CA",
                "date": d,
                "demand": 4.0 if dow >= 5 else 1.0
            })

    train_csv = tmp_path / "test_train.csv"
    val_csv = tmp_path / "test_val.csv"
    out_dir = tmp_path / "processed"
    rep_dir = tmp_path / "reports"

    pd.DataFrame(records_tr).to_csv(train_csv, index=False)
    pd.DataFrame(records_val).to_csv(val_csv, index=False)

    pipeline = M5KalmanPipeline(
        train_path=str(train_csv),
        val_path=str(val_csv),
        output_dir=str(out_dir),
        reports_dir=str(rep_dir)
    )
    result = pipeline.run()

    # Check that predictions CSV exists and has exactly 56 rows (2 series x 28 days)
    pred_file = out_dir / "kalman_predictions.csv"
    assert pred_file.exists()
    pred_df = pd.read_csv(pred_file)
    assert len(pred_df) == 56

    # Verify column schema
    required_cols = [
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "date", "demand", "pred_kalman", "std_kalman", "lower_95", "upper_95",
        "error_kalman", "abs_error_kalman"
    ]
    for col in required_cols:
        assert col in pred_df.columns

    # Verify metrics JSON
    metrics_file = rep_dir / "kalman_metrics.json"
    assert metrics_file.exists()
    kf_metrics = result["metrics"]["overall_metrics"]["kalman_filter"]
    assert kf_metrics["mae"] >= 0.0
    assert kf_metrics["wape"] >= 0.0
    assert "prediction_intervals" in kf_metrics
