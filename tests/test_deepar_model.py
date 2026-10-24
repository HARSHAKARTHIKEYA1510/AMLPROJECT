"""
test_deepar_model.py - Comprehensive Unit and Integration Tests for Global DeepAR Model.

CSAI Student Project 5:
Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models.

Tests:
1. test_forecast_horizon_length: Exactly 28 days predicted per series.
2. test_tensor_shapes: Sequence shapes, static embedding dimensions, and output sample shapes.
3. test_nonnegative_output_handling: Point forecasts, P10, P50, and P90 strictly >= 0.0.
4. test_quantile_monotonicity: P10 <= P50 <= P90 across all forecast steps.
5. test_prevention_of_validation_leakage: Modifying ground-truth validation demand produces zero forecast change.
6. test_pinball_loss_accuracy: Verified pinball loss calculation against hand-computed values.
7. test_prediction_interval_metrics: Coverage and mean width calculation correctness.
8. test_short_series_padding: Series with short history handled safely with zero-padding without error.
9. test_deepar_pipeline_integration: End-to-end mini training and inference run.
"""

import sys
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deepar_model import (
    GlobalDeepARForecaster,
    GlobalDeepARNet,
    DeepARDataProcessor,
    compute_pinball_loss,
    compute_deepar_probabilistic_metrics
)


@pytest.fixture
def synthetic_m5_data():
    """Generates synthetic multi-series train and validation datasets with calendar covariates."""
    np.random.seed(42)
    n_days = 120
    dates = pd.date_range("2016-01-01", periods=n_days, freq="D")
    train_dates = dates[:92]
    val_dates = dates[92:120]  # Exactly 28 days

    series_configs = [
        {"id": "FOODS_1_001_CA_1", "store_id": "CA_1", "cat_id": "FOODS", "dept_id": "FOODS_1", "state_id": "CA", "base": 6.0},
        {"id": "HOBBIES_1_002_TX_2", "store_id": "TX_2", "cat_id": "HOBBIES", "dept_id": "HOBBIES_1", "state_id": "TX", "base": 1.5},
    ]

    train_rows = []
    val_rows = []

    for cfg in series_configs:
        for d in train_dates:
            dow = d.day_of_week
            season = 2.0 if dow >= 5 else 0.0
            demand = max(0.0, float(np.random.poisson(cfg["base"] + season)))
            train_rows.append({
                "id": cfg["id"],
                "item_id": cfg["id"][:11],
                "store_id": cfg["store_id"],
                "cat_id": cfg["cat_id"],
                "dept_id": cfg["dept_id"],
                "state_id": cfg["state_id"],
                "date": d.strftime("%Y-%m-%d"),
                "demand": demand,
                "day_sin": np.sin(2 * np.pi * d.day / 31.0),
                "day_cos": np.cos(2 * np.pi * d.day / 31.0),
                "month_sin": np.sin(2 * np.pi * d.month / 12.0),
                "month_cos": np.cos(2 * np.pi * d.month / 12.0),
                "is_weekend": 1 if dow >= 5 else 0,
                "has_event_1": 0,
                "has_event_2": 0,
                "is_event_day": 0,
                "is_christmas": 0,
                "snap_active": 1 if d.day <= 10 else 0,
            })

        for d in val_dates:
            dow = d.day_of_week
            season = 2.0 if dow >= 5 else 0.0
            demand = max(0.0, float(np.random.poisson(cfg["base"] + season)))
            val_rows.append({
                "id": cfg["id"],
                "item_id": cfg["id"][:11],
                "store_id": cfg["store_id"],
                "cat_id": cfg["cat_id"],
                "dept_id": cfg["dept_id"],
                "state_id": cfg["state_id"],
                "date": d.strftime("%Y-%m-%d"),
                "demand": demand,
                "day_sin": np.sin(2 * np.pi * d.day / 31.0),
                "day_cos": np.cos(2 * np.pi * d.day / 31.0),
                "month_sin": np.sin(2 * np.pi * d.month / 12.0),
                "month_cos": np.cos(2 * np.pi * d.month / 12.0),
                "is_weekend": 1 if dow >= 5 else 0,
                "has_event_1": 0,
                "has_event_2": 0,
                "is_event_day": 0,
                "is_christmas": 0,
                "snap_active": 1 if d.day <= 10 else 0,
            })

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    return train_df, val_df


def test_forecast_horizon_length(synthetic_m5_data):
    """Verifies that DeepAR outputs exactly 28 days of predictions for every series."""
    train_df, val_df = synthetic_m5_data
    forecaster = GlobalDeepARForecaster(
        hidden_dim=16,
        num_layers=1,
        epochs=1,
        context_length=28,
        prediction_length=28,
        stride=14,
        n_samples=20,
        device="cpu",
        random_seed=42
    )
    forecaster.fit(train_df)
    results = forecaster.predict(val_df, train_df)

    preds_by_series = results["predictions_by_series"]
    for sid in train_df["id"].unique():
        s_preds = preds_by_series[sid]
        assert len(s_preds["p10"]) == 28, f"Horizon length mismatch for P10 on {sid}"
        assert len(s_preds["p50"]) == 28, f"Horizon length mismatch for P50 on {sid}"
        assert len(s_preds["p90"]) == 28, f"Horizon length mismatch for P90 on {sid}"
        assert len(s_preds["point"]) == 28, f"Horizon length mismatch for point on {sid}"


def test_tensor_shapes(synthetic_m5_data):
    """Verifies internal tensor dimensions for data extraction and Monte Carlo rollout."""
    train_df, val_df = synthetic_m5_data
    processor = DeepARDataProcessor(context_length=28, prediction_length=28, stride=14)
    processor.fit(train_df)

    # 1. Training loader tensor shapes
    loader = processor.prepare_training_dataset(train_df, batch_size=4)
    for b_target, b_dyn, b_cat, b_scale in loader:
        assert b_target.shape[1] == 56  # context 28 + prediction 28
        assert b_dyn.shape[1] == 56
        assert b_dyn.shape[2] == len(processor.dynamic_feature_cols)
        assert b_cat.shape[1] == len(processor.STATIC_CATEGORICAL_COLS)
        assert b_scale.shape[1] == 1
        break

    # 2. Inference inputs tensor shapes
    inputs = processor.prepare_inference_inputs(train_df, val_df)
    n_series = train_df["id"].nunique()
    assert inputs["context_target"].shape == (n_series, 28)
    assert inputs["context_dyn"].shape == (n_series, 28, 10)
    assert inputs["future_dyn"].shape == (n_series, 28, 10)
    assert inputs["scales"].shape == (n_series, 1)


def test_nonnegative_output_handling(synthetic_m5_data):
    """
    Verifies that all point forecasts, P10, P50, and P90 are strictly >= 0.0,
    reflecting physical retail inventory constraints.
    """
    train_df, val_df = synthetic_m5_data
    # Force very low demand (zeros)
    train_low = train_df.copy()
    train_low["demand"] = 0.0

    forecaster = GlobalDeepARForecaster(
        hidden_dim=16,
        num_layers=1,
        epochs=1,
        context_length=28,
        prediction_length=28,
        stride=14,
        n_samples=25,
        device="cpu",
        random_seed=42
    )
    forecaster.fit(train_low)
    results = forecaster.predict(val_df, train_low)

    for sid, s_preds in results["predictions_by_series"].items():
        assert np.all(s_preds["point"] >= 0.0), f"Point forecast negative for {sid}"
        assert np.all(s_preds["p10"] >= 0.0), f"P10 forecast negative for {sid}"
        assert np.all(s_preds["p50"] >= 0.0), f"P50 forecast negative for {sid}"
        assert np.all(s_preds["p90"] >= 0.0), f"P90 forecast negative for {sid}"


def test_quantile_monotonicity(synthetic_m5_data):
    """
    Verifies quantile ordering: for all time steps h, P10 <= P50 <= P90.
    """
    train_df, val_df = synthetic_m5_data
    forecaster = GlobalDeepARForecaster(
        hidden_dim=16,
        num_layers=1,
        epochs=1,
        context_length=28,
        prediction_length=28,
        stride=14,
        n_samples=50,
        device="cpu",
        random_seed=42
    )
    forecaster.fit(train_df)
    results = forecaster.predict(val_df, train_df)

    for sid, s_preds in results["predictions_by_series"].items():
        assert np.all(s_preds["p10"] <= s_preds["p50"] + 1e-5), f"P10 > P50 on {sid}"
        assert np.all(s_preds["p50"] <= s_preds["p90"] + 1e-5), f"P50 > P90 on {sid}"


def test_prevention_of_validation_leakage(synthetic_m5_data):
    """
    CRITICAL TEST: Guarantees zero validation data leakage.
    Altering or tampering with ground-truth validation demand MUST NOT
    alter DeepAR forecasts in any way.
    """
    train_df, val_df = synthetic_m5_data
    forecaster = GlobalDeepARForecaster(
        hidden_dim=16,
        num_layers=1,
        epochs=1,
        context_length=28,
        prediction_length=28,
        stride=14,
        n_samples=30,
        device="cpu",
        random_seed=42
    )
    forecaster.fit(train_df)

    # 1. Baseline prediction with pristine validation set
    pred_orig = forecaster.predict(val_df, train_df)
    orig_p50 = pred_orig["predictions_by_series"]["FOODS_1_001_CA_1"]["p50"]

    # 2. Tampered validation set: 100x higher demand
    val_tampered = val_df.copy()
    val_tampered["demand"] = val_tampered["demand"] * 100.0 + 999.0

    # Reset seed to guarantee identical Monte Carlo random stream
    forecaster.random_seed = 42
    forecaster._set_seeds()
    pred_tampered = forecaster.predict(val_tampered, train_df)
    tampered_p50 = pred_tampered["predictions_by_series"]["FOODS_1_001_CA_1"]["p50"]

    # Both predictions must be strictly identical
    np.testing.assert_allclose(
        orig_p50,
        tampered_p50,
        atol=1e-6,
        err_msg="CRITICAL LEAKAGE DETECTED: Modifying validation demand mutated DeepAR forecasts!"
    )


def test_pinball_loss_accuracy():
    """Verifies pinball loss against known mathematical values."""
    y_true = np.array([10.0, 20.0])
    # Case 1: Perfect prediction
    assert compute_pinball_loss(y_true, y_true, quantile=0.5) == 0.0
    assert compute_pinball_loss(y_true, y_true, quantile=0.1) == 0.0
    assert compute_pinball_loss(y_true, y_true, quantile=0.9) == 0.0

    # Case 2: Under-prediction: y = [10, 20], q = [8, 16], diff = [2, 4]
    # At tau = 0.90: loss = 0.9 * [2, 4] = [1.8, 3.6] -> mean = 2.7
    # At tau = 0.10: loss = 0.1 * [2, 4] = [0.2, 0.4] -> mean = 0.3
    q_under = np.array([8.0, 16.0])
    assert np.isclose(compute_pinball_loss(y_true, q_under, quantile=0.90), 2.7)
    assert np.isclose(compute_pinball_loss(y_true, q_under, quantile=0.10), 0.3)

    # Case 3: Over-prediction: y = [10, 20], q = [12, 24], diff = [-2, -4]
    # At tau = 0.90: loss = (0.9 - 1) * [-2, -4] = -0.1 * [-2, -4] = [0.2, 0.4] -> mean = 0.3
    # At tau = 0.10: loss = (0.1 - 1) * [-2, -4] = -0.9 * [-2, -4] = [1.8, 3.6] -> mean = 2.7
    q_over = np.array([12.0, 24.0])
    assert np.isclose(compute_pinball_loss(y_true, q_over, quantile=0.90), 0.3)
    assert np.isclose(compute_pinball_loss(y_true, q_over, quantile=0.10), 2.7)


def test_prediction_interval_metrics():
    """Verifies 80% prediction interval coverage and width calculation."""
    y_true = np.array([5.0, 10.0, 15.0, 20.0])
    p10 = np.array([4.0, 8.0, 10.0, 25.0])   # 5 in [4, 6], 10 in [8, 12], 15 in [10, 18], 20 not in [25, 30]
    p90 = np.array([6.0, 12.0, 18.0, 30.0])

    metrics = compute_deepar_probabilistic_metrics(y_true, p10=p10, p50=(p10 + p90) / 2, p90=p90)
    pi = metrics["prediction_intervals"]["ci_80"]

    # 3 out of 4 inside: 75% coverage
    assert pi["coverage"] == 0.75
    # Widths: [2, 4, 8, 5] -> mean width = 19 / 4 = 4.75
    assert pi["mean_width"] == 4.75


def test_short_series_padding(synthetic_m5_data):
    """
    Verifies that series with fewer historical observations than context_length
    are safely zero-padded without crashing during inference preparation.
    """
    train_df, val_df = synthetic_m5_data
    # Truncate training history to only 10 days (less than context_length 28)
    short_train = train_df[train_df["date"] >= "2016-03-20"].copy()

    processor = DeepARDataProcessor(context_length=28, prediction_length=28)
    processor.fit(short_train)
    inputs = processor.prepare_inference_inputs(short_train, val_df)

    assert inputs["context_target"].shape[1] == 28, "Context target should be padded to 28"
    assert inputs["context_dyn"].shape[1] == 28, "Context dyn should be padded to 28"


def test_deepar_pipeline_integration(tmp_path, synthetic_m5_data):
    """
    Integration test running the full DeepAR pipeline end-to-end on synthetic data.
    """
    train_df, val_df = synthetic_m5_data
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    out_dir = tmp_path / "processed"
    rep_dir = tmp_path / "reports"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    from src.run_deepar_pipeline import M5DeepARPipeline
    pipeline = M5DeepARPipeline(
        train_path=str(train_csv),
        val_path=str(val_csv),
        baseline_metrics_path="nonexistent.json",
        kalman_metrics_path="nonexistent.json",
        output_dir=str(out_dir),
        reports_dir=str(rep_dir),
        hidden_dim=16,
        num_layers=1,
        epochs=1,
        context_length=28,
        prediction_length=28,
        stride=14,
        n_samples=10,
        device="cpu",
        random_seed=42
    )

    results = pipeline.run()
    assert (out_dir / "deepar_predictions.csv").exists()
    assert (rep_dir / "deepar_metrics.json").exists()
    assert results["metrics"]["mae"] >= 0.0
    assert results["metrics"]["prediction_intervals"]["ci_80"]["coverage"] >= 0.0
