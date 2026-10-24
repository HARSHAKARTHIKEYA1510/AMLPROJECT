"""
kalman_filter_model.py - Structural State-Space Kalman Filtering per SKU for M5 Demand.

CSAI Student Project 5:
Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models.

This module implements:
1. SKUKalmanForecaster: A modular, univariate structural time-series model
   (Local Level + 7-Day Weekly Seasonality) fitted per product-store series.
2. M5KalmanPipeline: An end-to-end pipeline that fits separate filters per SKU,
   forecasts the 28-day out-of-sample validation horizon with uncertainty bounds,
   evaluates MAE, RMSE, WAPE, and prediction interval coverage/width,
   and exports results.

State-Space Mathematical Formulation:
-------------------------------------
State Vector x_t in R^7:
    x_t = [mu_t, s_{1, t}, s_{2, t}, s_{3, t}, s_{4, t}, s_{5, t}, s_{6, t}]^T
    where:
    - mu_t: Unobserved local demand level (intercept/drift).
    - s_{1, t} ... s_{6, t}: Unobserved weekly seasonal components satisfying
      the sum-to-zero seasonal constraint: sum_{j=1}^7 s_{j, t} = 0.

State Transition Equation:
    x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)
    F = [ [1,  0,  0,  0,  0,  0,  0],
          [0, -1, -1, -1, -1, -1, -1],
          [0,  1,  0,  0,  0,  0,  0],
          [0,  0,  1,  0,  0,  0,  0],
          [0,  0,  0,  1,  0,  0,  0],
          [0,  0,  0,  0,  1,  0,  0],
          [0,  0,  0,  0,  0,  1,  0] ]
    Q = diag(q_mu, q_s, 0, 0, 0, 0, 0)
    where:
    - q_mu: Process noise variance for the level evolution.
    - q_s: Process noise variance for weekly seasonal evolution.

Observation Equation:
    y_t = H x_t + v_t,   v_t ~ N(0, R)
    H = [1, 1, 0, 0, 0, 0, 0]
    => y_t = mu_t + s_{1, t} + v_t
    where:
    - R: Observation noise variance (measurement error / unpredictable shocks).

Missing Observation Handling:
    When y_t is missing (NaN), the filter performs the time update (prediction)
    without a measurement update, properly expanding uncertainty P_t.
    Zero demand (y_t = 0) is treated as a valid observation of zero sales,
    never incorrectly assumed to be a stockout.

Non-negative Demand Handling:
    Point forecasts and lower prediction intervals are clipped at 0.0,
    reflecting physical retail inventory constraints:
        y_hat = max(0.0, H x_{T+h|T})
        lower_bound = max(0.0, y_hat_raw - z * sigma)
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import baseline metrics utility to guarantee identical evaluation logic
from src.baseline_models import compute_forecasting_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SKUKalmanFilter")


def compute_prediction_interval_metrics(
    y_true: Union[np.ndarray, pd.Series],
    lower_bound: Union[np.ndarray, pd.Series],
    upper_bound: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Computes empirical prediction interval evaluation metrics:
    1. Coverage: Proportion of ground-truth observations falling within [lower, upper].
    2. Mean Interval Width: Average spread (upper - lower) measuring forecast sharpness.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    l_b = np.asarray(lower_bound, dtype=np.float64)
    u_b = np.asarray(upper_bound, dtype=np.float64)

    if len(y_t) == 0:
        raise ValueError("Cannot evaluate intervals on empty arrays.")

    inside = (y_t >= l_b) & (y_t <= u_b)
    coverage = float(np.mean(inside))
    width = float(np.mean(u_b - l_b))

    return {
        "coverage": round(coverage, 4),
        "mean_width": round(width, 4),
        "inside_count": int(np.sum(inside)),
        "total_count": len(y_t)
    }


class SKUKalmanForecaster:
    """
    Univariate Structural Time Series Kalman Filter for a single SKU.
    
    Implements a Local Level + 7-Day Weekly Seasonality state-space model:
        State vector dimension: 7
        State: [mu, s_1, s_2, s_3, s_4, s_5, s_6]
    
    Parameters are estimated strictly from training history:
    - R: Observation noise variance.
    - q_mu: Level process noise variance.
    - q_s: Seasonal process noise variance.
    """

    def __init__(
        self,
        q_level_ratio: float = 0.005,
        q_season_ratio: float = 0.001,
        r_ratio: float = 0.75,
        tune_parameters: bool = False
    ):
        """
        Initializes the SKU Kalman Forecaster.
        
        Args:
            q_level_ratio: Ratio of empirical variance allocated to level process noise.
            q_season_ratio: Ratio of empirical variance allocated to seasonal process noise.
            r_ratio: Ratio of empirical variance allocated to observation noise.
            tune_parameters: If True, evaluates candidate variance ratios using
                             training innovation log-likelihood to select optimal parameters.
        """
        self.q_level_ratio = q_level_ratio
        self.q_season_ratio = q_season_ratio
        self.r_ratio = r_ratio
        self.tune_parameters = tune_parameters

        self.dim_x = 7
        self.season_length = 7

        # Measurement matrix H: shape (1, 7)
        self.H = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)

        # Transition matrix F: shape (7, 7)
        self.F = np.zeros((7, 7), dtype=np.float64)
        self.F[0, 0] = 1.0  # Level random walk: mu_t = mu_{t-1} + w_{mu, t}
        self.F[1, 1:7] = -1.0  # Seasonal sum constraint: s_{1, t} = -sum_{j=1}^6 s_{j, t-1} + w_{s, t}
        for i in range(2, 7):
            self.F[i, i - 1] = 1.0  # Shift register: s_{j, t} = s_{j-1, t-1}

        # Fitted state and parameters
        self.R: Optional[float] = None
        self.q_mu: Optional[float] = None
        self.q_s: Optional[float] = None
        self.Q: Optional[np.ndarray] = None
        self.x_post: Optional[np.ndarray] = None  # Posterior state at origin T: shape (7,)
        self.P_post: Optional[np.ndarray] = None  # Posterior covariance at origin T: shape (7, 7)
        self.log_likelihood_: Optional[float] = None
        self.training_series_id: Optional[str] = None
        self.last_training_date: Optional[pd.Timestamp] = None

    def _build_Q(self, q_mu: float, q_s: float) -> np.ndarray:
        """Constructs the 7x7 process noise covariance matrix Q."""
        Q = np.zeros((7, 7), dtype=np.float64)
        Q[0, 0] = q_mu
        Q[1, 1] = q_s
        return Q

    def _evaluate_candidate_log_likelihood(
        self,
        y: np.ndarray,
        q_mu: float,
        q_s: float,
        R: float
    ) -> float:
        """
        Computes the average Gaussian innovation log-likelihood on training history:
            LL = -0.5 * sum_{t} [ ln(2 * pi * S_t) + (v_t^2) / S_t ]
        Evaluated after an initial burn-in period of 28 days.
        """
        Q = self._build_Q(q_mu, q_s)
        var_y = float(np.var(y)) if len(y) > 0 else 1.0
        x = np.zeros(7, dtype=np.float64)
        x[0] = float(np.mean(y[:min(len(y), 28)]))
        P = np.eye(7, dtype=np.float64) * max(1.0, var_y)

        total_ll = 0.0
        valid_steps = 0
        burn_in = min(28, len(y) // 4)

        for t, yt in enumerate(y):
            # Time update
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + Q

            if np.isnan(yt):
                x = x_pred
                P = P_pred
                continue

            # Measurement update
            S = float((self.H @ P_pred @ self.H.T)[0, 0] + R)
            innov = float(yt - (self.H @ x_pred)[0])

            if t >= burn_in:
                total_ll += -0.5 * (np.log(2.0 * np.pi * S) + (innov ** 2) / S)
                valid_steps += 1

            K = (P_pred @ self.H.T) / S  # shape (7, 1)
            x = x_pred + K.flatten() * innov
            P = (np.eye(7) - K @ self.H) @ P_pred

        return total_ll / max(1, valid_steps)

    def fit(
        self,
        train_series: Union[np.ndarray, pd.Series],
        series_id: Optional[str] = None,
        dates: Optional[pd.Series] = None
    ) -> "SKUKalmanForecaster":
        """
        Fits the Kalman Filter sequentially over the training demand series.
        
        Args:
            train_series: Array or Series of historical daily demand units.
            series_id: Optional identifier for logging and diagnostics.
            dates: Optional date series corresponding to observations.
        
        Returns:
            self: The fitted SKUKalmanForecaster instance.
        """
        self.training_series_id = series_id
        if dates is not None and len(dates) > 0:
            last_d = dates[-1] if hasattr(dates, "iloc") is False else dates.iloc[-1]
            self.last_training_date = pd.to_datetime(last_d)

        y = np.asarray(train_series, dtype=np.float64)
        if len(y) == 0:
            raise ValueError(f"Training series for SKU {series_id} is empty.")

        # Compute empirical training variance (excluding NaNs)
        valid_y = y[~np.isnan(y)]
        var_y = float(np.var(valid_y)) if len(valid_y) > 1 else 1.0
        if var_y < 1e-4:
            var_y = 1.0

        # Parameter Estimation / Tuning strictly on training data
        if self.tune_parameters and len(valid_y) >= 56:
            candidate_qls = [0.001, 0.005, 0.01, 0.02]
            candidate_qss = [0.0005, 0.001, 0.005]
            best_ll = -np.inf
            best_ql = self.q_level_ratio
            best_qs = self.q_season_ratio

            base_R = max(0.05, self.r_ratio * var_y)
            for ql in candidate_qls:
                for qs in candidate_qss:
                    cand_ll = self._evaluate_candidate_log_likelihood(
                        y, ql * var_y, qs * var_y, base_R
                    )
                    if cand_ll > best_ll:
                        best_ll = cand_ll
                        best_ql = ql
                        best_qs = qs

            self.q_mu = max(1e-4, best_ql * var_y)
            self.q_s = max(1e-5, best_qs * var_y)
            self.R = base_R
            self.log_likelihood_ = best_ll
        else:
            self.R = max(0.05, self.r_ratio * var_y)
            self.q_mu = max(1e-4, self.q_level_ratio * var_y)
            self.q_s = max(1e-5, self.q_season_ratio * var_y)

        self.Q = self._build_Q(self.q_mu, self.q_s)

        # Initialize State and Covariance
        # x_0: initial level is sample mean of first 28 days; seasonal components 0
        init_level = float(np.mean(valid_y[:min(len(valid_y), 28)])) if len(valid_y) > 0 else 0.0
        x = np.zeros(7, dtype=np.float64)
        x[0] = init_level

        # P_0: diffuse covariance for initial uncertainty
        P = np.eye(7, dtype=np.float64) * var_y

        # Forward Filtering Pass over training observations
        for yt in y:
            # 1. Time Update (Predict)
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q

            # 2. Check for Missing Observation
            # If observation is NaN/missing: skip measurement update, keep predicted state
            if np.isnan(yt):
                x = x_pred
                P = P_pred
                continue

            # 3. Measurement Update (Update)
            # Innovation (measurement residual)
            y_pred = float((self.H @ x_pred)[0])
            innov = yt - y_pred

            # Innovation covariance S_t
            S = float((self.H @ P_pred @ self.H.T)[0, 0] + self.R)

            # Kalman Gain K_t: shape (7, 1)
            K = (P_pred @ self.H.T) / S

            # State update
            x = x_pred + K.flatten() * innov

            # Covariance update (Joseph form / standard stabilized form)
            I_KH = np.eye(7, dtype=np.float64) - K @ self.H
            P = I_KH @ P_pred

            # Numerical stability: enforce exact symmetry on P
            P = 0.5 * (P + P.T)

        # Store posterior distribution at forecast origin T
        self.x_post = x
        self.P_post = P

        return self

    def predict(
        self,
        horizon: int = 28
    ) -> Dict[str, np.ndarray]:
        """
        Generates multi-step out-of-sample forecasts for horizon h = 1..horizon
        without consuming any actual ground-truth validation demand.
        
        State and covariance are propagated forward:
            x_{T+h|T} = F^h x_{T|T}
            P_{T+h|T} = F P_{T+h-1|T} F^T + Q
            y_hat_{T+h}^{raw} = H x_{T+h|T}
            sigma^2_{T+h} = H P_{T+h|T} H^T + R
        
        Non-negative demand handling:
            y_hat = max(0.0, y_hat_raw)
            lower_bound = max(0.0, y_hat_raw - z * sigma)
            upper_bound = max(0.0, y_hat_raw + z * sigma)
        
        Args:
            horizon: Number of days to forecast into the future (default 28).
        
        Returns:
            Dictionary containing:
            - 'point': Array of non-negative point forecasts (length horizon).
            - 'raw_point': Array of unclipped point forecasts.
            - 'std': Array of forecast standard deviations (length horizon).
            - 'lower_95': Lower bound of 95% prediction interval (clipped at 0.0).
            - 'upper_95': Upper bound of 95% prediction interval.
            - 'lower_80': Lower bound of 80% prediction interval (clipped at 0.0).
            - 'upper_80': Upper bound of 80% prediction interval.
        """
        if self.x_post is None or self.P_post is None:
            raise RuntimeError("Model must be fitted before predict() is called.")

        preds_point = []
        preds_raw = []
        preds_std = []
        lower_95 = []
        upper_95 = []
        lower_80 = []
        upper_80 = []

        # Standard normal quantiles
        z_95 = 1.95996  # 95% coverage (alpha = 0.05)
        z_80 = 1.28155  # 80% coverage (alpha = 0.20)

        x_h = self.x_post.copy()
        P_h = self.P_post.copy()

        for h in range(1, horizon + 1):
            # Propagate state and covariance forward
            x_h = self.F @ x_h
            P_h = self.F @ P_h @ self.F.T + self.Q
            P_h = 0.5 * (P_h + P_h.T)  # Ensure symmetry

            # Forecast measurement and total variance
            y_raw = float((self.H @ x_h)[0])
            var_pred = float((self.H @ P_h @ self.H.T)[0, 0] + self.R)
            std_pred = float(np.sqrt(max(0.0, var_pred)))

            # Non-negative handling for inventory demand
            point_clipped = max(0.0, y_raw)
            l95 = max(0.0, y_raw - z_95 * std_pred)
            u95 = max(0.0, y_raw + z_95 * std_pred)
            l80 = max(0.0, y_raw - z_80 * std_pred)
            u80 = max(0.0, y_raw + z_80 * std_pred)

            preds_point.append(point_clipped)
            preds_raw.append(y_raw)
            preds_std.append(std_pred)
            lower_95.append(l95)
            upper_95.append(u95)
            lower_80.append(l80)
            upper_80.append(u80)

        return {
            "point": np.array(preds_point, dtype=np.float64),
            "raw_point": np.array(preds_raw, dtype=np.float64),
            "std": np.array(preds_std, dtype=np.float64),
            "lower_95": np.array(lower_95, dtype=np.float64),
            "upper_95": np.array(upper_95, dtype=np.float64),
            "lower_80": np.array(lower_80, dtype=np.float64),
            "upper_80": np.array(upper_80, dtype=np.float64),
        }


class M5KalmanPipeline:
    """
    End-to-End Orchestrator for Structural State-Space Kalman Filtering per SKU.
    
    1. Loads existing processed training and validation sets.
    2. Fits a dedicated SKUKalmanForecaster for each product-store time series.
    3. Produces 28-day out-of-sample multi-step forecasts with analytical uncertainty bounds.
    4. Evaluates MAE, RMSE, WAPE (with zero-denominator handling) and interval coverage/width.
    5. Benchmarks results directly against Seasonal-Naive and GBDT baselines.
    6. Saves `data/processed/kalman_predictions.csv` and `reports/kalman_metrics.json`.
    """

    def __init__(
        self,
        train_path: str = "data/processed/train_features.csv",
        val_path: str = "data/processed/val_features.csv",
        baseline_metrics_path: str = "reports/baseline_metrics.json",
        output_dir: str = "data/processed",
        reports_dir: str = "reports",
        tune_parameters: bool = False
    ):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.baseline_metrics_path = Path(baseline_metrics_path)
        self.output_dir = Path(output_dir)
        self.reports_dir = Path(reports_dir)
        self.figures_dir = self.reports_dir / "figures"
        self.tune_parameters = tune_parameters

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Executes full Kalman filter per-SKU forecasting workflow."""
        start_time = time.time()
        logger.info("=" * 70)
        logger.info("  M5 INVENTORY DEMAND FORECASTING - KALMAN FILTER BENCHMARK")
        logger.info("=" * 70)

        # 1. Load Data
        logger.info(f"Loading training data from {self.train_path}...")
        train_df = pd.read_csv(self.train_path)
        train_df["date"] = pd.to_datetime(train_df["date"])

        logger.info(f"Loading validation data from {self.val_path}...")
        val_df = pd.read_csv(self.val_path)
        val_df["date"] = pd.to_datetime(val_df["date"])

        n_series = val_df["id"].nunique()
        val_horizon = val_df.groupby("id")["date"].count().iloc[0]
        val_min_date = val_df["date"].min().strftime("%Y-%m-%d")
        val_max_date = val_df["date"].max().strftime("%Y-%m-%d")

        logger.info(
            f"Dataset verified: {n_series} series, {len(train_df):,} train rows, "
            f"{len(val_df):,} validation rows ({val_horizon} days per series: "
            f"{val_min_date} to {val_max_date})."
        )

        # 2. Fit individual Kalman Filter per SKU and forecast 28 days
        logger.info(f"Fitting dedicated Kalman Filter models across all {n_series} SKUs...")
        unique_series = val_df["id"].unique()

        predictions_by_series = {}
        fitted_models = {}

        for idx, sid in enumerate(unique_series, 1):
            s_train = train_df[train_df["id"] == sid].sort_values("date")
            s_val = val_df[val_df["id"] == sid].sort_values("date")

            if len(s_train) == 0:
                raise ValueError(f"No training data found for series {sid}")

            # Fit separate filter per SKU
            forecaster = SKUKalmanForecaster(
                q_level_ratio=0.005,
                q_season_ratio=0.001,
                r_ratio=0.75,
                tune_parameters=self.tune_parameters
            )
            forecaster.fit(
                train_series=s_train["demand"].values,
                series_id=sid,
                dates=s_train["date"]
            )

            # Predict full validation horizon without validation demand inputs
            forecast_dict = forecaster.predict(horizon=len(s_val))
            forecast_dict["dates"] = s_val["date"].values
            forecast_dict["actuals"] = s_val["demand"].values

            predictions_by_series[sid] = forecast_dict
            fitted_models[sid] = forecaster

            if idx % 10 == 0 or idx == n_series:
                logger.info(f"  Processed {idx}/{n_series} series ({sid})")

        # 3. Assemble unified results DataFrame
        logger.info("Assembling prediction records and calculating errors...")
        pred_records = []
        for sid in unique_series:
            s_val = val_df[val_df["id"] == sid].sort_values("date")
            fc = predictions_by_series[sid]
            for h in range(len(s_val)):
                row_meta = s_val.iloc[h]
                y_act = float(row_meta["demand"])
                y_pred = float(fc["point"][h])
                y_std = float(fc["std"][h])
                l95 = float(fc["lower_95"][h])
                u95 = float(fc["upper_95"][h])
                l80 = float(fc["lower_80"][h])
                u80 = float(fc["upper_80"][h])
                err = y_pred - y_act
                abs_err = abs(err)

                pred_records.append({
                    "id": sid,
                    "item_id": row_meta["item_id"],
                    "dept_id": row_meta["dept_id"],
                    "cat_id": row_meta["cat_id"],
                    "store_id": row_meta["store_id"],
                    "state_id": row_meta["state_id"],
                    "date": row_meta["date"].strftime("%Y-%m-%d"),
                    "demand": y_act,
                    "pred_kalman": y_pred,
                    "std_kalman": y_std,
                    "lower_95": l95,
                    "upper_95": u95,
                    "lower_80": l80,
                    "upper_80": u80,
                    "error_kalman": err,
                    "abs_error_kalman": abs_err
                })

        results_df = pd.DataFrame(pred_records)
        output_csv = self.output_dir / "kalman_predictions.csv"
        results_df.to_csv(output_csv, index=False)
        logger.info(f"Saved Kalman predictions to {output_csv} ({len(results_df)} rows).")

        # 4. Evaluate Overall Metrics
        y_true = results_df["demand"].values
        y_pred = results_df["pred_kalman"].values
        l_95 = results_df["lower_95"].values
        u_95 = results_df["upper_95"].values
        l_80 = results_df["lower_80"].values
        u_80 = results_df["upper_80"].values

        overall_metrics = compute_forecasting_metrics(y_true, y_pred)
        pi_95_metrics = compute_prediction_interval_metrics(y_true, l_95, u_95)
        pi_80_metrics = compute_prediction_interval_metrics(y_true, l_80, u_80)

        logger.info("-" * 70)
        logger.info(
            f"  KALMAN FILTER EVALUATION: "
            f"MAE: {overall_metrics['mae']:.4f} | "
            f"RMSE: {overall_metrics['rmse']:.4f} | "
            f"WAPE: {overall_metrics['wape']*100:.2f}%"
        )
        logger.info(
            f"  95% PI Coverage: {pi_95_metrics['coverage']*100:.2f}% (Avg Width: {pi_95_metrics['mean_width']:.4f}) | "
            f"80% PI Coverage: {pi_80_metrics['coverage']*100:.2f}% (Avg Width: {pi_80_metrics['mean_width']:.4f})"
        )
        logger.info("-" * 70)

        # 5. Evaluate Category & Store Breakdowns
        cat_breakdown = {}
        for cat in results_df["cat_id"].unique():
            sub = results_df[results_df["cat_id"] == cat]
            sub_m = compute_forecasting_metrics(sub["demand"].values, sub["pred_kalman"].values)
            sub_pi = compute_prediction_interval_metrics(
                sub["demand"].values, sub["lower_95"].values, sub["upper_95"].values
            )
            sub_m["coverage_95"] = sub_pi["coverage"]
            sub_m["mean_width_95"] = sub_pi["mean_width"]
            cat_breakdown[cat] = sub_m

        store_breakdown = {}
        for store in results_df["store_id"].unique():
            sub = results_df[results_df["store_id"] == store]
            sub_m = compute_forecasting_metrics(sub["demand"].values, sub["pred_kalman"].values)
            sub_pi = compute_prediction_interval_metrics(
                sub["demand"].values, sub["lower_95"].values, sub["upper_95"].values
            )
            sub_m["coverage_95"] = sub_pi["coverage"]
            sub_m["mean_width_95"] = sub_pi["mean_width"]
            store_breakdown[store] = sub_m

        # 6. Load baseline benchmark metrics for fair side-by-side comparison
        baseline_summary = {}
        relative_comparison = {}
        if self.baseline_metrics_path.exists():
            try:
                with open(self.baseline_metrics_path, "r") as f:
                    b_data = json.load(f)
                b_snaive = b_data["overall_metrics"]["seasonal_naive"]
                b_gbdt = b_data["overall_metrics"]["gbdt_recursive"]
                baseline_summary = {
                    "seasonal_naive": b_snaive,
                    "gbdt_recursive": b_gbdt
                }

                # Relative improvements of Kalman over Seasonal-Naive
                kf_mae = overall_metrics["mae"]
                kf_rmse = overall_metrics["rmse"]
                kf_wape = overall_metrics["wape"]

                sn_mae = b_snaive["mae"]
                sn_rmse = b_snaive["rmse"]
                sn_wape = b_snaive["wape"]

                gb_mae = b_gbdt["mae"]
                gb_rmse = b_gbdt["rmse"]
                gb_wape = b_gbdt["wape"]

                relative_comparison = {
                    "vs_seasonal_naive": {
                        "mae_reduction_pct": round(((sn_mae - kf_mae) / sn_mae) * 100, 2) if sn_mae > 0 else 0.0,
                        "rmse_reduction_pct": round(((sn_rmse - kf_rmse) / sn_rmse) * 100, 2) if sn_rmse > 0 else 0.0,
                        "wape_reduction_pct": round(((sn_wape - kf_wape) / sn_wape) * 100, 2) if sn_wape > 0 else 0.0,
                    },
                    "vs_gbdt": {
                        "mae_reduction_pct": round(((gb_mae - kf_mae) / gb_mae) * 100, 2) if gb_mae > 0 else 0.0,
                        "rmse_reduction_pct": round(((gb_rmse - kf_rmse) / gb_rmse) * 100, 2) if gb_rmse > 0 else 0.0,
                        "wape_reduction_pct": round(((gb_wape - kf_wape) / gb_wape) * 100, 2) if gb_wape > 0 else 0.0,
                    }
                }
            except Exception as e:
                logger.warning(f"Could not parse baseline metrics file: {e}")

        # 7. Compile and save JSON metrics report
        metrics_payload = {
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "validation_horizon_days": int(val_horizon),
            "validation_date_range": [str(val_min_date), str(val_max_date)],
            "total_series": int(n_series),
            "total_validation_observations": int(len(results_df)),
            "state_space_model": {
                "architecture": "Local Level + Weekly Seasonality (Basic Structural Model)",
                "state_dimension": 7,
                "seasonality_period": 7,
                "observation_noise_ratio": 0.75,
                "level_process_noise_ratio": 0.005,
                "seasonal_process_noise_ratio": 0.001,
                "non_negative_clipping": True,
                "missing_observation_strategy": "predict_only_skip_update",
                "intermittent_zero_handling": "preserved_as_valid_demand_observation"
            },
            "overall_metrics": {
                "kalman_filter": {
                    "mae": overall_metrics["mae"],
                    "rmse": overall_metrics["rmse"],
                    "wape": overall_metrics["wape"],
                    "sum_actual": overall_metrics["sum_actual"],
                    "sum_abs_err": overall_metrics["sum_abs_err"],
                    "count": overall_metrics["count"],
                    "prediction_intervals": {
                        "ci_95": pi_95_metrics,
                        "ci_80": pi_80_metrics
                    }
                },
                "baseline_benchmarks": baseline_summary,
                "relative_improvements": relative_comparison
            },
            "category_breakdown": cat_breakdown,
            "store_breakdown": store_breakdown
        }

        output_json = self.reports_dir / "kalman_metrics.json"
        with open(output_json, "w") as f:
            json.dump(metrics_payload, f, indent=2)
        logger.info(f"Saved Kalman metrics report to {output_json}.")

        elapsed = time.time() - start_time
        logger.info(f"Kalman pipeline completed successfully in {elapsed:.2f} seconds.")

        return {
            "metrics": metrics_payload,
            "predictions_df": results_df,
            "models": fitted_models
        }
