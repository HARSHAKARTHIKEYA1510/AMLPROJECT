"""
baseline_models.py - AML Baseline Forecasting Models for M5 Inventory Demand.

CSAI Student Project 5:
Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models.

This module implements two foundational baseline forecasters:
1. SeasonalNaiveForecaster: Day-of-week seasonal persistence.
2. GBDTForecaster: LightGBM / HistGradientBoosting regressor with pure recursive multi-step forecasting.

Both models operate strictly without lookahead, forecasting a 28-day validation
horizon without consuming any ground-truth validation demand.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("M5Baselines")


def compute_forecasting_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Computes standard inventory forecasting metrics: MAE, RMSE, and WAPE.
    
    Robust Zero-Denominator Handling for WAPE:
    WAPE = sum(|y - y_hat|) / sum(y)
    - If sum(y) == 0 and sum(|y - y_hat|) == 0: WAPE is exactly 0.0 (perfect zero forecast).
    - If sum(y) == 0 and sum(|y - y_hat|) > 0: WAPE evaluates to np.nan (undefined denominator).
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) == 0 or len(y_p) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")
    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)}).")

    abs_errors = np.abs(y_t - y_p)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    sum_actual = float(np.sum(y_t))
    sum_abs_err = float(np.sum(abs_errors))

    if sum_actual == 0.0:
        if sum_abs_err == 0.0:
            wape = 0.0
        else:
            wape = float(np.nan)
    else:
        wape = float(sum_abs_err / sum_actual)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 4) if not np.isnan(wape) else None,
        "sum_actual": round(sum_actual, 2),
        "sum_abs_err": round(sum_abs_err, 2),
        "count": len(y_t),
    }


class SeasonalNaiveForecaster:
    """
    Seasonal-Naive Baseline Forecaster.
    
    Predicts future daily demand using the corresponding day of the week
    observed in the final completed week prior to the forecast origin T.
    
    For a 28-day horizon (4 weeks), the final 7 observed days
    [T-6, ..., T] are mapped by day_of_week and persisted across each week.
    Zero ground-truth validation demand is ever observed.
    """

    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self.seasonal_profile: Dict[str, Dict[int, float]] = {}
        self.last_observed_date: Optional[pd.Timestamp] = None

    def fit(self, train_df: pd.DataFrame) -> "SeasonalNaiveForecaster":
        """
        Learns the most recent seasonal profile for each series.
        """
        logger.info("Fitting Seasonal-Naive model on training history...")
        df = train_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        dow_col = "day_of_week" if "day_of_week" in df.columns else "wday"

        self.seasonal_profile = {}
        self.last_observed_date = df["date"].max()

        # For each series, extract the final season_length observations
        for sid, grp in df.groupby("id", observed=True):
            sorted_grp = grp.sort_values("date")
            last_window = sorted_grp.tail(self.season_length)
            
            # Map day of week to demand
            profile = {}
            for _, row in last_window.iterrows():
                profile[int(row[dow_col])] = float(row["demand"])
            
            self.seasonal_profile[str(sid)] = profile

        logger.info(
            f"Seasonal-Naive profile learned across {len(self.seasonal_profile)} series "
            f"(forecast origin: {self.last_observed_date.date()})."
        )
        return self

    def predict(self, val_df: pd.DataFrame) -> np.ndarray:
        """
        Forecasts demand across all validation observations using day-of-week persistence.
        """
        df = val_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        dow_col = "day_of_week" if "day_of_week" in df.columns else "wday"

        preds = []
        for _, row in df.iterrows():
            sid = str(row["id"])
            dow = int(row[dow_col])
            profile = self.seasonal_profile.get(sid, {})
            # Predict demand from corresponding day of week; default to 0.0 if unseen
            pred = profile.get(dow, 0.0)
            preds.append(max(0.0, pred))

        return np.array(preds, dtype=np.float64)


class GBDTForecaster:
    """
    Gradient-Boosted Tree Baseline Forecaster.
    
    Uses LightGBM (with HistGradientBoostingRegressor fallback) to learn
    non-linear relationships between lag/rolling demand, calendar dynamics,
    SNAP policy schedules, and product/store hierarchy.
    
    MULTI-STEP RECURSIVE FORECASTING WITHOUT TARGET LEAKAGE:
    To forecast the 28-day horizon without using actual validation-period demand,
    the model recursively rolls forward day by day:
    1. At day T+1: features use historical observations up to T.
    2. At day T+h (h >= 2): lag_1, lag_7, lag_28, and rolling statistics
       are updated iteratively using the PREDICTED demand values from previous steps.
    3. Ground-truth validation demand is NEVER fed as an input.
    """

    def __init__(
        self,
        n_estimators: int = 120,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        use_fallback: bool = False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.use_fallback = use_fallback

        self.model = None
        self.model_type = "lightgbm"
        self.feature_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.cat_mappings: Dict[str, Dict[Any, int]] = {}
        self.feature_importances: Dict[str, float] = {}

    def _init_model(self):
        """Initializes LightGBM or scikit-learn fallback."""
        if not self.use_fallback:
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    num_leaves=self.num_leaves,
                    max_depth=self.max_depth,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    random_state=self.random_state,
                    verbosity=-1,
                )
                self.model_type = "lightgbm"
                return
            except Exception as e:
                logger.warning(f"LightGBM initialization failed ({e}). Falling back to HistGradientBoostingRegressor.")

        from sklearn.ensemble import HistGradientBoostingRegressor
        self.model = HistGradientBoostingRegressor(
            max_iter=self.n_estimators,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.num_leaves,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model_type = "hist_gradient_boosting"

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
    ) -> "GBDTForecaster":
        """
        Trains the GBDT model strictly on training observations.
        """
        self._init_model()
        logger.info(f"Training GBDT model ({self.model_type}) on {len(train_df):,} records...")

        self.cat_cols = cat_cols if cat_cols is not None else [
            "item_id", "dept_id", "cat_id", "store_id", "state_id"
        ]

        if feature_cols is None:
            exclude = ["id", "date", "d", "demand", "weekday"]
            self.feature_cols = [c for c in train_df.columns if c not in exclude]
        else:
            self.feature_cols = [c for c in feature_cols if c != "weekday"]

        # Auto-detect all string/object/categorical columns
        auto_cats = [
            c for c in self.feature_cols
            if pd.api.types.is_string_dtype(train_df[c])
            or isinstance(train_df[c].dtype, pd.CategoricalDtype)
            or train_df[c].dtype.kind in ("O", "U", "S")
        ]
        self.cat_cols = list(dict.fromkeys(self.cat_cols + auto_cats))

        X_train = train_df[self.feature_cols].copy()
        y_train = train_df["demand"].values

        # Encode categoricals
        self.cat_dtypes: Dict[str, Any] = {}
        if self.model_type == "lightgbm":
            for c in self.cat_cols:
                if c in X_train.columns:
                    X_train[c] = X_train[c].astype("category")
                    self.cat_dtypes[c] = X_train[c].dtype
        else:
            # Ordinal encode for scikit-learn
            for c in self.cat_cols:
                if c in X_train.columns:
                    uniques = {val: idx for idx, val in enumerate(X_train[c].dropna().unique())}
                    self.cat_mappings[c] = uniques
                    X_train[c] = X_train[c].map(uniques).fillna(-1).astype(int)

        t0 = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - t0
        logger.info(f"GBDT ({self.model_type}) training completed in {elapsed:.2f} seconds.")

        # Extract feature importances
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            total = np.sum(importances)
            norm_imp = (importances / total) if total > 0 else importances
            self.feature_importances = {
                feat: round(float(norm_imp[i]), 4)
                for i, feat in enumerate(self.feature_cols)
            }
            # Sort descending
            self.feature_importances = dict(
                sorted(self.feature_importances.items(), key=lambda item: item[1], reverse=True)
            )

        return self

    def predict_recursive(
        self,
        val_df: pd.DataFrame,
        train_df: pd.DataFrame,
        horizon: int = 28
    ) -> np.ndarray:
        """
        Executes leak-free recursive multi-step forecasting across validation horizon.
        
        At each step h in [1..horizon]:
        - Dynamic features (lag_1, lag_7, lag_28, rolling stats) are derived
          purely from [historical train demand] + [model predictions up to step h-1].
        - Static calendar, cyclical, event, and SNAP policy indicators are known ex-ante.
        - Ground-truth validation demand is strictly withheld.
        """
        logger.info(
            f"Generating {horizon}-day recursive multi-step GBDT forecast (zero validation demand used)..."
        )
        val = val_df.copy()
        val["date"] = pd.to_datetime(val["date"])
        dates = sorted(val["date"].unique())

        if len(dates) != horizon:
            logger.warning(f"Found {len(dates)} unique dates in validation set, expected {horizon}.")

        # Build demand history buffers up to forecast origin T
        train = train_df.copy()
        train["date"] = pd.to_datetime(train["date"])
        
        # Maintain a rolling demand buffer per series
        # Buffer needs at least 35 days of history for lag_28 and rolling_28
        demand_buffers: Dict[str, List[float]] = {}
        for sid, grp in train.groupby("id", observed=True):
            sorted_demand = grp.sort_values("date")["demand"].tolist()
            # Retain last 35 observations for buffer
            demand_buffers[str(sid)] = sorted_demand[-35:] if len(sorted_demand) >= 35 else sorted_demand

        prediction_map: Dict[Tuple[str, pd.Timestamp], float] = {}

        # Step-by-step recursive forward rollout
        for step_idx, cur_date in enumerate(dates):
            step_rows = val[val["date"] == cur_date].copy()
            step_features = []

            for _, row in step_rows.iterrows():
                sid = str(row["id"])
                buf = demand_buffers[sid]

                # Dynamically construct leak-free lag features from buffer
                # lag_1 is the observation (or prediction) from the immediately preceding day
                lag_1 = float(buf[-1])
                lag_7 = float(buf[-7]) if len(buf) >= 7 else lag_1
                lag_28 = float(buf[-28]) if len(buf) >= 28 else lag_1

                # Dynamically construct rolling features from buffer
                window_7 = buf[-7:] if len(buf) >= 7 else buf
                rolling_mean_7 = float(np.mean(window_7))
                rolling_std_7 = float(np.std(window_7, ddof=1)) if len(window_7) > 1 else 0.0

                window_28 = buf[-28:] if len(buf) >= 28 else buf
                rolling_mean_28 = float(np.mean(window_28))
                rolling_std_28 = float(np.std(window_28, ddof=1)) if len(window_28) > 1 else 0.0

                feat_dict = {
                    "lag_1": lag_1,
                    "lag_7": lag_7,
                    "lag_28": lag_28,
                    "rolling_mean_7": rolling_mean_7,
                    "rolling_std_7": rolling_std_7,
                    "rolling_mean_28": rolling_mean_28,
                    "rolling_std_28": rolling_std_28,
                }

                # Copy all remaining static / exogenous features (calendar, events, SNAP, IDs)
                for c in self.feature_cols:
                    if c not in feat_dict:
                        feat_dict[c] = row[c]

                step_features.append(feat_dict)

            df_step = pd.DataFrame(step_features)[self.feature_cols]

            # Format categoricals
            if self.model_type == "lightgbm":
                for c in self.cat_cols:
                    if c in df_step.columns and c in self.cat_dtypes:
                        df_step[c] = df_step[c].astype(self.cat_dtypes[c])
            else:
                for c in self.cat_cols:
                    if c in df_step.columns:
                        mapping = self.cat_mappings.get(c, {})
                        df_step[c] = df_step[c].map(mapping).fillna(-1).astype(int)

            # Predict current step demand (clip at lower bound 0)
            raw_preds = self.model.predict(df_step)
            step_preds = np.clip(raw_preds, 0.0, None)

            # Store predictions and recursively update demand buffers
            for sid, pred in zip(step_rows["id"], step_preds):
                pred_val = float(pred)
                prediction_map[(str(sid), cur_date)] = pred_val
                # RECURSIVE INJECTION: Append predicted demand to series history
                demand_buffers[str(sid)].append(pred_val)

        # Assemble predictions matching original validation row ordering
        ordered_preds = [
            prediction_map[(str(r["id"]), r["date"])]
            for _, r in val.iterrows()
        ]
        return np.array(ordered_preds, dtype=np.float64)


class M5BaselinePipeline:
    """
    End-to-End Orchestrator for AML Baseline Models.
    
    1. Loads existing processed datasets (train_features.csv, val_features.csv).
    2. Fits Seasonal-Naive and GBDT forecasters strictly on training observations.
    3. Generates 28-day out-of-sample multi-step forecasts without validation demand input.
    4. Evaluates MAE, RMSE, and WAPE across identical series, dates, and horizon.
    5. Saves metrics, predictions, and publication-grade plots.
    """

    def __init__(
        self,
        train_path: str = "data/processed/train_features.csv",
        val_path: str = "data/processed/val_features.csv",
        output_dir: str = "data/processed",
        reports_dir: str = "reports",
    ):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.output_dir = Path(output_dir)
        self.reports_dir = Path(reports_dir)
        self.figures_dir = self.reports_dir / "figures"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Executes full baseline forecasting workflow."""
        start_time = time.time()
        logger.info("=" * 70)
        logger.info("  M5 INVENTORY DEMAND FORECASTING - BASELINE BENCHMARK")
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
        logger.info(
            f"Dataset verified: {n_series} series, {len(train_df):,} train rows, "
            f"{len(val_df):,} validation rows ({val_horizon} days per series)."
        )

        # 2. Fit and Predict: Seasonal-Naive
        snaive = SeasonalNaiveForecaster(season_length=7)
        snaive.fit(train_df)
        preds_snaive = snaive.predict(val_df)

        # 3. Fit and Predict: GBDT
        gbdt = GBDTForecaster(n_estimators=120, learning_rate=0.05, num_leaves=31)
        gbdt.fit(train_df)
        preds_gbdt = gbdt.predict_recursive(val_df, train_df, horizon=val_horizon)

        # 4. Attach predictions to validation frame
        results_df = val_df[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "date", "demand"]].copy()
        results_df["pred_seasonal_naive"] = preds_snaive
        results_df["pred_gbdt"] = preds_gbdt
        results_df["error_seasonal_naive"] = results_df["pred_seasonal_naive"] - results_df["demand"]
        results_df["error_gbdt"] = results_df["pred_gbdt"] - results_df["demand"]
        results_df["abs_error_seasonal_naive"] = np.abs(results_df["error_seasonal_naive"])
        results_df["abs_error_gbdt"] = np.abs(results_df["error_gbdt"])

        # 5. Evaluate Overall Metrics
        y_true = results_df["demand"].values
        metrics_snaive = compute_forecasting_metrics(y_true, preds_snaive)
        metrics_gbdt = compute_forecasting_metrics(y_true, preds_gbdt)

        logger.info("-" * 70)
        logger.info(f"Seasonal-Naive  -> MAE: {metrics_snaive['mae']:.4f} | RMSE: {metrics_snaive['rmse']:.4f} | WAPE: {metrics_snaive['wape']:.4f}")
        logger.info(f"GBDT (Recursive)-> MAE: {metrics_gbdt['mae']:.4f} | RMSE: {metrics_gbdt['rmse']:.4f} | WAPE: {metrics_gbdt['wape']:.4f}")
        logger.info("-" * 70)

        # 6. Evaluate Breakdown by Category & Intermittency
        cat_metrics = {}
        for cat in results_df["cat_id"].unique():
            sub = results_df[results_df["cat_id"] == cat]
            cat_metrics[str(cat)] = {
                "seasonal_naive": compute_forecasting_metrics(sub["demand"], sub["pred_seasonal_naive"]),
                "gbdt": compute_forecasting_metrics(sub["demand"], sub["pred_gbdt"]),
            }

        # 7. Evaluate Per-Series Metrics Summary
        series_summaries = []
        for sid, grp in results_df.groupby("id", observed=True):
            s_snaive = compute_forecasting_metrics(grp["demand"], grp["pred_seasonal_naive"])
            s_gbdt = compute_forecasting_metrics(grp["demand"], grp["pred_gbdt"])
            series_summaries.append({
                "id": str(sid),
                "cat_id": str(grp["cat_id"].iloc[0]),
                "store_id": str(grp["store_id"].iloc[0]),
                "total_demand": float(grp["demand"].sum()),
                "zero_demand_days": int((grp["demand"] == 0).sum()),
                "snaive_mae": s_snaive["mae"],
                "snaive_rmse": s_snaive["rmse"],
                "snaive_wape": s_snaive["wape"],
                "gbdt_mae": s_gbdt["mae"],
                "gbdt_rmse": s_gbdt["rmse"],
                "gbdt_wape": s_gbdt["wape"],
            })

        # 8. Save Predictions
        pred_csv_path = self.output_dir / "baseline_predictions.csv"
        results_df.to_csv(pred_csv_path, index=False)
        logger.info(f"[SAVED] Baseline predictions -> {pred_csv_path}")

        # 9. Save Metrics JSON
        metadata = {
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "validation_horizon_days": int(val_horizon),
            "validation_date_range": [
                str(val_df["date"].min().date()),
                str(val_df["date"].max().date())
            ],
            "total_series": int(n_series),
            "total_validation_observations": len(val_df),
            "overall_metrics": {
                "seasonal_naive": metrics_snaive,
                "gbdt_recursive": metrics_gbdt,
                "gbdt_relative_improvements": {
                    "mae_reduction_pct": round((metrics_snaive["mae"] - metrics_gbdt["mae"]) / metrics_snaive["mae"] * 100, 2) if metrics_snaive["mae"] > 0 else 0.0,
                    "rmse_reduction_pct": round((metrics_snaive["rmse"] - metrics_gbdt["rmse"]) / metrics_snaive["rmse"] * 100, 2) if metrics_snaive["rmse"] > 0 else 0.0,
                    "wape_reduction_pct": round((metrics_snaive["wape"] - metrics_gbdt["wape"]) / metrics_snaive["wape"] * 100, 2) if (metrics_snaive["wape"] is not None and metrics_snaive["wape"] > 0 and metrics_gbdt["wape"] is not None) else 0.0,
                }
            },
            "category_breakdown": cat_metrics,
            "feature_importances_top_10": dict(list(gbdt.feature_importances.items())[:10]),
            "model_configurations": {
                "seasonal_naive": {"season_length": 7, "multi_step_strategy": "day_of_week_persistence"},
                "gbdt": {
                    "library": gbdt.model_type,
                    "n_estimators": gbdt.n_estimators,
                    "learning_rate": gbdt.learning_rate,
                    "num_leaves": gbdt.num_leaves,
                    "max_depth": gbdt.max_depth,
                    "multi_step_strategy": "recursive_auto_regressive",
                    "features_count": len(gbdt.feature_cols),
                }
            },
            "sample_series_metrics": series_summaries[:10],
            "execution_time_seconds": round(time.time() - start_time, 2),
        }

        metrics_json_path = self.reports_dir / "baseline_metrics.json"
        with open(metrics_json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"[SAVED] Baseline metrics -> {metrics_json_path}")

        return {
            "results_df": results_df,
            "metadata": metadata,
            "metrics_snaive": metrics_snaive,
            "metrics_gbdt": metrics_gbdt,
        }
