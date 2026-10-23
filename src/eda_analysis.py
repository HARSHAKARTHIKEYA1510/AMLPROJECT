"""
eda_analysis.py - Advanced Exploratory Data Analysis & Statistical Modeling for M5.

Computes:
1. Multi-level hierarchical aggregations (Total, State, Store, Category, Department).
2. Intermittent demand classification using the Syntetos-Boylan / Croston framework (ADI vs CV^2).
3. Temporal dynamics, day-of-week seasonality, and monthly indices.
4. SNAP and holiday uplift analytics.
5. Price elasticity and promotional discounting dynamics.
6. Time-series stationarity diagnostics (ADF test, ACF/PACF).
7. Kalman Filter state-space demand tracking demonstration.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from statsmodels.tsa.stattools import adfuller, acf, pacf
from filterpy.kalman import KalmanFilter


class M5ExploratoryAnalysis:
    """Statistical and time-series analytical engine for M5 inventory data."""

    def __init__(self, df_cal: pd.DataFrame, df_prices: pd.DataFrame, df_sales: pd.DataFrame):
        self.df_cal = df_cal.copy()
        self.df_prices = df_prices.copy()
        self.df_sales = df_sales.copy()
        self.d_cols = [c for c in df_sales.columns if c.startswith("d_")]
        self.dates = df_cal.iloc[:len(self.d_cols)]["date"].values

        # Prepare calendar lookup mapped by d_col
        self.cal_lookup = df_cal.iloc[:len(self.d_cols)].set_index("d")

    def compute_hierarchical_aggregations(self) -> Dict[str, pd.DataFrame]:
        """Aggregate unit sales across 5 hierarchical levels."""
        print("[ANALYSIS] Computing hierarchical sales aggregations...")
        aggs = {}

        # Level 1: Total
        total_sales = self.df_sales[self.d_cols].sum(axis=0).values
        aggs["total"] = pd.DataFrame({
            "date": self.dates,
            "d": self.d_cols,
            "unit_sales": total_sales,
        })

        # Level 2: State
        state_df = self.df_sales.groupby("state_id", observed=True)[self.d_cols].sum().T
        state_df.index = self.dates
        aggs["state"] = state_df

        # Level 3: Store
        store_df = self.df_sales.groupby("store_id", observed=True)[self.d_cols].sum().T
        store_df.index = self.dates
        aggs["store"] = store_df

        # Level 4: Category
        cat_df = self.df_sales.groupby("cat_id", observed=True)[self.d_cols].sum().T
        cat_df.index = self.dates
        aggs["category"] = cat_df

        # Level 5: Department
        dept_df = self.df_sales.groupby("dept_id", observed=True)[self.d_cols].sum().T
        dept_df.index = self.dates
        aggs["department"] = dept_df

        return aggs

    def classify_demand_patterns(self, sample_size: int = None) -> pd.DataFrame:
        """
        Classifies time series into Croston / Syntetos-Boylan demand quadrants:
        - ADI (Average Demand Interval): Average time interval between non-zero demand
        - CV^2 (Coefficient of Variation squared): Variance of non-zero demand size / Mean^2
        
        Quadrants:
        - Smooth:       ADI < 1.32 & CV^2 < 0.49
        - Intermittent: ADI >= 1.32 & CV^2 < 0.49
        - Erratic:      ADI < 1.32 & CV^2 >= 0.49
        - Lumpy:        ADI >= 1.32 & CV^2 >= 0.49
        """
        print("[ANALYSIS] Calculating Croston / Syntetos-Boylan demand categorization...")
        df = self.df_sales if sample_size is None else self.df_sales.sample(n=sample_size, random_state=42)

        raw_sales = df[self.d_cols].values  # shape: (N, T)
        N, T = raw_sales.shape

        adi_list = []
        cv2_list = []
        zero_pct_list = []
        quadrant_list = []

        # Vectorized calculations per time-series
        for i in range(N):
            series = raw_sales[i]
            non_zeros = series[series > 0]
            n_non_zeros = len(non_zeros)

            if n_non_zeros <= 1:
                # Highly sparse / virtually dead series
                adi = float(T)
                cv2 = 1.0
                quadrant = "Lumpy"
            else:
                # Non-zero indices
                nz_indices = np.where(series > 0)[0]
                intervals = np.diff(nz_indices)
                adi = float(np.mean(intervals)) if len(intervals) > 0 else float(T)

                mean_nz = float(np.mean(non_zeros))
                std_nz = float(np.std(non_zeros, ddof=1)) if len(non_zeros) > 1 else 0.0
                cv2 = float((std_nz / mean_nz) ** 2) if mean_nz > 0 else 0.0

                if adi < 1.32 and cv2 < 0.49:
                    quadrant = "Smooth"
                elif adi >= 1.32 and cv2 < 0.49:
                    quadrant = "Intermittent"
                elif adi < 1.32 and cv2 >= 0.49:
                    quadrant = "Erratic"
                else:
                    quadrant = "Lumpy"

            adi_list.append(adi)
            cv2_list.append(cv2)
            zero_pct_list.append((T - n_non_zeros) / T * 100)
            quadrant_list.append(quadrant)

        res_df = pd.DataFrame({
            "id": df["id"].values,
            "item_id": df["item_id"].values,
            "dept_id": df["dept_id"].values,
            "cat_id": df["cat_id"].values,
            "store_id": df["store_id"].values,
            "state_id": df["state_id"].values,
            "ADI": adi_list,
            "CV2": cv2_list,
            "zero_pct": zero_pct_list,
            "quadrant": quadrant_list,
        })
        return res_df

    def analyze_seasonality_and_events(self, aggs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze day-of-week, monthly, event uplifts, and SNAP effects."""
        print("[ANALYSIS] Computing seasonality, SNAP, and calendar event effects...")
        total_df = aggs["total"].copy()
        total_df["date"] = pd.to_datetime(total_df["date"])
        total_df["day_name"] = total_df["date"].dt.day_name()
        total_df["dayofweek"] = total_df["date"].dt.dayofweek
        total_df["month"] = total_df["date"].dt.month
        total_df["month_name"] = total_df["date"].dt.month_name()
        total_df["year"] = total_df["date"].dt.year

        # Exclude Christmas day (d_697, d_1062, d_1427, d_1792) where sales are essentially 0 due to closures
        clean_total = total_df[~((total_df["date"].dt.month == 12) & (total_df["date"].dt.day == 25))]
        mean_sales = clean_total["unit_sales"].mean()

        # Day of week index
        dow_agg = clean_total.groupby(["dayofweek", "day_name"])["unit_sales"].mean().reset_index()
        dow_agg["index"] = (dow_agg["unit_sales"] / mean_sales).round(4)
        dow_agg = dow_agg.sort_values("dayofweek")

        # Month of year index
        month_agg = clean_total.groupby(["month", "month_name"])["unit_sales"].mean().reset_index()
        month_agg["index"] = (month_agg["unit_sales"] / mean_sales).round(4)
        month_agg = month_agg.sort_values("month")

        # SNAP analysis by state
        cal_sub = self.df_cal.iloc[:len(self.d_cols)].copy()
        cal_sub["date"] = pd.to_datetime(cal_sub["date"])
        cal_sub = cal_sub.set_index("date")

        state_sales = aggs["state"].copy()
        state_sales.index = pd.to_datetime(state_sales.index)

        snap_results = {}
        for state in ["CA", "TX", "WI"]:
            snap_col = f"snap_{state}"
            merged = pd.DataFrame({
                "sales": state_sales[state],
                "snap": cal_sub[snap_col].values,
                "is_christmas": (state_sales.index.month == 12) & (state_sales.index.day == 25),
            })
            merged = merged[~merged["is_christmas"]]

            snap_on_mean = merged[merged["snap"] == 1]["sales"].mean()
            snap_off_mean = merged[merged["snap"] == 0]["sales"].mean()
            uplift_pct = ((snap_on_mean - snap_off_mean) / snap_off_mean) * 100

            snap_results[state] = {
                "snap_on_mean": round(float(snap_on_mean), 1),
                "snap_off_mean": round(float(snap_off_mean), 1),
                "uplift_pct": round(float(uplift_pct), 2),
            }

        # Major Event Uplifts
        merged_events = pd.DataFrame({
            "date": total_df["date"],
            "unit_sales": total_df["unit_sales"],
            "event_1": self.df_cal.iloc[:len(self.d_cols)]["event_name_1"].values,
        })
        merged_events = merged_events[~((merged_events["date"].dt.month == 12) & (merged_events["date"].dt.day == 25))]

        event_sales = merged_events[merged_events["event_1"].notnull()].groupby("event_1", observed=True)["unit_sales"].mean()
        event_uplift = (((event_sales - mean_sales) / mean_sales) * 100).round(2).to_dict()

        return {
            "mean_daily_sales": round(float(mean_sales), 1),
            "day_of_week_index": dow_agg.to_dict(orient="records"),
            "monthly_index": month_agg.to_dict(orient="records"),
            "snap_analysis": snap_results,
            "top_event_uplifts": dict(sorted(event_uplift.items(), key=lambda x: x[1], reverse=True)[:10]),
            "bottom_event_uplifts": dict(sorted(event_uplift.items(), key=lambda x: x[1])[:5]),
        }

    def compute_stationarity_and_autocorrelation(self, aggs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Compute ADF test for stationarity and ACF/PACF at key lags."""
        print("[ANALYSIS] Running Augmented Dickey-Fuller (ADF) & Autocorrelation tests...")
        results = {}

        # Analyze Total sales
        total_series = aggs["total"]["unit_sales"].values
        # Differenced series
        diff_series = np.diff(total_series)

        adf_raw = adfuller(total_series, autolag="AIC")
        adf_diff = adfuller(diff_series, autolag="AIC")

        acf_vals = acf(total_series, nlags=35)
        pacf_vals = pacf(total_series, nlags=35)

        results["total_level"] = {
            "raw_series": {
                "adf_statistic": round(float(adf_raw[0]), 4),
                "p_value": round(float(adf_raw[1]), 5),
                "critical_values": {k: round(float(v), 3) for k, v in adf_raw[4].items()},
                "is_stationary": bool(adf_raw[1] < 0.05),
            },
            "differenced_series": {
                "adf_statistic": round(float(adf_diff[0]), 4),
                "p_value": round(float(adf_diff[1]), 5),
                "is_stationary": bool(adf_diff[1] < 0.05),
            },
            "autocorrelations": {
                "lag_1": round(float(acf_vals[1]), 4),
                "lag_7 (weekly)": round(float(acf_vals[7]), 4),
                "lag_14": round(float(acf_vals[14]), 4),
                "lag_21": round(float(acf_vals[21]), 4),
                "lag_28 (monthly)": round(float(acf_vals[28]), 4),
            },
            "partial_autocorrelations": {
                "lag_1": round(float(pacf_vals[1]), 4),
                "lag_7": round(float(pacf_vals[7]), 4),
            },
        }

        # Analyze State levels
        state_stationarity = {}
        for state in ["CA", "TX", "WI"]:
            s_series = aggs["state"][state].values
            adf_s = adfuller(s_series, autolag="AIC")
            state_stationarity[state] = {
                "adf_stat": round(float(adf_s[0]), 3),
                "p_value": round(float(adf_s[1]), 5),
                "is_stationary": bool(adf_s[1] < 0.05),
            }
        results["states"] = state_stationarity

        return results

    def demonstrate_kalman_filter(self, target_series: np.ndarray = None) -> Dict[str, Any]:
        """
        Fits a 1D Local Linear Trend / Random Walk Kalman Filter to demonstrate
        how state-space tracking separates the true latent demand state from observation noise.
        """
        print("[ANALYSIS] Running Kalman Filter State-Space Demand Tracking Demonstration...")
        if target_series is None:
            # Pick a representative moderately intermittent item from FOODS
            sample_sku = self.df_sales[self.df_sales["dept_id"] == "FOODS_1"].iloc[10]
            target_series = sample_sku[self.d_cols].values.astype(float)
            sku_name = sample_sku["id"]
        else:
            sku_name = "Custom Series"

        # Initialize 1D Random Walk Kalman Filter
        # State: x = [level], or 2D [level, trend]
        # x_{t} = A x_{t-1} + w_t,  w_t ~ N(0, Q)
        # z_t = H x_t + v_t,        v_t ~ N(0, R)
        kf = KalmanFilter(dim_x=2, dim_z=1)
        # State transition matrix: [level_t = level_{t-1} + trend_{t-1}, trend_t = trend_{t-1}]
        kf.F = np.array([[1.0, 1.0],
                         [0.0, 1.0]])
        # Measurement matrix: z_t = level_t
        kf.H = np.array([[1.0, 0.0]])
        # Initial state
        first_obs = target_series[target_series > 0]
        init_level = first_obs[0] if len(first_obs) > 0 else 1.0
        kf.x = np.array([[init_level], [0.0]])
        # Covariances
        kf.P *= 10.0
        kf.R = 4.0  # Observation noise (variance of sporadic daily demand)
        kf.Q = np.array([[0.05, 0.01],
                         [0.01, 0.01]])  # Process noise (true demand level variation)

        # Run filter
        T = len(target_series)
        filtered_states = np.zeros(T)
        state_variances = np.zeros(T)

        for t in range(T):
            z = target_series[t]
            kf.predict()
            kf.update(z)
            filtered_states[t] = kf.x[0, 0]
            state_variances[t] = kf.P[0, 0]

        return {
            "sku_id": str(sku_name),
            "raw_observations": target_series,
            "filtered_level": filtered_states,
            "state_variance": state_variances,
            "mean_observed": round(float(np.mean(target_series)), 3),
            "mean_filtered": round(float(np.mean(filtered_states)), 3),
            "zero_percentage": round(float((target_series == 0).sum() / T * 100), 2),
        }
