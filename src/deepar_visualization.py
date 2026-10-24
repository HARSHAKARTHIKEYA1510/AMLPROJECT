"""
deepar_visualization.py - Publication-Grade Visualizations for Global DeepAR Forecasting.

CSAI Student Project 5:
Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models.

Generates 300 DPI figures:
1. deepar_forecasts.png:
   Daily system aggregate actual demand vs Global DeepAR forecast with 80% prediction interval envelope.
2. deepar_prediction_intervals.png:
   Representative SKU forecasts with context window, forecast origin divider, actuals, P50 point predictions,
   and [P10, P90] predictive intervals.
3. 16_four_way_model_comparison.png:
   Four-panel comparative benchmark across Seasonal-Naive, GBDT, Kalman Filter, and Global DeepAR.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import PALETTE


class M5DeepARVisualizer:
    """Renders publication-grade diagnostic plots for Global DeepAR forecasts."""

    def __init__(self, figures_dir: str = "reports/figures"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def plot_aggregate_forecasts(
        self,
        results_df: pd.DataFrame,
        metrics_meta: Dict[str, Any],
        filename: str = "deepar_forecasts.png"
    ) -> Path:
        """
        Plots system-level daily aggregate actual demand vs DeepAR predictions
        with an empirical 80% prediction interval envelope ([P10, P90] aggregate).
        Lower panel displays daily absolute error residuals.
        """
        df = results_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Aggregate daily sales and uncertainty across all series
        daily = df.groupby("date").agg({
            "demand": "sum",
            "pred_deepar": "sum",
            "p10": "sum",
            "p90": "sum",
            "abs_error_deepar": "sum"
        }).reset_index()

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 9), sharex=True,
            gridspec_kw={"height_ratios": [2.5, 1.2]}
        )

        d_meta = metrics_meta["overall_metrics"]["deepar"]
        pi_80 = d_meta["prediction_intervals"]["ci_80"]

        # 1. Main Forecast Tracking Plot
        ax1.plot(
            daily["date"], daily["demand"],
            marker="o", markersize=5, linewidth=2.4, color=PALETTE["primary"],
            label="Actual Total Demand (50 SKUs)", zorder=4
        )
        ax1.plot(
            daily["date"], daily["pred_deepar"],
            marker="^", markersize=5, linewidth=2.4, color=PALETTE["accent4"],
            linestyle="--", label="DeepAR Global Point Forecast (P50)", zorder=3
        )
        ax1.fill_between(
            daily["date"], daily["p10"], daily["p90"],
            color=PALETTE["accent4"], alpha=0.20,
            label=f"Aggregate 80% Prediction Envelope (P10–P90, Coverage: {pi_80['coverage']*100:.1f}%)",
            zorder=2
        )

        ax1.set_title(
            "System-Level Daily Demand Tracking: Global DeepAR Probabilistic Forecast\n"
            f"Horizon: 28 Days (2016-03-28 to 2016-04-24) | MAE: {d_meta['mae']:.4f} | "
            f"RMSE: {d_meta['rmse']:.4f} | WAPE: {d_meta['wape']*100:.2f}% | "
            f"Mean Pinball Loss: {d_meta['pinball_loss']['mean']:.4f}",
            fontsize=13, fontweight="bold", pad=12, color=PALETTE["primary"]
        )
        ax1.set_ylabel("Total Units Sold / Day", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])
        ax1.legend(loc="upper right", frameon=True, framealpha=0.9, facecolor="white", fontsize=10)

        # 2. Residual Absolute Error Panel
        ax2.bar(
            daily["date"], daily["abs_error_deepar"],
            width=0.6, color=PALETTE["accent4"], alpha=0.75,
            edgecolor=PALETTE["primary"], linewidth=0.6,
            label="Daily Absolute Error (|y - y_hat|)"
        )
        mean_abs_err = daily["abs_error_deepar"].mean()
        ax2.axhline(
            mean_abs_err, color=PALETTE["accent1"], linestyle=":", linewidth=1.8,
            label=f"Mean Daily Error ({mean_abs_err:.1f} units)"
        )

        ax2.set_xlabel("Forecast Horizon Date (Daily)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Abs Error", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])
        ax2.legend(loc="upper right", frameon=True, framealpha=0.9, facecolor="white", fontsize=9)

        # Date formatting
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        output_path = self.figures_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return output_path

    def plot_prediction_intervals(
        self,
        results_df: pd.DataFrame,
        train_df: pd.DataFrame,
        filename: str = "deepar_prediction_intervals.png"
    ) -> Path:
        """
        Plots representative individual SKU trajectories across categories.
        Shows context training window (last 28 days), forecast origin divider,
        actual validation demand, DeepAR point forecast, and 80% prediction intervals [P10, P90].
        """
        df = results_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        tr = train_df.copy()
        tr["date"] = pd.to_datetime(tr["date"])

        # Pick 6 diverse series across categories
        selected_ids = []
        for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
            cat_series = df[df["cat_id"] == cat]["id"].unique()
            if len(cat_series) >= 2:
                selected_ids.extend(cat_series[:2])
            elif len(cat_series) == 1:
                selected_ids.append(cat_series[0])

        if len(selected_ids) < 6:
            remaining = [s for s in df["id"].unique() if s not in selected_ids]
            selected_ids.extend(remaining[: (6 - len(selected_ids))])

        fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
        axes = axes.flatten()

        for idx, sid in enumerate(selected_ids[:6]):
            ax = axes[idx]

            # Context history: last 28 days of training
            s_tr = tr[tr["id"] == sid].sort_values("date").tail(28)
            s_val = df[df["id"] == sid].sort_values("date")

            cat_name = s_val["cat_id"].iloc[0]
            store_name = s_val["store_id"].iloc[0]

            # 1. Historical Context
            ax.plot(
                s_tr["date"], s_tr["demand"],
                color=PALETTE["text"], alpha=0.45, linewidth=1.4,
                linestyle=":", label="Train Context (Last 28d)" if idx == 0 else ""
            )

            # 2. Ground Truth Validation Sales
            ax.plot(
                s_val["date"], s_val["demand"],
                color=PALETTE["primary"], marker="o", markersize=4,
                linewidth=1.8, label="Actual Demand" if idx == 0 else "", zorder=4
            )

            # 3. DeepAR Point Forecast (P50)
            ax.plot(
                s_val["date"], s_val["pred_deepar"],
                color=PALETTE["accent4"], marker="^", markersize=4,
                linewidth=2.0, linestyle="--", label="DeepAR Point (P50)" if idx == 0 else "", zorder=3
            )

            # 4. 80% Prediction Interval [P10, P90]
            ax.fill_between(
                s_val["date"], s_val["p10"], s_val["p90"],
                color=PALETTE["accent4"], alpha=0.25,
                label="80% Prediction Interval [P10, P90]" if idx == 0 else "", zorder=2
            )

            # Divider line at validation start
            split_date = s_val["date"].min()
            ax.axvline(split_date, color=PALETTE["accent1"], linestyle="-.", linewidth=1.5, alpha=0.8)

            # Compute empirical SKU coverage
            inside_80 = (s_val["demand"] >= s_val["p10"]) & (s_val["demand"] <= s_val["p90"])
            cov_80 = inside_80.mean() * 100
            mae_sku = np.abs(s_val["demand"] - s_val["pred_deepar"]).mean()

            # Clean label
            clean_id = sid.replace("_validation", "")
            ax.set_title(
                f"{clean_id} ({cat_name} | {store_name})\n"
                f"MAE: {mae_sku:.2f} units | 80% CI Coverage: {cov_80:.1f}%",
                fontsize=10, fontweight="bold", color=PALETTE["primary"], loc="left"
            )
            ax.set_ylabel("Units", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4, color=PALETTE["grid"])

        # Format x-axis for bottom row
        for ax in axes[-2:]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        fig.suptitle(
            "Global DeepAR Out-of-Sample Forecast Trajectories & 80% Prediction Intervals\n"
            "Representative M5 SKUs Across FOODS, HOBBIES, and HOUSEHOLD Categories",
            fontsize=14, fontweight="bold", color=PALETTE["primary"], y=0.99
        )
        fig.legend(
            loc="upper right", bbox_to_anchor=(0.98, 0.98),
            ncol=4, frameon=True, framealpha=0.95, facecolor="white", fontsize=10
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = self.figures_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return output_path

    def plot_four_way_comparison(
        self,
        deepar_meta: Dict[str, Any],
        baseline_meta_path: Path,
        kalman_meta_path: Path,
        filename: str = "16_four_way_model_comparison.png"
    ) -> Path:
        """
        Renders a four-panel academic benchmark comparison across:
        1. Seasonal-Naive
        2. GBDT (Recursive LightGBM)
        3. Structural Kalman Filter per SKU
        4. Global DeepAR Recurrent Network
        """
        import json
        with open(baseline_meta_path) as f:
            b_meta = json.load(f)
        with open(kalman_meta_path) as f:
            k_meta = json.load(f)

        sn_m = b_meta["overall_metrics"]["seasonal_naive"]
        gb_m = b_meta["overall_metrics"]["gbdt_recursive"]
        kf_m = k_meta["overall_metrics"]["kalman_filter"]
        da_m = deepar_meta["overall_metrics"]["deepar"]

        models = ["Seasonal-Naive", "GBDT (Recursive)", "Kalman Filter per SKU", "Global DeepAR"]
        maes = [sn_m["mae"], gb_m["mae"], kf_m["mae"], da_m["mae"]]
        rmses = [sn_m["rmse"], gb_m["rmse"], kf_m["rmse"], da_m["rmse"]]
        wapes = [sn_m["wape"] * 100, gb_m["wape"] * 100, kf_m["wape"] * 100, da_m["wape"] * 100]

        colors = [PALETTE["accent2"], PALETTE["secondary"], PALETTE["accent3"], PALETTE["accent4"]]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: MAE
        bars1 = ax1.bar(models, maes, color=colors, width=0.55, edgecolor=PALETTE["primary"], linewidth=1.0)
        ax1.set_title("Mean Absolute Error (MAE - Lower is Better)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("MAE (Units)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])
        for bar in bars1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax1.set_ylim(0, max(maes) * 1.18)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha="right")

        # Panel 2: RMSE
        bars2 = ax2.bar(models, rmses, color=colors, width=0.55, edgecolor=PALETTE["primary"], linewidth=1.0)
        ax2.set_title("Root Mean Squared Error (RMSE - Lower is Better)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("RMSE (Units)", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax2.set_ylim(0, max(rmses) * 1.18)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha="right")

        # Panel 3: WAPE
        bars3 = ax3.bar(models, wapes, color=colors, width=0.55, edgecolor=PALETTE["primary"], linewidth=1.0)
        ax3.set_title("Weighted Absolute Percentage Error (WAPE %)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("WAPE (%)", fontsize=10)
        ax3.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])
        for bar in bars3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax3.set_ylim(0, max(wapes) * 1.18)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha="right")

        # Panel 4: Category-Level WAPE Comparison
        cats = ["FOODS", "HOBBIES", "HOUSEHOLD"]
        x = np.arange(len(cats))
        w = 0.20

        b_cat = b_meta["category_breakdown"]
        k_cat = k_meta["category_breakdown"]
        d_cat = deepar_meta.get("category_breakdown", {})

        s_w = [b_cat[c]["seasonal_naive"]["wape"] * 100 for c in cats]
        g_w = [b_cat[c]["gbdt"]["wape"] * 100 for c in cats]
        k_w = [k_cat[c]["wape"] * 100 for c in cats]
        d_w = [d_cat.get(c, {}).get("wape", 0.0) * 100 for c in cats]

        ax4.bar(x - 1.5 * w, s_w, width=w, color=colors[0], label="SNaive", edgecolor=PALETTE["primary"], linewidth=0.8)
        ax4.bar(x - 0.5 * w, g_w, width=w, color=colors[1], label="GBDT", edgecolor=PALETTE["primary"], linewidth=0.8)
        ax4.bar(x + 0.5 * w, k_w, width=w, color=colors[2], label="Kalman", edgecolor=PALETTE["primary"], linewidth=0.8)
        ax4.bar(x + 1.5 * w, d_w, width=w, color=colors[3], label="DeepAR", edgecolor=PALETTE["primary"], linewidth=0.8)

        ax4.set_xticks(x)
        ax4.set_xticklabels(cats, fontweight="bold")
        ax4.set_title("Category-Level WAPE (%) Across Models", fontsize=11, fontweight="bold")
        ax4.set_ylabel("WAPE (%)", fontsize=10)
        ax4.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])
        ax4.legend(loc="upper right", frameon=True, fontsize=9)

        fig.suptitle(
            "CSAI Project 5: Four-Way Inventory Forecasting Benchmark\n"
            "Identical 50 Series | Identical 28-Day Horizon (2016-03-28 to 2016-04-24)",
            fontsize=13, fontweight="bold", color=PALETTE["primary"], y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = self.figures_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return output_path
