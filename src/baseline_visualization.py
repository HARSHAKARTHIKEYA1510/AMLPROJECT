"""
baseline_visualization.py - Publication-Quality Visualizations for AML Baseline Forecasts.

CSAI Student Project 5:
Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models.

Generates 300 DPI figures:
1. 10_baseline_aggregate_comparison.png: Daily aggregate demand vs Seasonal-Naive vs GBDT.
2. 11_baseline_series_predictions.png: Multi-series panel actual vs forecasts across categories.
3. 12_baseline_metric_comparison.png: Overall and category-level MAE, RMSE, and WAPE comparison.
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

from src.visualization import PALETTE


class M5BaselineVisualizer:
    """Renders publication-grade diagnostic plots for baseline forecasts."""

    def __init__(self, figures_dir: str = "reports/figures"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def plot_aggregate_comparison(
        self,
        results_df: pd.DataFrame,
        metrics_meta: Dict[str, Any],
        filename: str = "10_baseline_aggregate_comparison.png"
    ) -> Path:
        """
        Plots system-level daily aggregate actual demand vs Seasonal-Naive and GBDT.
        Includes a lower panel showing daily absolute error dynamics.
        """
        df = results_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Aggregate across all series per day
        daily = df.groupby("date").agg({
            "demand": "sum",
            "pred_seasonal_naive": "sum",
            "pred_gbdt": "sum",
            "abs_error_seasonal_naive": "sum",
            "abs_error_gbdt": "sum"
        }).reset_index()

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 9), sharex=True,
            gridspec_kw={"height_ratios": [2.5, 1.2]}
        )

        dates = daily["date"].values

        # Top panel: Aggregate Demand vs Predictions
        ax1.plot(
            dates, daily["demand"],
            label="Actual Total Demand", color=PALETTE["primary"],
            linewidth=2.5, marker="o", markersize=5
        )
        ax1.plot(
            dates, daily["pred_seasonal_naive"],
            label="Seasonal-Naive (7-day persistence)", color=PALETTE["accent1"],
            linewidth=2.0, linestyle="--", marker="s", markersize=4
        )
        ax1.plot(
            dates, daily["pred_gbdt"],
            label="GBDT (Recursive Multi-Step)", color=PALETTE["secondary"],
            linewidth=2.2, linestyle="-.", marker="^", markersize=4
        )

        s_meta = metrics_meta["overall_metrics"]["seasonal_naive"]
        g_meta = metrics_meta["overall_metrics"]["gbdt_recursive"]

        title_text = (
            "System-Level Demand Forecasting: Actual vs. Baselines (28-Day Horizon)\n"
            f"Seasonal-Naive: MAE={s_meta['mae']:.2f}, RMSE={s_meta['rmse']:.2f}, WAPE={s_meta['wape']*100:.1f}%  |  "
            f"GBDT (Recursive): MAE={g_meta['mae']:.2f}, RMSE={g_meta['rmse']:.2f}, WAPE={g_meta['wape']*100:.1f}%"
        )
        ax1.set_title(title_text, fontsize=13, fontweight="bold", pad=12)
        ax1.set_ylabel("Total Units Sold Across SKUs", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor=PALETTE["grid"])
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Bottom panel: Daily Absolute Errors
        bar_width = np.timedelta64(8, "h")
        ax2.bar(
            dates - bar_width / 2, daily["abs_error_seasonal_naive"],
            width=bar_width, label="Seasonal-Naive Abs Error",
            color=PALETTE["accent1"], alpha=0.65
        )
        ax2.bar(
            dates + bar_width / 2, daily["abs_error_gbdt"],
            width=bar_width, label="GBDT Abs Error",
            color=PALETTE["secondary"], alpha=0.75
        )

        ax2.set_ylabel("Aggregate Absolute Error", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Validation Forecast Date (2016)", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor=PALETTE["grid"])
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Format X-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n(%a)"))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

        plt.tight_layout()
        out_path = self.figures_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_representative_series(
        self,
        results_df: pd.DataFrame,
        filename: str = "11_baseline_series_predictions.png",
        n_series: int = 6
    ) -> Path:
        """
        Plots actual vs predicted trajectories for representative series
        spanning various product categories and intermittency levels.
        """
        df = results_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Pick diverse series: 2 Foods, 2 Hobbies, 2 Household
        selected_ids = []
        for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
            cat_series = df[df["cat_id"] == cat]["id"].unique()
            if len(cat_series) >= 2:
                selected_ids.extend(cat_series[:2])
            elif len(cat_series) == 1:
                selected_ids.append(cat_series[0])

        if len(selected_ids) < n_series:
            remaining = [s for s in df["id"].unique() if s not in selected_ids]
            selected_ids.extend(remaining[:(n_series - len(selected_ids))])

        selected_ids = selected_ids[:6]
        fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
        axes = axes.flatten()

        for idx, sid in enumerate(selected_ids):
            ax = axes[idx]
            sub = df[df["id"] == sid].sort_values("date")
            dates = sub["date"].values

            ax.plot(
                dates, sub["demand"],
                label="Actual Demand", color=PALETTE["primary"],
                linewidth=2.0, marker="o", markersize=4
            )
            ax.plot(
                dates, sub["pred_seasonal_naive"],
                label="Seasonal-Naive", color=PALETTE["accent1"],
                linewidth=1.8, linestyle="--", alpha=0.85
            )
            ax.plot(
                dates, sub["pred_gbdt"],
                label="GBDT (Recursive)", color=PALETTE["secondary"],
                linewidth=2.0, linestyle="-.", alpha=0.9
            )

            actual_sum = sub["demand"].sum()
            s_mae = np.mean(np.abs(sub["demand"] - sub["pred_seasonal_naive"]))
            g_mae = np.mean(np.abs(sub["demand"] - sub["pred_gbdt"]))

            store = sub["store_id"].iloc[0]
            cat = sub["cat_id"].iloc[0]
            ax.set_title(
                f"{sid} ({cat} | {store}) — Total Units: {int(actual_sum)} | "
                f"SNaive MAE: {s_mae:.2f} | GBDT MAE: {g_mae:.2f}",
                fontsize=11, fontweight="bold", pad=8
            )
            ax.set_ylabel("Daily Units", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)

            if idx == 0:
                ax.legend(loc="upper right", frameon=True, framealpha=0.9)

        for ax in axes[-2:]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax.set_xlabel("Validation Date (2016)", fontsize=11, fontweight="bold")

        fig.suptitle(
            "Out-of-Sample 28-Day Forecasts: Actual Demand vs. AML Baselines per SKU",
            fontsize=15, fontweight="bold", y=0.995
        )
        plt.tight_layout()
        out_path = self.figures_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_metric_comparison(
        self,
        metadata: Dict[str, Any],
        filename: str = "12_baseline_metric_comparison.png"
    ) -> Path:
        """
        Bar chart comparing MAE, RMSE, and WAPE across models and categories.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left panel: Overall Metrics Comparison
        models = ["Seasonal-Naive", "GBDT (Recursive)"]
        s_meta = metadata["overall_metrics"]["seasonal_naive"]
        g_meta = metadata["overall_metrics"]["gbdt_recursive"]

        metrics_names = ["MAE", "RMSE", "WAPE (%)"]
        s_vals = [s_meta["mae"], s_meta["rmse"], s_meta["wape"] * 100]
        g_vals = [g_meta["mae"], g_meta["rmse"], g_meta["wape"] * 100]

        x = np.arange(len(metrics_names))
        width = 0.35

        rects1 = ax1.bar(
            x - width / 2, s_vals, width,
            label="Seasonal-Naive", color=PALETTE["accent1"], alpha=0.85, edgecolor="#9f1239"
        )
        rects2 = ax1.bar(
            x + width / 2, g_vals, width,
            label="GBDT (Recursive)", color=PALETTE["secondary"], alpha=0.85, edgecolor="#0369a1"
        )

        ax1.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
        ax1.set_title("Overall Forecast Accuracy Comparison", fontsize=13, fontweight="bold", pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names, fontsize=11, fontweight="bold")
        ax1.legend(frameon=True, framealpha=0.95)
        ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

        # Value annotations
        for rect in rects1:
            h = rect.get_height()
            ax1.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )
        for rect in rects2:
            h = rect.get_height()
            ax1.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

        # Right panel: Category Breakdown WAPE (%)
        cat_data = metadata.get("category_breakdown", {})
        categories = list(cat_data.keys())
        s_cat_wape = [cat_data[c]["seasonal_naive"]["wape"] * 100 for c in categories]
        g_cat_wape = [cat_data[c]["gbdt"]["wape"] * 100 for c in categories]

        x_cat = np.arange(len(categories))
        ax2.bar(
            x_cat - width / 2, s_cat_wape, width,
            label="Seasonal-Naive", color=PALETTE["accent1"], alpha=0.85, edgecolor="#9f1239"
        )
        ax2.bar(
            x_cat + width / 2, g_cat_wape, width,
            label="GBDT (Recursive)", color=PALETTE["secondary"], alpha=0.85, edgecolor="#0369a1"
        )

        ax2.set_ylabel("WAPE (%) — Lower is Better", fontsize=11, fontweight="bold")
        ax2.set_title("Forecast Error by Product Category (WAPE)", fontsize=13, fontweight="bold", pad=10)
        ax2.set_xticks(x_cat)
        ax2.set_xticklabels(categories, fontsize=11, fontweight="bold")
        ax2.legend(frameon=True, framealpha=0.95)
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

        for i, val in enumerate(s_cat_wape):
            ax2.annotate(
                f"{val:.1f}%",
                xy=(x_cat[i] - width / 2, val),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )
        for i, val in enumerate(g_cat_wape):
            ax2.annotate(
                f"{val:.1f}%",
                xy=(x_cat[i] + width / 2, val),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

        plt.tight_layout()
        out_path = self.figures_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
