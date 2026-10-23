"""
visualization.py - Publication-Quality Visualizations for M5 Demand Forecasting.

Generates high-resolution (300 DPI) figures for:
1. Multi-level hierarchical sales dynamics.
2. Croston / Syntetos-Boylan demand categorization (ADI vs CV^2).
3. Active vs raw zero-inflation distribution.
4. Day-of-week and monthly seasonality indices.
5. SNAP uplift and major event demand shifts.
6. Price distribution and promotional discounting dynamics.
7. Store-level performance and category compositions.
8. ACF / PACF and stationarity structure.
9. Kalman Filter latent state demand tracking vs noisy observations.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any


# Define cohesive, high-contrast, modern publication palette
PALETTE = {
    "primary": "#0f172a",    # Dark Slate
    "secondary": "#0284c7",  # Vibrant Teal/Cyan
    "accent1": "#f43f5e",    # Coral/Rose
    "accent2": "#f59e0b",    # Amber
    "accent3": "#10b981",    # Emerald
    "accent4": "#8b5cf6",    # Violet
    "bg_light": "#f8fafc",   # Light background
    "grid": "#e2e8f0",       # Gridline color
    "text": "#1e293b",       # Text slate
}

# Apply global styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"],
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "text.color": PALETTE["text"],
    "axes.labelcolor": PALETTE["text"],
    "xtick.color": PALETTE["text"],
    "ytick.color": PALETTE["text"],
    "figure.titlesize": 16,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


class M5Visualizer:
    """Publication visual generator for M5 demand analysis."""

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_hierarchical_overview(self, aggs: Dict[str, pd.DataFrame]) -> Path:
        """Plot hierarchical total, state, and category sales time-series."""
        fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

        # 1. Total Daily Sales with 30-day MA
        ax_top = fig.add_subplot(gs[0, :])
        total_df = aggs["total"].copy()
        total_df["date"] = pd.to_datetime(total_df["date"])
        total_df["ma30"] = total_df["unit_sales"].rolling(30, center=True).mean()

        ax_top.plot(total_df["date"], total_df["unit_sales"], color="#94a3b8", alpha=0.45, label="Daily Total Sales", linewidth=0.9)
        ax_top.plot(total_df["date"], total_df["ma30"], color=PALETTE["secondary"], linewidth=2.5, label="30-Day Moving Average")

        # Highlight Christmas store closures
        xmas_dates = total_df[(total_df["date"].dt.month == 12) & (total_df["date"].dt.day == 25)]["date"]
        for d in xmas_dates:
            ax_top.axvline(d, color=PALETTE["accent1"], linestyle=":", alpha=0.7)
        ax_top.scatter(xmas_dates, [0]*len(xmas_dates), color=PALETTE["accent1"], s=45, zorder=5, label="Christmas Closure (Sales ~ 0)")

        ax_top.set_title("Walmart M5: System-Wide Daily Unit Sales (2011 – 2016)", fontweight="bold", pad=10)
        ax_top.set_ylabel("Total Units Sold")
        ax_top.xaxis.set_major_locator(mdates.YearLocator())
        ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_top.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")

        # 2. State-Level Sales
        ax_state = fig.add_subplot(gs[1, 0])
        state_df = aggs["state"].rolling(30, center=True).mean()
        state_colors = {"CA": PALETTE["secondary"], "TX": PALETTE["accent2"], "WI": PALETTE["accent3"]}
        for state in ["CA", "TX", "WI"]:
            ax_state.plot(state_df.index, state_df[state], label=f"State {state}", color=state_colors[state], linewidth=2.0)
        ax_state.set_title("State-Level Demand Dynamics (30-Day Rolling Mean)", fontweight="bold", pad=8)
        ax_state.set_ylabel("Units Sold per Day")
        ax_state.xaxis.set_major_locator(mdates.YearLocator())
        ax_state.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_state.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")

        # 3. Category-Level Sales
        ax_cat = fig.add_subplot(gs[1, 1])
        cat_df = aggs["category"].rolling(30, center=True).mean()
        cat_colors = {"FOODS": PALETTE["accent1"], "HOUSEHOLD": PALETTE["accent4"], "HOBBIES": PALETTE["accent2"]}
        for cat in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
            ax_cat.plot(cat_df.index, cat_df[cat], label=f"Category {cat}", color=cat_colors[cat], linewidth=2.0)
        ax_cat.set_title("Product Category Proportions (30-Day Rolling Mean)", fontweight="bold", pad=8)
        ax_cat.set_ylabel("Units Sold per Day")
        ax_cat.xaxis.set_major_locator(mdates.YearLocator())
        ax_cat.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_cat.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")

        out_path = self.output_dir / "01_hierarchical_sales_overview.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[FIGURE] Saved {out_path.name}")
        return out_path

    def plot_croston_quadrants(self, demand_df: pd.DataFrame) -> Path:
        """Plot Croston / Syntetos-Boylan demand classification scatter and distribution."""
        fig, (ax_scatter, ax_pie) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.4, 1]})

        # Clip extreme values for visualization clarity
        plot_df = demand_df.copy()
        plot_df["ADI_clipped"] = np.clip(plot_df["ADI"], 0, 10)
        plot_df["CV2_clipped"] = np.clip(plot_df["CV2"], 0, 4)

        quadrant_colors = {
            "Smooth": PALETTE["accent3"],
            "Intermittent": PALETTE["secondary"],
            "Erratic": PALETTE["accent2"],
            "Lumpy": PALETTE["accent1"],
        }

        # 1. Scatter plot
        for q, color in quadrant_colors.items():
            sub = plot_df[plot_df["quadrant"] == q]
            # sample up to 2000 points per quadrant for clean plotting
            sample_sub = sub.sample(min(len(sub), 2000), random_state=42)
            ax_scatter.scatter(sample_sub["ADI_clipped"], sample_sub["CV2_clipped"],
                               color=color, alpha=0.35, s=18, label=f"{q} ({len(sub):,} SKUs)")

        # Quadrant threshold lines
        ax_scatter.axvline(1.32, color=PALETTE["primary"], linestyle="--", linewidth=1.8, label="ADI = 1.32 Cutoff")
        ax_scatter.axhline(0.49, color=PALETTE["primary"], linestyle=":", linewidth=1.8, label="CV² = 0.49 Cutoff")

        # Quadrant Annotations
        ax_scatter.text(0.6, 0.2, "SMOOTH\n(Regular, Low Variance)", color=PALETTE["accent3"], fontweight="bold", fontsize=10, ha="center")
        ax_scatter.text(5.5, 0.2, "INTERMITTENT\n(Sporadic, Constant Size)", color=PALETTE["secondary"], fontweight="bold", fontsize=10, ha="center")
        ax_scatter.text(0.6, 2.5, "ERRATIC\n(Regular, High Variance)", color=PALETTE["accent2"], fontweight="bold", fontsize=10, ha="center")
        ax_scatter.text(5.5, 2.5, "LUMPY\n(Sporadic, High Variance)", color=PALETTE["accent1"], fontweight="bold", fontsize=10, ha="center")

        ax_scatter.set_title("Syntetos-Boylan / Croston Demand Categorization Matrix", fontweight="bold")
        ax_scatter.set_xlabel("Average Demand Interval (ADI) [Days between sales]")
        ax_scatter.set_ylabel("Squared Coefficient of Variation (CV²)")
        ax_scatter.set_xlim(0, 10)
        ax_scatter.set_ylim(0, 4)
        ax_scatter.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)

        # 2. Donut / Pie distribution
        counts = demand_df["quadrant"].value_counts()
        colors = [quadrant_colors[q] for q in counts.index]
        wedges, texts, autotexts = ax_pie.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
            pctdistance=0.75,
            textprops=dict(fontweight="bold")
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(11)

        ax_pie.set_title("Proportion of SKUs by Demand Pattern", fontweight="bold")

        fig.tight_layout()
        out_path = self.output_dir / "02_intermittent_croston_matrix.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[FIGURE] Saved {out_path.name}")
        return out_path

    def plot_zero_demand_distribution(self, df_sales: pd.DataFrame) -> Path:
        """Plot raw vs active zero-sales percentage distribution across all 30,490 series."""
        fig, ax = plt.subplots(figsize=(10, 5.5))
        d_cols = [c for c in df_sales.columns if c.startswith("d_")]
        raw_vals = df_sales[d_cols].values

        # Raw zero percentage
        raw_zeros = (raw_vals == 0).sum(axis=1) / len(d_cols) * 100

        # Active zeros (excluding leading zeros before first sale)
        first_sale = np.argmax(raw_vals > 0, axis=1)
        active_days = len(d_cols) - first_sale
        # Count non-zero sales
        non_zeros = (raw_vals > 0).sum(axis=1)
        active_zeros = np.where(active_days > 0, (active_days - non_zeros) / active_days * 100, 100.0)

        sns.kdeplot(raw_zeros, fill=True, color=PALETTE["accent1"], alpha=0.4, label="Raw Observed Zeros (Includes Pre-Launch)", ax=ax, linewidth=2)
        sns.kdeplot(active_zeros, fill=True, color=PALETTE["secondary"], alpha=0.4, label="Active Shelf Zeros (Excludes Pre-Launch)", ax=ax, linewidth=2)

        ax.axvline(np.median(raw_zeros), color=PALETTE["accent1"], linestyle="--", linewidth=1.5, label=f"Raw Median ({np.median(raw_zeros):.1f}%)")
        ax.axvline(np.median(active_zeros), color=PALETTE["secondary"], linestyle="--", linewidth=1.5, label=f"Active Median ({np.median(active_zeros):.1f}%)")

        ax.set_title("Intermittent Demand Density: Raw vs. Active Shelf Zero-Sales Rates", fontweight="bold", pad=12)
        ax.set_xlabel("Zero-Sales Rate (%)")
        ax.set_ylabel("Kernel Density")
        ax.set_xlim(0, 100)
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")

        fig.tight_layout()
        out_path = self.output_dir / "03_zero_demand_distribution.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[FIGURE] Saved {out_path.name}")
        return out_path

    def plot_seasonality(self, season_data: Dict[str, Any]) -> Path:
        """Plot Day-of-Week and Monthly seasonality indices."""
        fig, (ax_dow, ax_month) = plt.subplots(1, 2, figsize=(14, 5))

        dow_df = pd.DataFrame(season_data["day_of_week_index"])
        month_df = pd.DataFrame(season_data["monthly_index"])

        # 1. Day of Week
        colors_dow = [PALETTE["secondary"] if i < 5 else PALETTE["accent1"] for i in dow_df["dayofweek"]]
        bars_dow = ax_dow.bar(dow_df["day_name"], dow_df["index"], color=colors_dow, edgecolor="white", width=0.65)
        ax_dow.axhline(1.0, color=PALETTE["primary"], linestyle="--", linewidth=1.2, label="Base Average (1.0)")
        ax_dow.set_title("Weekly Seasonality: Weekend Sales Surge", fontweight="bold", pad=10)
        ax_dow.set_ylabel("Demand Index (Sales / Mean)")
        ax_dow.set_ylim(0.8, 1.3)
        for bar in bars_dow:
            h = bar.get_height()
            ax_dow.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax_dow.legend(loc="upper left", frameon=True, facecolor="white")

        # 2. Month of Year
        bars_month = ax_month.bar(month_df["month_name"].str[:3], month_df["index"], color=PALETTE["accent4"], edgecolor="white", width=0.65)
        ax_month.axhline(1.0, color=PALETTE["primary"], linestyle="--", linewidth=1.2, label="Base Average (1.0)")
        ax_month.set_title("Monthly Seasonality: Annual Cycles", fontweight="bold", pad=10)
        ax_month.set_ylabel("Demand Index (Sales / Mean)")
        ax_month.set_ylim(0.85, 1.15)
        for bar in bars_month:
            h = bar.get_height()
            ax_month.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax_month.legend(loc="upper left", frameon=True, facecolor="white")

        fig.tight_layout()
        out_path = self.output_dir / "04_seasonality_and_calendar_effects.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[FIGURE] Saved {out_path.name}")
        return out_path

    def plot_snap_and_events(self, season_data: Dict[str, Any]) -> Path:
        """Plot SNAP uplift across states and top holiday uplifts."""
        fig, (ax_snap, ax_event) = plt.subplots(1, 2, figsize=(14, 5.5))

        # 1. SNAP by State
        snap_data = season_data["snap_analysis"]
        states = list(snap_data.keys())
        uplifts = [snap_data[s]["uplift_pct"] for s in states]
        colors = [PALETTE["secondary"], PALETTE["accent2"], PALETTE["accent3"]]

        bars = ax_snap.bar(states, uplifts, color=colors, edgecolor="white", width=0.55)
        ax_snap.axhline(0, color="black", linewidth=0.8)
        ax_snap.set_title("SNAP Program Impact: Food & Retail Demand Uplift", fontweight="bold", pad=10)
        ax_snap.set_ylabel("Average Daily Sales Uplift (%)")
        for bar in bars:
            h = bar.get_height()
            ax_snap.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"+{h:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # 2. Top Holiday Uplifts
        top_events = season_data["top_event_uplifts"]
        event_names = list(top_events.keys())[:8]
        event_vals = [top_events[k] for k in event_names]

        y_pos = np.arange(len(event_names))
        ax_event.barh(y_pos, event_vals, color=PALETTE["accent1"], edgecolor="white", height=0.6)
        ax_event.set_yticks(y_pos)
        ax_event.set_yticklabels(event_names)
        ax_event.invert_yaxis()
        ax_event.set_title("Top Calendar Event Demand Spikes", fontweight="bold", pad=10)
        ax_event.set_xlabel("Sales Uplift vs. Chain Baseline (%)")
        for i, v in enumerate(event_vals):
            ax_event.text(v + 0.5, i, f"+{v:.1f}%", va="center", fontsize=9, fontweight="bold")

        fig.tight_layout()
        out_path = self.output_dir / "05_holiday_and_snap_impact.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[FIGURE] Saved {out_path.name}")
        return out_path

    def plot_price_elasticity(self, df_prices: pd.DataFrame, df_sales: pd.DataFrame) -> Path:
        """Plot price distribution across departments and promotional price elasticity."""
        fig, (ax_kde, ax_dept) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Sample price data
        price_sample = df_prices.sample(n=min(len(df_prices), 100000), random_state=42)
        price_sample["cat_id"] = price_sample["item_id"].str.split("_").str[0]

        cat_colors = {"FOODS": PALETTE["accent1"], "HOUSEHOLD": PALETTE["accent4"], "HOBBIES": PALETTE["accent2"]}
        for cat, col in cat_colors.items():
            sub = price_sample[price_sample["cat_id"] == cat]
            sns.kdeplot(sub["sell_price"], ax=ax_kde, color=col, fill=True, alpha=0.3, label=f"Category {cat}", linewidth=2)

        ax_kde.set_title("Sell Price Distribution by Category", fontweight="bold")
        ax_kde.set_xlabel("Sell Price ($)")
        ax_kde.set_ylabel("Density")
        ax_kde.set_xlim(0, 20)
        ax_kde.legend(loc="upper right", frameon=True, facecolor="white")

        # Department Median & IQR Prices
        dept_prices = df_prices.sample(50000, random_state=42).copy()
        dept_prices["dept_id"] = dept_prices["item_id"].str.extract(r"([A-Z]+_[0-9])")
        dept_order = sorted(dept_prices["dept_id"].dropna().unique())

        sns.boxplot(data=dept_prices, x="dept_id", y="sell_price", order=dept_order, ax=ax_dept,
                    palette="Blues_r", showfliers=False)
        ax_dept.set_title("Price Dispersion Across Retail Departments (Excluding Outliers)", fontweight="bold")
        ax_dept.set_xlabel("Department ID")
        ax_dept.set_ylabel("Sell Price ($)")
        ax_dept.tick_params(axis="x", rotation=30)

        fig.tight_layout()
        out_path = self.output_dir / "06_price_elasticity_analysis.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[FIGURE] Saved {out_path.name}")
        return out_path

    def plot_store_comparison(self, aggs: Dict[str, pd.DataFrame]) -> Path:
        """Plot Store-level sales distribution and volume."""
        fig, ax = plt.subplots(figsize=(12, 5.5))
        store_df = aggs["store"]

        store_means = store_df.mean().sort_values(ascending=False)
        store_stds = store_df.std()[store_means.index]

        colors = [PALETTE["secondary"] if "CA" in s else (PALETTE["accent2"] if "TX" in s else PALETTE["accent3"]) for s in store_means.index]

        bars = ax.bar(store_means.index, store_means.values, yerr=store_stds.values, capsize=5,
                      color=colors, edgecolor="white", width=0.6, error_kw=dict(ecolor="#64748b", lw=1.5))

        ax.set_title("Store Performance: Daily Unit Volume & Variance (Grouped by State)", fontweight="bold", pad=12)
        ax.set_ylabel("Average Daily Units Sold")
        ax.set_xlabel("Store Identifier")

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h/2, f"{int(h):,}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

        # Legend for states
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=PALETTE["secondary"], label="California (CA)"),
            Patch(facecolor=PALETTE["accent2"], label="Texas (TX)"),
            Patch(facecolor=PALETTE["accent3"], label="Wisconsin (WI)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white")

        fig.tight_layout()
        out_path = self.output_dir / "07_store_level_comparison.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[FIGURE] Saved {out_path.name}")
        return out_path

    def plot_acf_pacf(self, stationarity_data: Dict[str, Any]) -> Path:
        """Plot Autocorrelation Function (ACF) showing multi-lag weekly periodicity."""
        fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(14, 5))

        acf_dict = stationarity_data["total_level"]["autocorrelations"]
        pacf_dict = stationarity_data["total_level"]["partial_autocorrelations"]

        # Bar plot of key ACF lags
        lags = list(acf_dict.keys())
        vals = list(acf_dict.values())
        colors = [PALETTE["accent1"] if "7" in l else PALETTE["secondary"] for l in lags]

        bars = ax_acf.bar(lags, vals, color=colors, edgecolor="white", width=0.55)
        ax_acf.axhline(0, color="black", linewidth=0.8)
        ax_acf.set_title("Autocorrelation (ACF) of Total Sales", fontweight="bold")
        ax_acf.set_ylabel("Correlation Coefficient")
        ax_acf.set_ylim(0, 1.0)
        for bar in bars:
            h = bar.get_height()
            ax_acf.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax_acf.tick_params(axis="x", rotation=20)

        # ADF test scorecard display in PACF axis
        ax_pacf.axis("off")
        raw_adf = stationarity_data["total_level"]["raw_series"]
        diff_adf = stationarity_data["total_level"]["differenced_series"]

        summary_text = (
            "Augmented Dickey-Fuller (ADF) Stationarity Analysis\n"
            "----------------------------------------------------\n\n"
            f"1. Raw Level Total Sales:\n"
            f"   • ADF Statistic: {raw_adf['adf_statistic']}\n"
            f"   • p-value:       {raw_adf['p_value']:.4f}\n"
            f"   • 5% Critical:   {raw_adf['critical_values']['5%']}\n"
            f"   • Stationary:    {raw_adf['is_stationary']} (Strong trend / cycles)\n\n"
            f"2. First-Differenced Total Sales (Δy_t):\n"
            f"   • ADF Statistic: {diff_adf['adf_statistic']}\n"
            f"   • p-value:       {diff_adf['p_value']:.4e}\n"
            f"   • Stationary:    {diff_adf['is_stationary']} (Stationary series!)\n\n"
            "Key Insights for Phase 1 Modeling:\n"
            "• Prominent 7-day cyclicality requires seasonal differencing or\n"
            "  harmonic state matrices in Kalman filtering.\n"
            "• Strong stationarity post-differencing proves appropriateness\n"
            "  of state-space innovation models."
        )
        ax_pacf.text(0.05, 0.95, summary_text, transform=ax_pacf.transAxes,
                     fontsize=10.5, verticalalignment="top", fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=1", facecolor=PALETTE["bg_light"], edgecolor="#cbd5e1"))

        fig.tight_layout()
        out_path = self.output_dir / "08_acf_pacf_stationarity.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[FIGURE] Saved {out_path.name}")
        return out_path

    def plot_kalman_filter_tracking(self, kf_results: Dict[str, Any]) -> Path:
        """
        Plot Kalman Filter state estimation on noisy, intermittent demand series.
        Visualizes raw observations vs latent demand level and 95% confidence intervals.
        """
        fig, ax = plt.subplots(figsize=(14, 5.5))

        raw = kf_results["raw_observations"]
        level = kf_results["filtered_level"]
        var = kf_results["state_variance"]
        std = np.sqrt(np.maximum(var, 0))

        # Show the most active 120-day window
        window = slice(100, 220)
        t_steps = np.arange(len(raw))[window]

        ax.bar(t_steps, raw[window], color="#94a3b8", alpha=0.5, width=0.8, label="Observed Noisy Demand (Sales Units)")
        ax.plot(t_steps, level[window], color=PALETTE["secondary"], linewidth=2.5, label="Kalman Filter Latent State (Estimated True Demand)")
        ax.fill_between(t_steps, np.maximum(0, level[window] - 1.96*std[window]), level[window] + 1.96*std[window],
                        color=PALETTE["secondary"], alpha=0.2, label="95% State Confidence Bound (±1.96σ)")

        ax.set_title(f"Kalman Filter State-Space Tracking on Intermittent Demand ({kf_results['sku_id']})", fontweight="bold", pad=12)
        ax.set_xlabel("Time Horizon (Consecutive Days)")
        ax.set_ylabel("Demand Units")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1")

        fig.tight_layout()
        out_path = self.output_dir / "09_kalman_filter_motivation.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[FIGURE] Saved {out_path.name}")
        return out_path
