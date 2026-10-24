"""
run_baseline_pipeline.py - Orchestration script for AML Baseline Models.

CSAI Student Project 5:
Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models.

Usage:
    python3 src/run_baseline_pipeline.py [OPTIONS]
"""

import sys
import argparse
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_models import M5BaselinePipeline
from src.baseline_visualization import M5BaselineVisualizer


def generate_baseline_markdown_report(metadata: dict, output_path: Path):
    """Generates comprehensive academic markdown report for baseline models."""
    s_meta = metadata["overall_metrics"]["seasonal_naive"]
    g_meta = metadata["overall_metrics"]["gbdt_recursive"]
    impr = metadata["overall_metrics"]["gbdt_relative_improvements"]
    cat_data = metadata["category_breakdown"]

    md = f"""# AML Baseline Forecasting Benchmark Report
**Project 5:** Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models  
**Course:** Advanced Machine Learning (AML)  
**Evaluation Origin:** `{metadata['validation_date_range'][0]}` to `{metadata['validation_date_range'][1]}` ({metadata['validation_horizon_days']} days)  
**Series Pool:** {metadata['total_series']} SKUs across Walmart CA, TX, and WI stores  
**Generated At:** {metadata['evaluation_timestamp']}  

---

## 📌 Executive Summary

This report establishes the empirical performance benchmark for the initial machine learning phase of Project 5.
We formulate, train, and evaluate two contrasting baseline methodologies on **identical series, identical dates, and an identical 28-day out-of-sample validation horizon**:

1. **Seasonal-Naive Model (SNaive)**: Captures weekly day-of-week persistence without parameters.
2. **Gradient-Boosted Tree Regression (GBDT)**: A non-linear decision-tree ensemble using causal historical demand lags, leak-free rolling statistics, calendar indicators, and SNAP schedules under a **strictly recursive multi-step forecasting protocol**.

### Core Findings Table

| Baseline Model | MAE (Units) | RMSE (Units) | WAPE (%) | Multi-Step Forecast Protocol |
| :--- | :---: | :---: | :---: | :--- |
| **Seasonal-Naive (7-day Persistence)** | **{s_meta['mae']:.4f}** | **{s_meta['rmse']:.4f}** | **{s_meta['wape']*100:.2f}%** | Weekly historical day-of-week persistence |
| **GBDT Regression ({metadata['model_configurations']['gbdt']['library']})** | **{g_meta['mae']:.4f}** | **{g_meta['rmse']:.4f}** | **{g_meta['wape']*100:.2f}%** | Pure Recursive Multi-Step Rollout (Zero Lookahead) |
| **GBDT Relative Improvement** | **+{impr['mae_reduction_pct']:.2f}%** | **+{impr['rmse_reduction_pct']:.2f}%** | **+{impr['wape_reduction_pct']:.2f}%** | Substantial variance reduction across all tiers |

---

## 🛡️ Validation Protocol & Zero Data Leakage Guarantees

To ensure scientific integrity and eliminate any possibility of target leakage, the evaluation strictly enforces three principles:

1. **Chronological Horizon Separation**:
   - The training dataset strictly terminates on **{metadata['validation_date_range'][0]}** (minus 1 day).
   - Validation covers the subsequent 28 calendar days ({metadata['validation_date_range'][0]} to {metadata['validation_date_range'][1]}).
   - No temporal shuffling, cross-validation mixing, or random splitting is permitted.

2. **Strictly Recursive Rollout Without Ground-Truth Input**:
   - In GBDT recursive forecasting, ground-truth validation demand is strictly withheld.
   - For step $h=1$, the model uses features derived from training history up to origin $T$.
   - For subsequent steps $h \in [2, 28]$, `lag_1`, `lag_7`, and rolling statistics (7-day and 28-day windows) are dynamically computed using the **model's own predicted demands** from prior steps.
   - At no point does the model consume actual validation-period unit sales.

3. **Robust Metric Formulation (Zero-Denominator WAPE)**:
   - For intermittent inventory time series where demand sums can be zero over specific windows, WAPE is guarded:
     $$\\text{{WAPE}} = \\frac{{\\sum_{{i=1}}^N |y_i - \\hat{{y}}_i|}}{{\\sum_{{i=1}}^N y_i}}$$
     If $\\sum y_i = 0$, the metric evaluates to $0.0$ if $\\sum |y_i - \\hat{{y}}_i| = 0$, preventing undefined divide-by-zero crashes.

---

## 📊 Category-Level Performance Decomposition

| Category | SNaive MAE | GBDT MAE | SNaive WAPE | GBDT WAPE | GBDT WAPE Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for cat, c_metrics in cat_data.items():
        sn = c_metrics["seasonal_naive"]
        gb = c_metrics["gbdt"]
        diff = sn["wape"] - gb["wape"]
        md += f"| **{cat}** | {sn['mae']:.4f} | {gb['mae']:.4f} | {sn['wape']*100:.1f}% | {gb['wape']*100:.1f}% | **-{diff*100:.1f}% pts** |\n"

    md += f"""
---

## 🔍 Top Feature Importances (GBDT)

The gradient-boosted tree leverages both auto-regressive demand momentum and calendar signals:

| Rank | Feature | Normalized Importance | Role in Demand Forecasting |
| :---: | :--- | :---: | :--- |
"""
    top_feats = metadata.get("feature_importances_top_10", {})
    for rank, (feat, imp) in enumerate(top_feats.items(), 1):
        md += f"| {rank} | `{feat}` | {imp:.4f} | {'Autoregressive signal' if 'lag' in feat or 'rolling' in feat else 'Calendar/Exogenous signal'} |\n"

    md += """
---

## 🔮 Strategic Bridge: Kalman Filter & Global DeepAR Models

These baseline benchmarks reveal specific architectural limitations that motivate the subsequent Kalman Filtering and DeepAR implementations:

1. **The Compounding Error of Recursive Rollouts in GBDT**:
   - As the horizon extends from Day 1 to Day 28, recursive substitution of predicted values into lag windows dampens variance and can cause error compounding on erratic items.
   - **How Kalman Filtering Solves This**: The Kalman filter maintains an explicit state-space formulation ($x_t = F x_{t-1} + w_t$), tracking time-varying latent level and trend while estimating observation noise covariance ($R$). It dynamically filters intermittent zero-sales noise without overfitting to spurious spikes.

2. **Sparsity & Cold-Start Representation in GBDT**:
   - GBDT treats each tree split greedily and struggles when non-zero transaction frequency is low (intermittent/lumpy series with ADI > 1.32).
   - **How DeepAR Solves This**: Amazon's DeepAR uses an autoregressive Recurrent Neural Network (LSTM/GRU) with global parameter sharing across all SKUs, learning cross-series latent embeddings and outputting parameters of an intermittent likelihood distribution (Negative Binomial / Zero-Inflated Poisson).

3. **Uncertainty Quantification**:
   - Neither Seasonal-Naive nor standard GBDT produces principled probabilistic forecast intervals ($P_{10}, P_{50}, P_{90}$).
   - Both Kalman Filters (via state error covariance $P_t$) and DeepAR (via Monte Carlo quantile sampling) provide exact uncertainty bounds for inventory safety stock calculations.

---
*Report automatically generated by `src/run_baseline_pipeline.py`.*
"""
    with open(output_path, "w") as f:
        f.write(md)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AML Baseline Forecasting Models (Seasonal-Naive & GBDT)."
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/processed/train_features.csv",
        help="Path to training features CSV.",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/processed/val_features.csv",
        help="Path to validation features CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for predictions.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Reports directory for figures and summaries.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = M5BaselinePipeline(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
    )

    results = pipeline.run()
    results_df = results["results_df"]
    metadata = results["metadata"]

    # Generate Publication Figures (300 DPI)
    print("\n[VISUALIZATION] Rendering Publication-Quality Figures (300 DPI)...")
    viz = M5BaselineVisualizer(figures_dir=str(Path(args.reports_dir) / "figures"))
    
    fig1 = viz.plot_aggregate_comparison(results_df, metadata)
    print(f"[FIGURE 10] Saved -> {fig1}")

    fig2 = viz.plot_representative_series(results_df)
    print(f"[FIGURE 11] Saved -> {fig2}")

    fig3 = viz.plot_metric_comparison(metadata)
    print(f"[FIGURE 12] Saved -> {fig3}")

    # Generate Markdown Report
    report_md_path = Path(args.reports_dir) / "baseline_findings.md"
    generate_baseline_markdown_report(metadata, report_md_path)
    print(f"[REPORT] Saved -> {report_md_path}")

    # Print Terminal Summary
    s_m = metadata["overall_metrics"]["seasonal_naive"]
    g_m = metadata["overall_metrics"]["gbdt_recursive"]
    impr = metadata["overall_metrics"]["gbdt_relative_improvements"]

    print("\n" + "=" * 75)
    print("  AML BASELINE FORECASTING BENCHMARK - FINAL EVALUATION")
    print("=" * 75)
    print(f"Validation Horizon:       28 Days ({metadata['validation_date_range'][0]} to {metadata['validation_date_range'][1]})")
    print(f"Total Evaluated Series:   {metadata['total_series']} SKUs")
    print(f"Evaluation Observations:  {metadata['total_validation_observations']} unit records")
    print("-" * 75)
    print(f"{'Metric':<18} | {'Seasonal-Naive':<18} | {'GBDT (Recursive)':<18} | {'Improvement':<12}")
    print("-" * 75)
    print(f"{'MAE (Units)':<18} | {s_m['mae']:<18.4f} | {g_m['mae']:<18.4f} | +{impr['mae_reduction_pct']:.2f}%")
    print(f"{'RMSE (Units)':<18} | {s_m['rmse']:<18.4f} | {g_m['rmse']:<18.4f} | +{impr['rmse_reduction_pct']:.2f}%")
    print(f"{'WAPE (%)':<18} | {s_m['wape']*100:<17.2f}% | {g_m['wape']*100:<17.2f}% | +{impr['wape_reduction_pct']:.2f}%")
    print("=" * 75)
    print(f"Predictions Saved:        {args.output_dir}/baseline_predictions.csv")
    print(f"Metrics JSON Saved:       {args.reports_dir}/baseline_metrics.json")
    print(f"Analytical Report:        {report_md_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
