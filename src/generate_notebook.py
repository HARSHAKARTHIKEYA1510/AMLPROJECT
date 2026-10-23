"""
generate_notebook.py - Script to construct 01_data_cleaning_and_eda.ipynb.
"""

import nbformat as nbf
from pathlib import Path


def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# Project 5: Inventory Demand Forecasting via Kalman Filters & DeepAR Global Models
## Phase 1: Dataset Collection, Data Quality Audit, and Exploratory Data Analysis (EDA)

**Course:** CSAI - Machine Learning Evaluation (Phase 1)  
**Dataset:** Walmart M5 Forecasting - Accuracy (Kaggle / Hugging Face)  
**Focus Area:** Data Ingestion, Schema Conformance, Intermittent Demand Characterization, and State-Space Motivation  
"""))

    # Setup and Imports
    cells.append(nbf.v4.new_code_cell("""import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
PROJECT_ROOT = Path('.').resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_calendar, load_sell_prices, load_sales_train
from src.data_quality import M5DataQualityAuditor
from src.eda_analysis import M5ExploratoryAnalysis
from src.visualization import M5Visualizer

print('Environment initialized successfully.')
"""))

    # Step 1: Loading
    cells.append(nbf.v4.new_markdown_cell("""## 1. Automated Dataset Loading & Memory Downcasting
The M5 dataset spans 30,490 time-series across 1,913 days (58.3 million entries). In raw 64-bit precision, loading these tables exceeds standard RAM capacity. We employ an intelligent downcasting strategy (`int64` -> `int16`/`int8`, `float64` -> `float32`, objects -> categories) achieving a ~75% memory footprint reduction.
"""))

    cells.append(nbf.v4.new_code_cell("""df_cal = load_calendar('../data/raw/calendar.csv')
df_prices = load_sell_prices('../data/raw/sell_prices.csv')
df_sales = load_sales_train('../data/raw/sales_train_validation.csv')

print(f'Calendar shape: {df_cal.shape}')
print(f'Prices shape:   {df_prices.shape}')
print(f'Sales shape:    {df_sales.shape}')
"""))

    # Step 2: Quality Audit
    cells.append(nbf.v4.new_markdown_cell("""## 2. Comprehensive Data Quality Audit
We evaluate:
- **Completeness:** Missing value rates in sales, calendar, and price tables.
- **Validity:** Ensuring strictly non-negative unit sales and positive sell prices.
- **Structural Integrity:** Verifying 3,049 items * 10 stores = 30,490 valid time-series.
- **Pre-Release Inactivity:** Differentiating leading zeros (unreleased items) from genuine zero-demand days.
"""))

    cells.append(nbf.v4.new_code_cell("""auditor = M5DataQualityAuditor(raw_dir='../data/raw', reports_dir='../reports')
cal_audit = auditor.audit_calendar(df_cal)
price_audit = auditor.audit_sell_prices(df_prices)
sales_audit = auditor.audit_sales_train(df_sales)

print(f"Total Observations: {sales_audit['total_observations']:,}")
print(f"Missing Values:     {sales_audit['data_cleanliness']['missing_values']}")
print(f"Negative Sales:     {sales_audit['data_cleanliness']['negative_sales']}")
print(f"Raw Zero Rate:      {sales_audit['intermittent_demand_profile']['raw_zero_sales_pct']}%")
print(f"Active Zero Rate:   {sales_audit['intermittent_demand_profile']['active_zero_sales_pct']}%")
"""))

    # Step 3: Croston / Intermittent Demand
    cells.append(nbf.v4.new_markdown_cell("""## 3. Demand Classification: Syntetos-Boylan / Croston Framework
Retail inventory demand is classified into four distinct regimes using:
- **Average Demand Interval (ADI):** Cutoff = 1.32
- **Squared Coefficient of Variation ($CV^2$):** Cutoff = 0.49
"""))

    cells.append(nbf.v4.new_code_cell("""eda = M5ExploratoryAnalysis(df_cal, df_prices, df_sales)
aggs = eda.compute_hierarchical_aggregations()
demand_df = eda.classify_demand_patterns(sample_size=None)

croston_summary = demand_df['quadrant'].value_counts(normalize=True).mul(100).round(2).to_frame(name='Percentage (%)')
croston_summary['SKU_Count'] = demand_df['quadrant'].value_counts()
print(croston_summary)
"""))

    # Step 4: Seasonality and Events
    cells.append(nbf.v4.new_markdown_cell("""## 4. Seasonality, SNAP Policy Uplift, and Holiday Effects
Analysis of day-of-week demand multipliers, SNAP program distribution surges, and store holiday closures.
"""))

    cells.append(nbf.v4.new_code_cell("""season_data = eda.analyze_seasonality_and_events(aggs)
print("SNAP Demand Uplifts:")
for state, metrics in season_data["snap_analysis"].items():
    print(f"  State {state}: +{metrics['uplift_pct']}%")

print("\nTop Event Uplifts:")
for ev, up in list(season_data["top_event_uplifts"].items())[:5]:
    print(f"  {ev}: +{up}%")
"""))

    # Step 5: Stationarity & Autocorrelation
    cells.append(nbf.v4.new_markdown_cell("""## 5. Stationarity and Autocorrelation Analysis
Evaluation of Augmented Dickey-Fuller (ADF) tests and periodic ACF harmonics.
"""))

    cells.append(nbf.v4.new_code_cell("""stationarity_data = eda.compute_stationarity_and_autocorrelation(aggs)
raw_adf = stationarity_data["total_level"]["raw_series"]
diff_adf = stationarity_data["total_level"]["differenced_series"]

print(f"Raw Series ADF Stat:          {raw_adf['adf_statistic']} (p-val: {raw_adf['p_value']:.4f}, Stationary: {raw_adf['is_stationary']})")
print(f"Differenced Series ADF Stat:  {diff_adf['adf_statistic']} (p-val: {diff_adf['p_value']:.4e}, Stationary: {diff_adf['is_stationary']})")
print("ACF Lags (Weekly Harmonics):", stationarity_data["total_level"]["autocorrelations"])
"""))

    # Step 6: Kalman Filter Demonstration
    cells.append(nbf.v4.new_markdown_cell("""## 6. Motivation for Kalman Filters & DeepAR Global Models
- **Kalman Filtering:** Separates underlying demand velocity from zero-inflated observation noise and provides dynamic safety-stock error bands.
- **DeepAR Global Models (Phase 2):** Learns cross-series representations across 30,490 SKUs simultaneously using negative binomial likelihoods.
"""))

    cells.append(nbf.v4.new_code_cell("""kf_demo = eda.demonstrate_kalman_filter()
print(f"Demonstration SKU:    {kf_demo['sku_id']}")
print(f"Zero-Sales Rate:      {kf_demo['zero_percentage']}%")
print(f"Mean Observed Sales:  {kf_demo['mean_observed']}")
print(f"Mean Filtered Level:  {kf_demo['mean_filtered']}")
"""))

    # Step 7: Visual Artifacts
    cells.append(nbf.v4.new_markdown_cell("""## 7. High-Resolution Visual Artifacts
All figures are pre-rendered at 300 DPI in `reports/figures/`.
"""))

    cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display

figure_files = [
    "../reports/figures/01_hierarchical_sales_overview.png",
    "../reports/figures/02_intermittent_croston_matrix.png",
    "../reports/figures/03_zero_demand_distribution.png",
    "../reports/figures/04_seasonality_and_calendar_effects.png",
    "../reports/figures/05_holiday_and_snap_impact.png",
    "../reports/figures/06_price_elasticity_analysis.png",
    "../reports/figures/07_store_level_comparison.png",
    "../reports/figures/08_acf_pacf_stationarity.png",
    "../reports/figures/09_kalman_filter_motivation.png",
]

for fig_path in figure_files:
    if Path(fig_path).exists():
        print(f"Displaying: {Path(fig_path).name}")
        display(Image(filename=fig_path, width=800))
"""))

    nb["cells"] = cells

    out_file = Path("notebooks/01_data_cleaning_and_eda.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        nbf.write(nb, f)

    print(f"Successfully generated {out_file.resolve()}")


if __name__ == "__main__":
    create_notebook()
