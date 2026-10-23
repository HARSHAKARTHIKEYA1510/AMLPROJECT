"""
generate_notebooks.py - Programmatically creates the two distinct, decoupled notebooks:
1. notebooks/01_data_cleaning.ipynb: Dedicated to ingestion, schema validation, downcasting, stockouts, cold-starts.
2. notebooks/02_exploratory_data_analysis.ipynb: Dedicated to hierarchical dynamics, Croston matrix, seasonality, SNAP uplift, ADF, and Kalman filter.
"""

import nbformat as nbf
from pathlib import Path


def generate_cleaning_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# Project 5: Inventory Demand Forecasting
## Notebook 1: Dataset Collection, Data Cleaning & Preprocessing

**Course:** CSAI - Machine Learning Evaluation (Phase 1)  
**Dataset:** Walmart M5 Forecasting - Accuracy  
**Module Focus:**
- Ingestion from `data/raw/` (calendar, sell_prices, sales_train_validation).
- Schema integrity, completeness, and downcasting (~75% RAM reduction).
- Product lifecycle evaluation: Cold-start detection and pre-launch identification.
- **Fair handling of stockouts:** Marking rather than imputing stockout periods.
- Feature engineering: Cyclical calendar encodings and causal lags (without lookahead).
- Exporting structured, memory-optimized tables to `data/processed/`.
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path('.').resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_calendar, load_sell_prices, load_sales_train, reduce_mem_usage
from src.data_quality import M5DataQualityAuditor
from src.data_cleaning import M5DataCleaner

print("Data Cleaning environment initialized.")
"""))

    # Step 1: Ingestion
    cells.append(nbf.v4.new_markdown_cell("""### 1. Raw Data Ingestion (`data/raw/`)
We load the raw CSV files and inspect schema dimensions.
- `calendar.csv`: 1,969 days with calendar attributes, SNAP dates, and events.
- `sell_prices.csv`: 6,841,121 weekly price observations per store and item.
- `sales_train_validation.csv`: 30,490 time series over 1,913 days (58,327,370 unit observations).
"""))

    cells.append(nbf.v4.new_code_cell("""raw_dir = Path("../data/raw")

df_cal_raw = pd.read_csv(raw_dir / "calendar.csv")
df_prices_raw = pd.read_csv(raw_dir / "sell_prices.csv")
df_sales_raw = pd.read_csv(raw_dir / "sales_train_validation.csv")

print(f"Raw Calendar: {df_cal_raw.shape}")
print(f"Raw Prices:   {df_prices_raw.shape}")
print(f"Raw Sales:    {df_sales_raw.shape}")
"""))

    # Step 2: Quality Audit
    cells.append(nbf.v4.new_markdown_cell("""### 2. Comprehensive Data Quality Audit
We verify:
1. **Completeness:** 0 null values across all 58.3 million unit sales entries.
2. **Validity:** Strictly non-negative integer sales and positive prices ($0.01 to $107.32).
3. **Continuity:** 1,969 consecutive days without date gaps.
"""))

    cells.append(nbf.v4.new_code_cell("""auditor = M5DataQualityAuditor(raw_dir="../data/raw", reports_dir="../reports")
cal_audit = auditor.audit_calendar(load_calendar(str(raw_dir / "calendar.csv")))
price_audit = auditor.audit_sell_prices(load_sell_prices(str(raw_dir / "sell_prices.csv")))
sales_audit = auditor.audit_sales_train(load_sales_train(str(raw_dir / "sales_train_validation.csv")))

print(f"Total Observations Audited: {sales_audit['total_observations']:,}")
print(f"Sales Missing Values:       {sales_audit['data_cleanliness']['missing_values']}")
print(f"Negative Sales Detected:    {sales_audit['data_cleanliness']['negative_sales']}")
print(f"Negative Prices Detected:   {price_audit['anomalies']['negative_prices']}")
"""))

    # Step 3: Lifecycle, Cold-Starts & Stockouts
    cells.append(nbf.v4.new_markdown_cell("""### 3. Product Lifecycle, Cold Starts & Stockout Marking
In retail demand modeling:
- **Pre-launch Inactivity vs. Zero Customer Demand:** Items introduced late in the 5-year timeline have leading zeros before their first launch. Misclassifying these as 'zero demand' biases models.
- **Cold-Start Items:** Items active for less than 1 year (365 days) require global representation sharing.
- **Fair Stockout Handling:** Rather than imputing missing sales, we construct an explicit `is_unlisted_stockout` indicator for periods where an active item disappears from store price listings.
"""))

    cells.append(nbf.v4.new_code_cell("""cleaner = M5DataCleaner(raw_dir="../data/raw", processed_dir="../data/processed")
sku_meta = cleaner.analyze_lifecycle_and_stockouts(df_sales_raw)

print(f"Total SKUs Analyzed:        {len(sku_meta):,}")
print(f"Cold-Start SKUs (<365 days): {sku_meta['is_cold_start'].sum():,} ({sku_meta['is_cold_start'].mean()*100:.2f}%)")
display(sku_meta.head(10))
"""))

    # Step 4: Full Cleaning & Export
    cells.append(nbf.v4.new_markdown_cell("""### 4. End-to-End Pipeline Execution & Parquet Export (`data/processed/`)
We run the full cleaning pipeline and export memory-optimized Parquet tables into `data/processed/`:
1. `cleaned_calendar.parquet`: Cyclical day/month trigonometric features, SNAP flags, Christmas marker.
2. `cleaned_prices.parquet`: Regular price baselines, discount depth percentages, promo markers.
3. `sku_lifecycle_metadata.parquet`: Cold-start flags and active shelf duration.
4. `cleaned_sales_matrix.parquet`: Downcasted sales matrix.
5. `m5_starter_modeling_panel.parquet`: Complete panel for Store `CA_1` `FOODS_1` (50 SKUs) with causal lag and rolling features without lookahead.
"""))

    cells.append(nbf.v4.new_code_cell("""summary = cleaner.run_cleaning_pipeline()

import os
processed_files = os.listdir("../data/processed")
print("\\nGenerated Clean Files in data/processed/:")
for f in sorted(processed_files):
    sz = (Path("../data/processed") / f).stat().st_size / (1024*1024)
    print(f"  • {f:<35} ({sz:.2f} MB)")
"""))

    nb["cells"] = cells
    out_path = Path("notebooks/01_data_cleaning.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        nbf.write(nb, f)
    print(f"[NOTEBOOK CREATED] {out_path.resolve()}")


def generate_eda_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# Project 5: Inventory Demand Forecasting
## Notebook 2: Exploratory Data Analysis & Statistical Modeling Insights

**Course:** CSAI - Machine Learning Evaluation (Phase 1)  
**Dataset Source:** Preprocessed Datasets from `data/processed/`  
**Core Analytical Objectives:**
1. Multi-level hierarchical sales dynamics (System, State, Store, Category, Department).
2. **Syntetos-Boylan / Croston Demand Matrix:** Classifying all 30,490 SKUs by $ADI$ vs $CV^2$.
3. Active vs. Raw Zero-Inflation distributions.
4. Weekly (Day-of-Week) and Annual (Monthly) seasonality indices.
5. **SNAP Policy Demand Uplift** across California, Texas, and Wisconsin.
6. Price elasticity and promotional discounting effects.
7. Time-series stationarity testing (ADF test) and ACF weekly harmonics.
8. **Kalman Filter Latent State Tracking:** Theoretical and empirical demonstration for Phase 1.
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

# Add project root to sys.path
PROJECT_ROOT = Path('.').resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eda_analysis import M5ExploratoryAnalysis
from src.visualization import M5Visualizer

print("EDA Environment initialized.")
"""))

    # Step 1: Loading Processed Data
    cells.append(nbf.v4.new_markdown_cell("""### 1. Ingestion from `data/processed/`
We load the clean, validated Parquet datasets directly from `data/processed/`.
"""))

    cells.append(nbf.v4.new_code_cell("""proc_dir = Path("../data/processed")

df_cal = pd.read_parquet(proc_dir / "cleaned_calendar.parquet")
df_prices = pd.read_parquet(proc_dir / "cleaned_prices.parquet")
df_sales = pd.read_parquet(proc_dir / "cleaned_sales_matrix.parquet")

print(f"Loaded Processed Calendar: {df_cal.shape}")
print(f"Loaded Processed Prices:   {df_prices.shape}")
print(f"Loaded Processed Sales:    {df_sales.shape}")
"""))

    # Step 2: Hierarchical Aggregation
    cells.append(nbf.v4.new_markdown_cell("""### 2. Multi-Level Hierarchical Sales Dynamics
M5 demand spans 5 hierarchical tiers. We aggregate daily volume across:
- **Level 1:** Total Chain
- **Level 2:** State (CA, TX, WI)
- **Level 3:** Store (10 Stores)
- **Level 4:** Category (FOODS, HOBBIES, HOUSEHOLD)
- **Level 5:** Department (7 Departments)
"""))

    cells.append(nbf.v4.new_code_cell("""eda = M5ExploratoryAnalysis(df_cal, df_prices, df_sales)
aggs = eda.compute_hierarchical_aggregations()

print("Hierarchical Aggregations ready.")
display(aggs["total"].head())
"""))

    # Step 3: Croston Classification
    cells.append(nbf.v4.new_markdown_cell("""### 3. Syntetos-Boylan & Croston Intermittent Demand Matrix
We classify all 30,490 time series using:
- **Average Demand Interval (ADI):** Cut-off $\\alpha = 1.32$
- **Squared Coefficient of Variation ($CV^2$):** Cut-off $\\beta = 0.49$

#### Demand Regimes:
1. **Smooth ($ADI < 1.32, CV^2 < 0.49$):** Regular demand, low variance (classical ARIMA works).
2. **Intermittent ($ADI \\ge 1.32, CV^2 < 0.49$):** Sporadic arrivals, stable transaction sizes (suited to Croston / Kalman filter).
3. **Erratic ($ADI < 1.32, CV^2 \\ge 0.49$):** Frequent arrivals, high size variance.
4. **Lumpy ($ADI \\ge 1.32, CV^2 \\ge 0.49$):** Sporadic arrivals AND volatile order quantities (requires state-space / DeepAR).
"""))

    cells.append(nbf.v4.new_code_cell("""demand_df = eda.classify_demand_patterns(sample_size=None)

croston_summary = demand_df["quadrant"].value_counts(normalize=True).mul(100).round(2).to_frame(name="Percentage (%)")
croston_summary["SKU_Count"] = demand_df["quadrant"].value_counts()
display(croston_summary)
"""))

    # Step 4: Seasonality, SNAP & Holidays
    cells.append(nbf.v4.new_markdown_cell("""### 4. Seasonality, Policy Shocks (SNAP), and Store Closures
- **Weekly Seasonality:** Weekend surge (Saturday: 1.21, Sunday: 1.20) vs mid-week dip (Tuesday/Wednesday: ~0.88).
- **SNAP Program Impact:** Demand uplift during food voucher distribution days across states.
- **Christmas Closure:** Chain-wide closure on December 25 where sales drop to near zero.
"""))

    cells.append(nbf.v4.new_code_cell("""season_data = eda.analyze_seasonality_and_events(aggs)

print("SNAP Policy Demand Uplifts:")
for state, metrics in season_data["snap_analysis"].items():
    print(f"  • State {state}: +{metrics['uplift_pct']}% uplift on SNAP days")

print("\\nTop Event Uplifts:")
for ev, up in list(season_data["top_event_uplifts"].items())[:5]:
    print(f"  • {ev:<15}: +{up}%")
"""))

    # Step 5: Stationarity & Autocorrelation
    cells.append(nbf.v4.new_markdown_cell("""### 5. Stationarity Testing & Autocorrelation (ACF) Harmonics
- **Augmented Dickey-Fuller (ADF) Test:** Raw sales series is non-stationary ($p = 0.501$).
- **First Differencing ($\\Delta y_t$):** Achieves strong stationarity ($p < 10^{-5}$), confirming appropriateness of state-space models.
- **ACF Peaks:** Lags 7, 14, 21, and 28 exhibit significant correlation, proving weekly harmonic seasonality.
"""))

    cells.append(nbf.v4.new_code_cell("""stat_data = eda.compute_stationarity_and_autocorrelation(aggs)
raw_adf = stat_data["total_level"]["raw_series"]
diff_adf = stat_data["total_level"]["differenced_series"]

print(f"Raw Series ADF Stat:         {raw_adf['adf_statistic']} (p-value: {raw_adf['p_value']:.4f}, Stationary: {raw_adf['is_stationary']})")
print(f"Differenced Series ADF Stat: {diff_adf['adf_statistic']} (p-value: {diff_adf['p_value']:.4e}, Stationary: {diff_adf['is_stationary']})")
print("\\nWeekly ACF Harmonics:")
for lag, val in stat_data["total_level"]["autocorrelations"].items():
    print(f"  • {lag:<18}: {val}")
"""))

    # Step 6: Kalman Filter Demonstration
    cells.append(nbf.v4.new_markdown_cell("""### 6. Kalman Filter State-Space Tracking
Daily observed sales $z_t$ are noisy manifestations of an underlying latent demand state $x_t$:
$$z_t = H x_t + v_t, \\quad v_t \\sim \\mathcal{N}(0, R)$$
The Kalman filter recursive update filters out intermittent zero spikes, tracks latent demand velocity, and computes error covariance $P_t$ for dynamic safety stock calculation.
"""))

    cells.append(nbf.v4.new_code_cell("""kf_demo = eda.demonstrate_kalman_filter()

print(f"Demonstration SKU:    {kf_demo['sku_id']}")
print(f"Zero-Sales Rate:      {kf_demo['zero_percentage']}%")
print(f"Observed Daily Mean:  {kf_demo['mean_observed']}")
print(f"Filtered State Level: {kf_demo['mean_filtered']}")
"""))

    # Step 7: Visual Gallery
    cells.append(nbf.v4.new_markdown_cell("""### 7. Publication-Quality Visual Gallery (300 DPI)
Pre-rendered figures illustrating empirical findings:
"""))

    cells.append(nbf.v4.new_code_cell("""figures = [
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

for fig in figures:
    p = Path(fig)
    if p.exists():
        print(f"Displaying: {p.name}")
        display(Image(filename=str(p), width=800))
"""))

    nb["cells"] = cells
    out_path = Path("notebooks/02_exploratory_data_analysis.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        nbf.write(nb, f)
    print(f"[NOTEBOOK CREATED] {out_path.resolve()}")


if __name__ == "__main__":
    generate_cleaning_notebook()
    generate_eda_notebook()
