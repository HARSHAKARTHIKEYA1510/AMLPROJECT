# Project 5: Inventory Demand Forecasting via Kalman Filters and DeepAR Global Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Phase](https://img.shields.io/badge/Phase-1%20Machine%20Learning-emerald.svg)]()
[![Evaluation](https://img.shields.io/badge/Evaluation-Dataset%20Quality%20%26%20EDA-blueviolet.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Walmart%20M5%20Forecasting-orange.svg)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **CSAI Student Project 5**  
> **Title:** Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models  
> **Course:** Advanced Machine Learning (AML)  
> **Phase 1 Focus:** Data Ingestion, Data Cleaning & Preprocessing, Comprehensive Data Quality Audit, Exploratory Data Analysis (EDA), and Visualizations.

---

## 📌 Project Overview & Abstract

Forecast demand for retail time series while comparing local state tracking with global probabilistic neural forecasts. The project emphasizes:
1. **Fair handling of stockouts:** Marking rather than imputing stockout periods.
2. **Cold start items:** Flagging products with limited transaction histories (< 1 year).
3. **Forecast uncertainty:** Quantifying predictive intervals via state covariance (Kalman filter) and quantile pinball loss.
4. **The 'Hybrid' Logic:** Bridging local state estimation (Kalman filters per SKU adapting quickly to local level/trend) with global sequence learning (DeepAR cross-series representation sharing across sparse product histories).

---

## 🗂️ Data Architecture: Raw vs. Processed Separation

The project enforces clean separation between raw immutable data and cleaned, feature-engineered artifacts:

```text
data/
├── raw/                                      # 5 Raw files from Kaggle / M5 source
│   ├── calendar.csv                          # Daily calendar attributes & events (1,969 days)
│   ├── sell_prices.csv                       # Weekly store-item sell prices (6,841,121 rows)
│   ├── sales_train_validation.csv            # Daily sales matrix (days 1 to 1913)
│   ├── sales_train_evaluation.csv            # Daily sales matrix (extended to day 1941)
│   └── sample_submission.csv                 # Official forecasting submission template
│
└── processed/                                # Exactly 5 Cleaned CSV files (1-to-1 counterpart)
    ├── cleaned_calendar.csv                  # Cyclical encodings (sin/cos), SNAP & holiday markers
    ├── cleaned_sell_prices.csv               # Regular price baselines, discount depth percentages
    ├── cleaned_sales_train_validation.csv    # Cleaned unit sales matrix (days 1 to 1913)
    ├── cleaned_sales_train_evaluation.csv    # Cleaned unit sales matrix (days 1 to 1941)
    └── cleaned_sample_submission.csv         # Validated submission template
```

---

## 📓 Decoupled Jupyter Notebooks

As per requirements, data cleaning and exploratory data analysis are split into two dedicated, interactive notebooks in [`notebooks/`](notebooks/):

1. **[`01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb)**:
   * Loading raw CSVs from `data/raw/`.
   * Schema audit, zero-null verification, and memory downcasting.
   * Product lifecycle analysis: identifying pre-launch days vs. active demand.
   * **Marking stockout periods:** Creating `is_unlisted_stockout` flags rather than imputing zeros.
   * Causal feature engineering: lags ($t-1, 7, 14, 28$) and rolling windows ($7, 28$) without lookahead.
   * Exporting structured tables into `data/processed/`.

2. **[`02_exploratory_data_analysis.ipynb`](notebooks/02_exploratory_data_analysis.ipynb)**:
   * Ingestion of cleaned tables from `data/processed/`.
   * Multi-tier hierarchical sales dynamics (System $\rightarrow$ State $\rightarrow$ Store $\rightarrow$ Category $\rightarrow$ Department).
   * **Syntetos-Boylan / Croston Demand Categorization:** Computing $ADI$ and $CV^2$ across all 30,490 SKUs.
   * Intermittent demand characterization (active vs. raw zero rates).
   * Day-of-week surges (Saturday/Sunday indices) and monthly seasonality.
   * **SNAP Policy Demand Uplift:** +21.3% (WI), +11.0% (TX), and +7.6% (CA).
   * Augmented Dickey-Fuller (ADF) stationarity testing and ACF weekly harmonic decay.
   * **Kalman Filter Latent State Tracking:** Demonstration of filtering observation noise and estimating safety stock bounds.

---

## 📂 Repository Tree

```text
AMLPROJECT/
├── README.md                           # Comprehensive documentation & project guide
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Excludes raw & processed big data (>100MB)
│
├── data/
│   ├── raw/                            # Raw M5 dataset (calendar, prices, sales)
│   └── processed/                      # Cleaned CSV tables & starter panel
│
├── src/
│   ├── __init__.py                     # Package init
│   ├── data_loader.py                  # Automated streaming acquisition & downcasting
│   ├── data_quality.py                 # Automated data quality audit engine
│   ├── data_cleaning.py                # Cleaning pipeline, stockout & cold-start markers
│   ├── eda_analysis.py                 # Croston classification, SNAP uplift, ADF, Kalman filter
│   ├── visualization.py                # 300 DPI publication plotting engine
│   └── run_eda_pipeline.py             # End-to-end execution script
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb          # Dedicated Data Cleaning notebook
│   └── 02_exploratory_data_analysis.ipynb # Dedicated EDA & Visualizations notebook
│
└── reports/
    ├── data_quality_report.md          # Formal data quality scorecard
    ├── data_quality_report.json        # Machine-readable audit metrics
    ├── eda_findings.md                 # Academic EDA findings & theoretical justification
    ├── eda_metrics.json                # Numerical EDA metrics
    └── figures/                        # 9 High-resolution publication plots (300 DPI)
        ├── 01_hierarchical_sales_overview.png
        ├── 02_intermittent_croston_matrix.png
        ├── 03_zero_demand_distribution.png
        ├── 04_seasonality_and_calendar_effects.png
        ├── 05_holiday_and_snap_impact.png
        ├── 06_price_elasticity_analysis.png
        ├── 07_store_level_comparison.png
        ├── 08_acf_pacf_stationarity.png
        └── 09_kalman_filter_motivation.png
```

---

## 📊 Publication-Grade Visual Artifacts

Rendered at 300 DPI in `reports/figures/`:

| Figure | Description | Key Insight |
| :--- | :--- | :--- |
| **01. Hierarchical Overview** | System, State & Category sales (2011–2016) | Long-term growth & Christmas store closure |
| **02. Croston Demand Matrix** | $ADI$ vs $CV^2$ Scatter & Quadrant Donut | **90.84%** of SKUs are Intermittent or Lumpy |
| **03. Zero-Demand Density** | Raw observed zeros vs. Active shelf zeros | 12.38M zeros represent pre-release shelf days |
| **04. Seasonality Indices** | Day-of-Week & Monthly Normalized Indices | Weekend sales surge by **+20.7%** over weekday baseline |
| **05. Policy & Event Impact** | SNAP uplift by state & Holiday spikes | SNAP boosts Wisconsin store food demand by **+21.31%** |
| **06. Price Elasticity** | Density distribution by category & dept | Clear dispersion between Foods vs Hobbies/Household |
| **07. Store Comparison** | Daily sales volume & variance across 10 stores | CA_3 and CA_1 lead in total unit turnover |
| **08. Stationarity & ACF** | Autocorrelation function at key lag multiples | Multi-lag weekly harmonics at Lags 7, 14, 21, and 28 |
| **09. Kalman Filter Tracking** | Latent state estimation vs. noisy sales observations | Filters zero noise and extracts true consumption velocity |

---

## 🚀 Execution Guide

### 1. Run Data Cleaning & Feature Generation Pipeline
```bash
python3 src/data_cleaning.py
```
*Loads raw data, performs schema checks, creates stockout/cold-start flags, and exports clean CSV tables to `data/processed/` in ~7 seconds.*

### 2. Run Full EDA & Visualization Pipeline
```bash
python3 src/run_eda_pipeline.py
```
*Computes Croston classification, SNAP effects, stationarity tests, and generates all 9 figures in ~7 seconds.*

### 3. Open the Separate Interactive Notebooks
```bash
# Data Cleaning & Ingestion
jupyter notebook notebooks/01_data_cleaning.ipynb

# Exploratory Data Analysis & Modeling Insights
jupyter notebook notebooks/02_exploratory_data_analysis.ipynb
```

---

## 🎯 Phase 1 Evaluation Alignment

| Evaluation Criteria | Implementation Deliverable | Location in Repository |
| :--- | :--- | :--- |
| **Dataset Quality & Ingestion** | Full data audit, memory downcasting, schema validation | `src/data_loader.py`, `src/data_cleaning.py`, `reports/data_quality_report.md` |
| **Exploratory Data Analysis** | Syntetos-Boylan matrix, SNAP uplift, seasonality | `src/eda_analysis.py`, `reports/eda_findings.md`, `reports/eda_metrics.json` |
| **Visualizations** | 9 Publication-grade 300 DPI figures | `reports/figures/`, `src/visualization.py` |
| **Theoretical Rigor & Methodology**| Mathematical justification for Kalman Filters & DeepAR | `reports/eda_findings.md`, Section 5 |
| **Code Quality & Reproducibility**| Modular package, type hints, PEP 8, git documentation | `src/`, `notebooks/`, and `data/` |

---
*Created for CSAI Student Project 5 - Phase 1 Submission.*
