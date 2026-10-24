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
└── processed/                                # Cleaned CSV files & feature-engineered artifacts
    ├── cleaned_calendar.csv                  # Cyclical encodings (sin/cos), SNAP & holiday markers
    ├── cleaned_sell_prices.csv               # Regular price baselines, discount depth percentages
    ├── cleaned_sales_train_validation.csv    # Cleaned unit sales matrix (days 1 to 1913)
    ├── cleaned_sales_train_evaluation.csv    # Cleaned unit sales matrix (days 1 to 1941)
    ├── cleaned_sample_submission.csv         # Validated submission template
    ├── train_features.csv                    # Chronological training split (leak-free features)
    ├── val_features.csv                      # Chronological 28-day validation horizon
    ├── selected_features_full.csv            # Complete subset feature panel (all 1913 days)
    └── feature_metadata.json                 # Audit schemas, split dates, and missing-value logs
```

---

## 📓 Decoupled Jupyter Notebooks

As per requirements, data cleaning and exploratory data analysis are split into two dedicated, interactive notebooks in [`notebooks/`](notebooks/):

1. **[`01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb)**:
   * Loading raw CSVs from `data/raw/`.
   * Schema audit, zero-null verification, and memory downcasting.
   * Product lifecycle analysis: identifying pre-launch days vs. active demand.
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
├── requirements.txt                    # Project dependencies (including pytest)
├── .gitignore                          # Excludes raw & processed big data (>100MB)
│
├── data/
│   ├── raw/                            # Raw M5 dataset (calendar, prices, sales)
│   └── processed/                      # Cleaned CSV tables, feature sets & split metadata
│
├── src/
│   ├── __init__.py                     # Package init
│   ├── data_loader.py                  # Automated streaming acquisition & downcasting
│   ├── data_quality.py                 # Automated data quality audit engine
│   ├── data_cleaning.py                # 1-to-1 cleaning pipeline
│   ├── eda_analysis.py                 # Croston classification, SNAP uplift, ADF, Kalman filter
│   ├── visualization.py                # 300 DPI publication plotting engine
│   ├── run_eda_pipeline.py             # End-to-end EDA execution script
│   ├── feature_engineering.py          # Modular, leak-free feature engineering engine
│   ├── run_feature_pipeline.py         # Configurable CLI feature pipeline runner
│   ├── baseline_models.py              # Seasonal-Naive & GBDT recursive forecasters
│   ├── baseline_visualization.py       # 300 DPI baseline diagnostic visualizer
│   ├── run_baseline_pipeline.py        # Baseline evaluation & report orchestration
│   ├── kalman_filter_model.py          # Structural State-Space Kalman Filter per SKU
│   ├── kalman_visualization.py         # 300 DPI Kalman tracking & 3-way benchmark visualizer
│   ├── run_kalman_pipeline.py          # Kalman pipeline runner & report generator
│   ├── deepar_model.py                 # Global DeepAR Probabilistic Autoregressive Forecaster
│   ├── deepar_visualization.py         # 300 DPI DeepAR tracking & 4-way benchmark visualizer
│   └── run_deepar_pipeline.py          # DeepAR pipeline orchestrator & report generator
│
├── tests/
│   ├── test_feature_engineering.py     # Feature engineering pytest suite
│   ├── test_baseline_models.py         # Baseline forecasting & zero-leakage pytest suite
│   ├── test_kalman_filter.py           # Kalman filter unit & integration test suite (7 tests)
│   └── test_deepar_model.py            # DeepAR unit & integration test suite (9 tests)
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
    ├── baseline_findings.md            # Comprehensive AML baseline benchmark report
    ├── baseline_metrics.json           # Machine-readable baseline metrics (MAE, RMSE, WAPE)
    ├── kalman_findings.md              # Comprehensive Kalman Filter benchmark report
    ├── kalman_metrics.json             # Machine-readable Kalman metrics & CI coverage
    ├── deepar_findings.md              # Comprehensive Global DeepAR benchmark report
    ├── deepar_metrics.json             # Machine-readable DeepAR metrics, pinball & CI coverage
    └── figures/                        # High-resolution publication plots (300 DPI)
        ├── 01_hierarchical_sales_overview.png
        ├── 02_intermittent_croston_matrix.png
        ├── 03_zero_demand_distribution.png
        ├── 04_seasonality_and_calendar_effects.png
        ├── 05_holiday_and_snap_impact.png
        ├── 06_price_elasticity_analysis.png
        ├── 07_store_level_comparison.png
        ├── 08_acf_pacf_stationarity.png
        ├── 09_kalman_filter_motivation.png
        ├── 10_baseline_aggregate_comparison.png
        ├── 11_baseline_series_predictions.png
        ├── 12_baseline_metric_comparison.png
        ├── 13_kalman_aggregate_tracking.png
        ├── 14_kalman_sku_trajectories.png
        ├── 15_three_way_model_comparison.png
        ├── 16_four_way_model_comparison.png
        ├── deepar_forecasts.png
        └── deepar_prediction_intervals.png
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
| **10. Baseline Aggregate Forecast** | 28-day daily system demand vs SNaive vs GBDT | GBDT tracks aggregate trajectories with superior fidelity |
| **11. SKU Baseline Trajectories** | Out-of-sample forecasts for representative items | Captures category-specific intermittency without lookahead |
| **12. Model Metric Comparison** | MAE, RMSE, and WAPE comparison across categories | GBDT achieves **+14.64%** lower WAPE over Seasonal-Naive |
| **13. Kalman Aggregate Tracking** | 28-day system demand tracking vs Kalman + 95% band | 90.07% empirical coverage across portfolio aggregate |
| **14. Kalman SKU Trajectories** | Multi-panel SKU forecasts with 95% & 80% confidence bands | Adapts to local trend & weekly harmonic cycles per SKU |
| **15. Three-Way Model Benchmark** | SNaive vs GBDT vs Kalman Filter per SKU | Kalman Filter achieves lowest MAE (0.9333) & WAPE (80.06%) |
| **16. Four-Way Model Benchmark** | SNaive vs GBDT vs Kalman Filter vs Global DeepAR | **Global DeepAR achieves lowest RMSE (2.2221) & 91.79% 80% CI coverage** |
| **DeepAR Forecasts** | Daily system demand vs DeepAR P50 + 80% CI envelope | Captures aggregate momentum with strict causal rollout |
| **DeepAR Prediction Intervals** | SKU trajectories with [P10, P90] predictive intervals | Non-negative quantile bounds across categories |

---

## 🚀 Execution Guide

### 1. Run Data Ingestion & Streaming (if raw files missing)
```bash
python3 src/data_loader.py
```
*Downloads the M5 raw datasets directly from the Hugging Face mirror into `data/raw/` with verified checksums.*

### 2. Run Data Cleaning Pipeline
```bash
python3 src/data_cleaning.py
```
*Loads raw data, performs schema checks, and exports 1-to-1 clean CSV tables to `data/processed/`.*

### 3. Run Full EDA & Visualization Pipeline
```bash
python3 src/run_eda_pipeline.py
```
*Computes Croston classification, SNAP effects, stationarity tests, and generates all 9 publication-grade figures in `reports/figures/`.*

### 4. Run Feature Engineering & Train/Val Splitting Pipeline
```bash
# Default: 50 diverse series across all 3 categories & 10 stores, 28-day validation horizon
python3 src/run_feature_pipeline.py

# Custom: 100 series from store CA_1 with specific lags & rolling windows
python3 src/run_feature_pipeline.py --n_series 100 --store_id CA_1 --val_days 28 --lags 1 7 28 --windows 7 28

# Filter by category and department with top-volume selection
python3 src/run_feature_pipeline.py --n_series 50 --cat_id FOODS --dept_id FOODS_1 --selection_strategy top_volume
```
*Transforms wide sales into tidy long format, verifies daily date index continuity, calculates causal lag (1, 7, 28) and leak-free rolling features (mean, std for 7 and 28 days), constructs calendar, event, and SNAP policy indicators, and saves chronological train/val splits to `data/processed/`.*

### 5. Run AML Baseline Forecasting Pipeline
```bash
python3 src/run_baseline_pipeline.py
```
*Trains Seasonal-Naive (weekly persistence) and GBDT (LightGBM) under a strictly leak-free recursive multi-step forecasting protocol. Evaluates MAE, RMSE, and zero-denominator-safe WAPE across identical series and dates, renders 3 publication-grade figures, and exports predictions and analytical reports.*

### 6. Run Structural Kalman Filter per SKU Pipeline
```bash
python3 src/run_kalman_pipeline.py
```
*Fits an independent Basic Structural Time Series Kalman Filter per SKU (Local Level + 7-Day Weekly Seasonality), forecasts the 28-day out-of-sample validation horizon with exact analytical uncertainty bands, computes MAE, RMSE, WAPE, and prediction interval coverage (90.07% at 95% CI), and generates publication-grade figures 13, 14, 15.*

### 7. Run Global DeepAR Probabilistic Pipeline
```bash
# Default: Full training across all 50 SKUs with 200 Monte Carlo paths
python3 src/run_deepar_pipeline.py

# Configurable experiment: Custom epochs, learning rate, and samples
python3 src/run_deepar_pipeline.py --epochs 15 --lr 0.001 --hidden_dim 64 --n_samples 200 --device auto

# Fast test run on subset of series
python3 src/run_deepar_pipeline.py --n_series 10 --epochs 5 --n_samples 50
```
*Trains a unified Global Autoregressive LSTM network across multiple series with static entity embeddings and dynamic calendar/event covariates. Generates 28-day out-of-sample probabilistic forecasts via strictly causal Monte Carlo rollout without ground-truth validation sales, computes MAE, RMSE, WAPE, Pinball losses (P10, P50, P90), and 80% prediction interval coverage (91.79%), and renders publication-grade figures.*

### 8. Run Unit & Integration Tests
```bash
pytest tests/ -v
```
*Runs the full 30-test suite across 4 test modules:*
- `test_feature_engineering.py` (9 tests): Lags, leak-free rolling, continuous daily date indexing, calendar features, split integrity.
- `test_baseline_models.py` (5 tests): Seasonal-Naive persistence, GBDT recursive rollout, zero-leakage proof, safe WAPE.
- `test_kalman_filter.py` (7 tests): 28-day horizon, non-negative clipping, missing observation predict-only handling, zero-demand preservation, and interval calibration.
- `test_deepar_model.py` (9 tests): Horizon length, tensor shapes, non-negative clipping, quantile monotonicity (P10 <= P50 <= P90), zero validation leakage, pinball loss precision, interval metrics, short-series padding, and integration.

### 9. Open the Separate Interactive Notebooks
```bash
# Data Cleaning & Ingestion
jupyter notebook notebooks/01_data_cleaning.ipynb

# Exploratory Data Analysis & Modeling Insights
jupyter notebook notebooks/02_exploratory_data_analysis.ipynb
```

---

## 🛠️ Feature Engineering & Causal Formulation

| Category | Features | Construction & Leakage Prevention |
| :--- | :--- | :--- |
| **Identifiers** | `id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` | Preserved unchanged throughout pipeline for hierarchical mapping. |
| **Target** | `demand` | Pristine daily unit sales. Original observations are never silently mutated or imputed. |
| **Causal Lags** | `lag_1`, `lag_7`, `lag_28` | Grouped by `id`: $\text{demand}_{t-k}$ strictly using past historical demand. |
| **Rolling Stats** | `rolling_mean_7`, `rolling_std_7`, `rolling_mean_28`, `rolling_std_28` | **Shift(1) prior to rolling**: Grouped by `id`, demand is shifted by 1 day so day $t$ features only use $\{t-1, \dots, t-W\}$. Mathematically verified to guarantee zero target leakage. |
| **Calendar Components**| `day_of_week`, `month`, `year`, `day`, `is_weekend`, `quarter`, `week_of_year` | Extracted from daily date timestamps (0 = Monday, 6 = Sunday; weekend flag for Sat/Sun). |
| **Cyclical Encodings** | `day_sin`, `day_cos`, `month_sin`, `month_cos` | Continuous periodic representation $\sin(2\pi t / T)$ and $\cos(2\pi t / T)$. |
| **Events & Holidays** | `event_name_1`, `event_type_1`, `has_event_1`, `has_event_2`, `is_event_day`, `is_christmas` | Binary event flags and specialized Walmart Christmas closure marker. |
| **Policy Indicators** | `snap_active`, `snap_CA`, `snap_TX`, `snap_WI` | `snap_active` dynamically resolves each series' store state to its state-specific SNAP schedule. |

### Chronological Train / Validation Split:
- **Validation Horizon**: Exactly 28 days per series (`2016-03-28` to `2016-04-24`).
- **Training Period**: `2011-02-26` to `2016-03-27` (strictly prior to validation start).
- **Leak-Free Warmup**: Initial 28 days with uncomputable lags/rolling statistics are trimmed for model-ready training without altering original demand observations.
- **Strict Time Splitting**: Time series observations are never randomly shuffled or randomly split.

---

## 📈 Four-Way Forecasting Benchmark: Baselines vs Kalman Filter vs Global DeepAR

All models are evaluated on **identical product-store series (50 SKUs), identical dates, and an identical 28-day out-of-sample forecast horizon (2016-03-28 to 2016-04-24)**:

| Model Architecture | MAE (Units) | RMSE (Units) | WAPE (%) | Mean Pinball Loss | 80% CI Coverage | Uncertainty Protocol |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Seasonal-Naive (7-day Persistence)** | 1.1143 | 2.6030 | 95.59% | - | - | Deterministic weekly persistence |
| **GBDT (Recursive LightGBM)** | 0.9512 | 2.2679 | 81.60% | - | - | Deterministic recursive rollout |
| **Kalman Filter per SKU (Local Level + Seasonality)** | **0.9333** | 2.3303 | **80.06%** | - | 80.93% | Exact closed-form covariance $\mathbf{P}_t$ |
| **Global DeepAR (Autoregressive LSTM)** | 1.0416 | **2.2221** | 89.35% | **0.3194** | **91.79%** | **Monte Carlo Quantile Rollout** |

### Benchmark Key Takeaways:
- **Lowest RMSE: Global DeepAR (2.2221)**: Achieves the best RMSE across the entire portfolio, surpassing GBDT (2.2679), Kalman Filter (2.3303), and Seasonal-Naive (2.6030) due to its shared recurrent representation penalizing extreme deviations while generalizing across sparse series.
- **Lowest MAE & WAPE: Kalman Filter per SKU (0.9333 MAE, 80.06% WAPE)**: Structural local level and weekly harmonic state updates adapt rapidly to individual SKU shifts without suffering from recursive variance dampening.
- **Calibrated Prediction Intervals**:
  - Kalman Filter (Analytical Covariance): **80.93% coverage** at 80% CI (width: 2.30 units); **90.07% coverage** at 95% CI (width: 3.14 units).
  - Global DeepAR (Monte Carlo Quantile Rollout): **91.79% coverage** at 80% CI (width: 3.20 units); Mean Pinball Loss: **0.3194** ($P_{10}=0.1116, P_{50}=0.5208, P_{90}=0.3258$).

---

## 🎯 Phase 1 Evaluation Alignment

| Evaluation Criteria | Implementation Deliverable | Location in Repository |
| :--- | :--- | :--- |
| **Dataset Quality & Ingestion** | Full data audit, memory downcasting, schema validation | `src/data_loader.py`, `src/data_cleaning.py`, `reports/data_quality_report.md` |
| **Exploratory Data Analysis** | Syntetos-Boylan matrix, SNAP uplift, seasonality | `src/eda_analysis.py`, `reports/eda_findings.md`, `reports/eda_metrics.json` |
| **Feature Engineering** | Causal lags, shift(1) rolling, cyclical, SNAP, 28d split | `src/feature_engineering.py`, `src/run_feature_pipeline.py` |
| **AML Baseline Forecasting** | Seasonal-Naive, GBDT recursive rollout, zero leakage | `src/baseline_models.py`, `src/run_baseline_pipeline.py`, `reports/baseline_findings.md` |
| **Kalman Filter Forecasting** | Local Level + Weekly Seasonality per SKU, analytic intervals | `src/kalman_filter_model.py`, `src/run_kalman_pipeline.py`, `reports/kalman_findings.md` |
| **Global DeepAR Forecasting** | Autoregressive LSTM, scale normalization, Monte Carlo rollout | `src/deepar_model.py`, `src/run_deepar_pipeline.py`, `reports/deepar_findings.md` |
| **Visualizations** | 18 Publication-grade 300 DPI figures | `reports/figures/`, `src/visualization.py`, `src/kalman_visualization.py`, `src/deepar_visualization.py` |
| **Theoretical Rigor & Methodology**| Mathematical formulation for Kalman Filters & DeepAR | `reports/eda_findings.md`, `reports/kalman_findings.md`, `reports/deepar_findings.md` |
| **Testing & Reproducibility** | Full 30-test pytest suite, zero target leakage proofs | `tests/test_feature_engineering.py`, `tests/test_baseline_models.py`, `tests/test_kalman_filter.py`, `tests/test_deepar_model.py` |

---
*Created for CSAI Student Project 5 - Phase 1 Submission.*

