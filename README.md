# Project 5: Inventory Demand Forecasting via Kalman Filters and DeepAR Global Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Phase](https://img.shields.io/badge/Phase-1%20Machine%20Learning-emerald.svg)]()
[![Evaluation](https://img.shields.io/badge/Evaluation-Dataset%20Quality%20%26%20EDA-blueviolet.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Walmart%20M5%20Forecasting-orange.svg)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **CSAI Student Project 5**  
> **Course:** Advanced Machine Learning (AML)  
> **Focus for Current Deliverable (Phase 1):** Dataset Collection, Cleaning, Comprehensive Data Quality Audit, Exploratory Data Analysis (EDA), and Visualizations.

---

## 📌 Executive Summary

Accurate retail demand forecasting is vital for multi-echelon inventory management, supply chain resilience, and minimizing stockout risk while optimizing working capital. This repository hosts the end-to-end implementation for **Project 5**, comparing and synthesizing **State-Space Kalman Filters** with **DeepAR Global Models**.

In **Phase 1**, we establish the complete data foundation on the official **Walmart M5 Forecasting Hierarchical Dataset** (30,490 SKUs across 10 stores in California, Texas, and Wisconsin over 1,913 days, totaling **58,327,370 unit-day observations**).

### Core Phase 1 Empirical Milestones:
1. **Automated Data Acquisition & Memory Optimization:** Designed an automated streaming downloader with dynamic downcasting (`int64` $\rightarrow$ `int16`/`int8`, `float64` $\rightarrow$ `float32`, categoricals), slashing memory consumption by **~75%** (from 2.5 GB to ~450 MB).
2. **Data Quality & Integrity Audit:** Validated 100% schema continuity, 0 missing matrix values, 0 negative values, and characterized **12.38 million pre-release leading zeros** (distinguishing inactive shelf periods from genuine zero-demand days).
3. **Croston / Syntetos-Boylan Classification:** Categorized all 30,490 SKUs across the 4 classic demand quadrants ($ADI$ vs $CV^2$):
   - **Intermittent:** 22,121 SKUs (**72.55%**)
   - **Lumpy:** 5,577 SKUs (**18.29%**)
   - **Smooth:** 1,912 SKUs (**6.27%**)
   - **Erratic:** 880 SKUs (**2.89%**)
   - *Key finding:* Over **90.8% of items are intermittent or lumpy**, providing concrete empirical justification for state-space Kalman filtering and non-Gaussian probabilistic DeepAR models.
4. **Exogenous Shocks & Policy Analytics:**
   - **SNAP Program Impact:** Measured massive demand surges on government food assistance distribution days: **+21.31%** in Wisconsin, **+11.05%** in Texas, and **+7.56%** in California.
   - **Weekly Cycles:** Saturday (Index **1.21**) and Sunday (Index **1.20**) drive peak volumes, while mid-week days drop to ~0.88.
   - **Christmas Closure Anomaly:** Identified deterministic chain-wide closure on Dec 25 (sales drop to 11 units chain-wide).
5. **Stationarity & Kalman Demonstration:** Augmented Dickey-Fuller tests proved raw demand non-stationarity ($p = 0.501$) and strong post-differencing stationarity ($p < 10^{-5}$). Implemented a Kalman Filter demonstrating continuous latent demand state tracking from noisy, zero-inflated daily sales.

---

## 📂 Repository Architecture

```text
AMLPROJECT/
├── README.md                      # Comprehensive project documentation & guide
├── requirements.txt               # Locked project dependencies
├── .gitignore                     # Git tracking exclusions (ignoring large raw CSVs)
│
├── data/
│   ├── raw/                       # Downloaded M5 CSVs (calendar, prices, sales)
│   └── processed/                 # Memory-optimized subsets and parquet files
│
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # Automated download, memory downcasting, loader
│   ├── data_quality.py            # Rigorous data quality scorecard & audit engine
│   ├── eda_analysis.py            # Hierarchical aggregations, Croston classification, ADF
│   ├── visualization.py           # Publication-grade plotting suite (300 DPI)
│   ├── run_eda_pipeline.py        # End-to-end pipeline orchestration script
│   └── generate_notebook.py       # Programmatic generator for Jupyter notebook
│
├── notebooks/
│   └── 01_data_cleaning_and_eda.ipynb  # Interactive walkthrough notebook
│
└── reports/
    ├── data_quality_report.md     # Full statistical data quality audit scorecard
    ├── data_quality_report.json   # Machine-readable audit metrics
    ├── eda_findings.md            # In-depth academic EDA & inventory insights
    ├── eda_metrics.json           # Machine-readable EDA summary metrics
    ├── viva_preparation_guide.md  # 10 comprehensive viva voce defense Q&As
    └── figures/                   # 9 High-resolution publication-quality plots (300 DPI)
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

All figures are rendered at 300 DPI in `reports/figures/`:

| Figure | Description | Key Insight |
| :--- | :--- | :--- |
| **01. Hierarchical Overview** | System, State & Category sales (2011–2016) | Demonstrates multi-year growth and Christmas store closure |
| **02. Croston Demand Matrix** | ADI vs $CV^2$ Scatter & Quadrant Donut | **90.84%** of SKUs are Intermittent or Lumpy |
| **03. Zero-Demand Density** | Raw observed zeros vs. Active shelf zeros | Pre-release shelf inactivity represents 12.38M zeros |
| **04. Seasonality Indices** | Day-of-Week & Monthly Normalized Indices | Weekend sales surge by **+20.7%** over weekday baseline |
| **05. Policy & Event Impact** | SNAP uplift by state & Holiday spikes | SNAP boosts Wisconsin store food demand by **+21.31%** |
| **06. Price Elasticity** | Density distribution by category & dept | Clear dispersion between Foods vs Hobbies/Household |
| **07. Store Comparison** | Daily sales volume & variance across 10 stores | CA_3 and CA_1 lead in total unit turnover |
| **08. Stationarity & ACF** | Autocorrelation function at key lag multiples | Multi-lag weekly harmonics at Lags 7, 14, 21, and 28 |
| **09. Kalman Filter Tracking** | Latent state estimation vs. noisy sales observations | Filters zero noise and extracts true consumption velocity |

---

## 🚀 Quickstart & Pipeline Execution

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/HARSHAKARTHIKEYA1510/AMLPROJECT.git
cd AMLPROJECT
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
To trigger automated data downloading, quality audit, Croston categorization, statistical testing, and visualization generation:
```bash
python3 src/run_eda_pipeline.py
```
*Execution takes ~7 to 10 seconds due to vectorized matrix operations!*

### 3. Launch the Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/01_data_cleaning_and_eda.ipynb
```

---

## 🎯 Phase 1 Evaluation Alignment

| Evaluation Criteria | Implementation Deliverable | Location in Repository |
| :--- | :--- | :--- |
| **Dataset Quality & Ingestion** | Full data audit, memory downcasting, schema validation | `src/data_loader.py`, `src/data_quality.py`, `reports/data_quality_report.md` |
| **Exploratory Data Analysis** | Syntetos-Boylan matrix, SNAP uplift, seasonality | `src/eda_analysis.py`, `reports/eda_findings.md`, `reports/eda_metrics.json` |
| **Visualizations** | 9 Publication-grade 300 DPI figures | `reports/figures/`, `src/visualization.py` |
| **Theoretical Rigor** | Mathematical justification for Kalman Filters & DeepAR | `reports/eda_findings.md`, Section 5 |
| **Viva Voce Preparation** | 10 Detailed Technical Defense Q&As | `reports/viva_preparation_guide.md` |
| **Code Quality & Reproducibility**| Modular package, type hints, PEP 8, git documentation | Entire `src/` and `notebooks/` directory |

---

## 🔮 Roadmap to Phase 2 (Deep Learning & Hybrid Approach)

- **Phase 1 (Current):** Classical Machine Learning baselines, Local Linear Trend Kalman Filter state estimation, LightGBM hierarchical benchmarks.
- **Phase 2 (Upcoming):** DeepAR Global Autoregressive RNN with Negative Binomial likelihood, ablation studies, and the **Hybrid Architecture** (feeding Kalman latent state estimations and uncertainty bounds into DeepAR for extra credit evaluation).

---
*Created for CSAI Student Project 5 - Phase 1 Submission.*
