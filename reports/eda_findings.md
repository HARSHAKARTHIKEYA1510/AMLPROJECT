# Exploratory Data Analysis & Statistical Modeling Insights
**Project:** Inventory Demand Forecasting via Kalman Filters & DeepAR Global Models (CSAI Phase 1)  
**Dataset:** Walmart M5 Forecasting - Accuracy  
**Document Type:** Comprehensive Phase 1 EDA & Theoretical Justification  

---

## 1. Research Overview & Inventory Forecasting Context

Demand forecasting in large-scale modern retail presents complex structural challenges that render standard univariate classical methods (such as standard ARIMA or naive exponential smoothing) ineffective. The Walmart M5 dataset exhibits three dominant characteristics:
1. **Severe Zero-Inflation & Intermittency:** The majority of SKUs experience non-consecutive, sporadic sales.
2. **Deep Hierarchical Aggregation:** Unit sales span 30,490 SKUs nested within 7 departments, 3 categories, 10 stores, and 3 states (CA, TX, WI).
3. **Exogenous Calendar & Policy Drivers:** Massive structural demand shifts induced by the Supplemental Nutrition Assistance Program (SNAP), major holiday closures (Christmas), and weekend purchasing surges.

This investigation establishes the empirical and theoretical foundation for deploying **State-Space Kalman Filters** (for continuous latent demand tracking) and **DeepAR Global Models** (for cross-series probabilistic representation learning).

---

## 2. Demand Categorization via Syntetos-Boylan & Croston Framework

Using the Syntetos-Boylan classification scheme, all 30,490 time series were categorized based on:
- **Average Demand Interval (ADI):** Mean time gap (in days) between successive positive transactions (cut-off: alpha = 1.32).
- **Squared Coefficient of Variation (CV^2):** Variance of non-zero order quantities normalized by the mean squared (cut-off: beta = 0.49).

### Empirical Distribution Across 30,490 SKUs
| Demand Category | Diagnostic Criteria | SKU Count | Percentage | Forecasting Challenge & Methodological Suitability |
| :--- | :--- | :--- | :--- | :--- |
| **Lumpy** | ADI >= 1.32, CV^2 >= 0.49 | **5,577** | **18.29%** | **Most severe:** Sporadic arrivals AND volatile order quantities. Requires probabilistic state-space / DeepAR. |
| **Intermittent** | ADI >= 1.32, CV^2 < 0.49 | **22,121** | **72.55%** | Sporadic occurrence with steady volume. Suited to Croston / SBA / Kalman Filters. |
| **Erratic** | ADI < 1.32, CV^2 >= 0.49 | **880** | **2.89%** | Regular transactions but fluctuating basket sizes. Global neural networks capture cross-item volume variance. |
| **Smooth** | ADI < 1.32, CV^2 < 0.49 | **1,912** | **6.27%** | Highly regular, predictable demand. Classical ML / linear baselines perform well. |

> **Key Theoretical Takeaway:** Over **75% of the inventory** falls into the *Lumpy* or *Intermittent* regimes. This confirms that classical Gaussian assumptions are violated at the SKU level, mathematically validating the necessity of state-space filtering and non-Gaussian probabilistic models.

---

## 3. Temporal Dynamics, Seasonality, and Exogenous Shocks

### A. Weekly Seasonality (Day-of-Week Effect)
Aggregated daily demand reveals a sharp, deterministic weekly cycle:
- **Weekend Surge:** Saturday (**Index: 0.9542**) and Sunday (**Index: 0.8852**) account for peak foot traffic and basket volume.
- **Mid-Week Dip:** Monday through Thursday exhibit lower baseline indices (~0.90 to 0.94).
- **Implication:** Features must include cyclical day-of-week embeddings and order-7 seasonal lag operators.

### B. Monthly & Annual Cycles
- Demand exhibits seasonal contraction in February (Index ~0.92) and steady acceleration through summer (July/August Index ~1.04).
- Year-over-year volume demonstrates sustained organic growth from 2011 to 2015.

### C. The SNAP (Food Assistance) Policy Effect
The Supplemental Nutrition Assistance Program (SNAP) injects massive purchasing power on scheduled days each month:
- **California (CA):** Daily sales uplift of **+7.56%** on SNAP distribution days.
- **Texas (TX):** Daily sales uplift of **+11.05%** on SNAP distribution days.
- **Wisconsin (WI):** Daily sales uplift of **+21.31%** on SNAP distribution days.
- **Category Specificity:** The uplift is concentrated in `FOODS` departments, creating sharp demand pulses that require state-space intervention matrices or binary covariate indicators.

### D. Holiday Shocks & Store Closure Anomaly
- **Christmas Day:** Total chain sales drop to near zero units chain-wide because all Walmart stores are closed.
- **Major Uplift Events:** Significant demand surges precede major national cultural and sporting events (Super Bowl, Thanksgiving Eve, Labor Day, Memorial Day).

---

## 4. Stationarity and Autocorrelation Analysis

- **Raw Sales Level:** Augmented Dickey-Fuller test yielded t = -1.5654 (p = 0.5010), confirming non-stationarity driven by trend and multi-scale seasonality.
- **First Differencing (Delta y_t):** ADF test on Delta y_t yielded t = -23.4596 (p = 0.00e+00), decisively establishing stationary dynamics suitable for stationary Kalman state equations.
- **ACF / PACF Structure:** Pronounced, statistically significant autocorrelation peaks occur at integer multiples of 7:
  - Lag 1: r = 0.6734
  - Lag 7 (Weekly cycle): r = 0.8082
  - Lag 14: r = 0.7364
  - Lag 28 (Monthly cycle): r = 0.8164

---

## 5. Theoretical Justification for Kalman Filters & DeepAR

### Why the Kalman Filter for Phase 1?
1. **Separation of Latent Demand from Observation Noise:** Daily sales observations z_t are noisy manifestations of an underlying, unobserved true demand state x_t. The measurement equation:
   z_t = H x_t + v_t,  v_t ~ N(0, R)
   allows filtering out high-frequency zero spikes while tracking true consumption velocity.
2. **Recursive Bayesian Updating:** The state-space formulation enables instantaneous online updating (O(1) time complexity per observation) as new daily sales arrive without retraining the entire history.
3. **Uncertainty Quantification:** The filter maintains the error covariance matrix P_t, providing exact confidence intervals for inventory safety stock calculations:
   Safety Stock = z_alpha * sqrt(H P_(t+1|t) H^T + R)

### Why DeepAR Global Models for Phase 2?
1. **Cross-Series Representation Learning:** With 30,490 time series, local models cannot share statistical strength across related SKUs. DeepAR trains an autoregressive Recurrent Neural Network (LSTM/GRU) across all series simultaneously.
2. **Parametric Negative Binomial / Zero-Inflated Distributions:** DeepAR replaces Gaussian assumptions with Negative Binomial or Student-t distributions, directly handling integer-valued intermittent sales.
3. **Hybrid Architecture Potential:** Combining Kalman filter state-level estimates as input features to DeepAR global networks creates an optimal hybrid model that captures both local state dynamics and global cross-series patterns.

---

## 6. Generated Visual Artifacts

The following 9 high-resolution figures (300 DPI) were rendered and saved to `reports/figures/`:
1. `01_hierarchical_sales_overview.png`: Multi-tier historical sales trajectories and holiday anomalies.
2. `02_intermittent_croston_matrix.png`: Syntetos-Boylan ADI vs CV^2 demand quadrant scatter and donut distribution.
3. `03_zero_demand_distribution.png`: Kernel density estimation contrasting raw vs active shelf zero rates.
4. `04_seasonality_and_calendar_effects.png`: Day-of-week and monthly normalized demand indices.
5. `05_holiday_and_snap_impact.png`: State-wise SNAP impact and major event demand uplift ranking.
6. `06_price_elasticity_analysis.png`: Category price dispersion and department price variations.
7. `07_store_level_comparison.png`: Volume, variance, and geographic clustering across 10 stores.
8. `08_acf_pacf_stationarity.png`: Multi-harmonic weekly ACF decay and ADF stationarity scorecard.
9. `09_kalman_filter_motivation.png`: State-space latent demand level estimation on intermittent SKU demand.

---
*Authored autonomously for CSAI Student Project 5 - Phase 1 Submission.*
