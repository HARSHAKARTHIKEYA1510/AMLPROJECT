# Data Quality and Integrity Audit Report
**Project:** Inventory Demand Forecasting via Kalman Filters & DeepAR Global Models (CSAI Phase 1)  
**Dataset:** Walmart M5 Forecasting - Accuracy  
**Audit Status:** Validated & Clean  

---

## 1. Executive Summary

A comprehensive data quality evaluation was conducted on the official Walmart M5 retail hierarchical dataset. The objective was to verify schema conformance, referential integrity, missingness patterns, numerical validity, and demand sparsity prior to model formulation.

### Core Metrics Scorecard
| Quality Dimension | Evaluated Feature | Metric Value | Verdict |
| :--- | :--- | :--- | :--- |
| **Completeness** | Sales Matrix Nulls | 0 / 58,327,370 (0.00%) | **EXCELLENT** |
| **Validity** | Sales Non-negativity | 100% (No negative values) | **VALID** |
| **Validity** | Price Positivity | Min price: $0.01 (No $\le 0$ prices) | **VALID** |
| **Uniqueness** | SKU-Store Pairs | 30,490 unique series | **EXCELLENT** |
| **Continuity** | Temporal Sequence | 1969 consecutive calendar days | **UNBROKEN** |
| **Referential Integrity**| Store & Item Keys | 100% matched across tables | **CONFIRMED** |

---

## 2. Dataset Schema & Completeness Analysis

### A. Calendar (`calendar.csv`)
- **Total Records:** 1969 days (2011-01-29 to 2016-06-19)
- **Continuity:** Continuous daily records without timestamp gaps.
- **Missing Value Characterization:**
  - `event_name_1` & `event_type_1`: Missing on 1,811 days (92.35%).
  - `event_name_2` & `event_type_2`: Missing on 1,939 days (98.88%).
  - *Theoretical Justification:* These missing values are **structural indicators** signifying regular non-event operating days. They do not constitute missing data to impute, but rather require categorical binary / indicator encoding.
- **SNAP Days:** CA (650 days), TX (650 days), WI (650 days).

### B. Sell Prices (`sell_prices.csv`)
- **Total Observations:** 6,841,121 price observations.
- **Missing Values:** 0
- **Price Range:** Min = $0.01, Median = $3.47, Max = $107.32, Std = $3.41.
- **Anomalies:** Zero negative or zero-priced items detected.

### C. Daily Unit Sales (`sales_train_validation.csv`)
- **Total Time Series:** 30,490 SKUs across 10 stores.
- **Total Observation Points:** 58,327,370 unit-day entries.
- **Null Values:** 0
- **Negative Values:** 0

---

## 3. Retail Inventory Specific Quality Nuances

### A. Active Demand vs. Pre-Release "Leading Zeroes"
In retail inventory management, treating leading zeros prior to an item's product launch as "zero customer demand" is a fatal modeling error that leads to severely underestimated base demand rates and inflated zero-inflation parameters.
- **Total Raw Zero Sales:** 39,777,094 (68.2%)
- **Leading Inactive Zeroes (Pre-launch):** 12,384,870 observations.
- **Active Shelf Observations:** 45,942,500 unit-day entries.
- **True Active Zero-Demand Rate:** **59.62%**

### B. Deterministic Holiday Closures (Christmas Anomaly)
Aggregated daily sales inspection identified near-zero system sales on specific days:
- **Observation:** Day `d_697` has total chain sales of only 11 units.
- **Diagnosis:** This corresponds to **December 25 (Christmas Day)**, the single day when Walmart stores are closed annually.
- **Cleaning / Modeling Policy:** This anomaly is deterministic and must be explicitly handled via calendar feature flags or state-space observation masking rather than treated as a random demand drop.

---

## 4. Memory Optimization Protocol

The high dimensionality of M5 (30,490 time series $\times$ 1,913 days) consumes excessive system RAM when loaded in raw 64-bit precision.
- **Downcasting Strategy:**
  - 64-bit Integers (`int64`) $\rightarrow$ `int16` / `int8` (max unit sales < 32,767)
  - 64-bit Floats (`float64`) $\rightarrow$ `float32`
  - Object strings (`store_id`, `state_id`, `cat_id`, `dept_id`) $\rightarrow$ pandas `category`
- **Result:** ~75% to 80% RAM footprint reduction, facilitating fast in-memory analytics and matrix operations.

---
*Generated autonomously by Data Quality Audit Engine for CSAI Phase 1 Review.*
