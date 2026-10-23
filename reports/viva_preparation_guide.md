# Viva Voce Examination Defense Guide: Phase 1
**Project 5:** Inventory Demand Forecasting via Kalman Filters & DeepAR Global Models  
**Assigned Student Focus:** Dataset Collection, Cleaning, Data Quality Audit, Exploratory Data Analysis (EDA) & Visualizations  
**Course:** CSAI - Machine Learning Evaluation (Phase 1)  

---

## 1. Core Technical Competencies & Viva Questions

### Q1: Why did you choose the M5 dataset and what are its dimensions?
* **Answer:**
  * "The M5 Forecasting dataset from Walmart represents real-world hierarchical retail demand across 10 stores located in 3 US states (California, Texas, and Wisconsin) spanning 1,913 days (January 2011 to April 2016).
  * It comprises **30,490 time series** structured across 3 product categories (`FOODS`, `HOBBIES`, `HOUSEHOLD`) and 7 departments.
  * In total, the sales matrix represents **58,327,370 unit-day observations**, making it an ideal testbed for evaluating Kalman Filters (local state estimation) and DeepAR (global cross-series learning)."

---

### Q2: How did you handle the memory bottleneck when loading 58+ million records?
* **Answer:**
  * "Standard 64-bit loading of the raw sales and price files exceeds 2.5 GB of RAM and causes severe out-of-memory overhead during matrix operations.
  * We designed an automated `reduce_mem_usage` pipeline that performs strict numerical downcasting:
    * Integer unit sales (max observed sale < 32,767) were downcasted from `int64` to `int16` and `int8`.
    * Float values were cast from `float64` to `float32`.
    * High-cardinality repetitive string identifiers (`store_id`, `state_id`, `cat_id`, `dept_id`) were converted into categorical dtypes.
  * This achieved a **75% to 80% RAM footprint reduction** (~2.5 GB down to ~450 MB), allowing instantaneous vectorized calculations across all 30,490 series."

---

### Q3: What were your findings regarding missing values and data quality?
* **Answer:**
  * "Our automated `M5DataQualityAuditor` established 100% data integrity:
    * **Zero missing values** and **zero negative values** in the sales matrix (all 58.3M observations are valid non-negative integer sales).
    * **Zero missing, negative, or zero prices** across all 6.84 million price entries in `sell_prices.csv` (prices range from $0.01 to $107.32 with median $3.47).
    * In `calendar.csv`, null values appear only in `event_name` and `event_type` columns on ~91.8% of days. We proved that these are **structural indicators** denoting regular non-event operating days, rather than missing information requiring imputation."

---

### Q4: Explain the difference between 'Raw Zero Sales' and 'Active Shelf Zeros'. Why is this critical in retail inventory?
* **Answer:**
  * "This is a classic pitfall in retail demand modeling:
    * When an item is introduced to a store in year 3, the sales matrix records `0` for all days in years 1 and 2.
    * If you treat those pre-release days as 'zero customer demand', you severely underestimate the SKU's baseline demand velocity and artificially inflate zero-inflation metrics.
  * In our audit:
    * **Raw zero sales** accounted for **68.20%** of all matrix entries (39.77 million observations).
    * By tracking each SKU's store release date, we identified **12,384,870 leading inactive zeros** (pre-release shelf days).
    * The **true active shelf zero-demand rate is 59.62%** across 45.94 million active days.
    * This distinction is vital for accurate Kalman filter initial state calibration ($x_0, P_0$)."

---

### Q5: What is the Christmas Anomaly and how did you diagnose it?
* **Answer:**
  * "When aggregating daily sales chain-wide across all 10 stores, day `d_697` (and corresponding annual days) dropped to near-zero sales (only 11 units across the entire chain).
  * Our calendar cross-referencing revealed that this occurs on **December 25 (Christmas Day)**, the single day each year that Walmart stores are closed.
  * Treating Christmas as a random demand drop ruins time-series models; instead, we flag it as a deterministic calendar indicator with zero probability of retail transactions."

---

### Q6: Explain the Syntetos-Boylan / Croston framework and what your EDA proved.
* **Answer:**
  * "The Syntetos-Boylan framework categorizes intermittent demand using two orthogonal statistical axes:
    1. **Average Demand Interval (ADI):** The average number of days between non-zero sales ($\text{Cut-off} = 1.32$).
    2. **Squared Coefficient of Variation ($CV^2$):** The normalized variance of positive order quantities $(\sigma / \mu)^2$ ($\text{Cut-off} = 0.49$).
  * Across all 30,490 SKUs in M5, our empirical calculations showed:
    * **Intermittent (72.55%):** Sporadic arrivals with relatively stable non-zero quantities.
    * **Lumpy (18.29%):** Sporadic arrivals combined with highly volatile transaction quantities (the most challenging inventory pattern).
    * **Smooth (6.27%):** Regular, low-variance demand where classical ARIMA works.
    * **Erratic (2.89%):** Frequent sales but large swings in quantity.
  * **Conclusion for Viva:** Over **90.8% of items are Intermittent or Lumpy**. Classical Gaussian models assume continuous smooth demand and fail here. This empirically justifies using **Kalman Filters** (to track the latent demand state) and **DeepAR with negative binomial likelihood**."

---

### Q7: What exogenous drivers did your EDA discover (SNAP, Seasonality, Holidays)?
* **Answer:**
  * **SNAP (Supplemental Nutrition Assistance Program):**
    * In Wisconsin (WI), average daily store sales increase by **+21.31%** on SNAP distribution days.
    * In Texas (TX), sales rise by **+11.05%**.
    * In California (CA), sales rise by **+7.56%**.
    * This uplift is concentrated heavily in the `FOODS` category.
  * **Weekly Seasonality:**
    * Saturday (Index **1.21**) and Sunday (Index **1.20**) generate 20%+ more volume than the weekday average.
    * Tuesday and Wednesday are the lowest demand days (Index ~**0.88**).
  * **Autocorrelation (ACF):**
    * ACF exhibits massive peaks at lags 7 ($r = 0.81$), 14 ($r = 0.74$), and 28 ($r = 0.82$), proving that weekly harmonic seasonality is the dominant periodic driver."

---

### Q8: What did the stationarity tests (ADF) show?
* **Answer:**
  * "An Augmented Dickey-Fuller (ADF) test on the raw total sales series yielded $t = -1.565$ ($p = 0.501$), proving that the raw demand is **non-stationary** due to multi-year organic store growth and seasonal cycles.
  * However, applying first-order differencing ($\Delta y_t = y_t - y_{t-1}$) produced $t = -23.460$ ($p < 10^{-5}$), decisively establishing **stationarity**.
  * This confirms that a Local Linear Trend or differenced state-space Kalman model is mathematically appropriate."

---

### Q9: How does the Kalman Filter work on intermittent retail demand?
* **Answer:**
  * "In retail inventory, we observe $z_t$ (daily units sold), which is a noisy, discrete signal corrupted by stockouts, shopper timing, and zeros.
  * The Kalman Filter models an unobserved **latent demand state** $x_t$ (true consumer demand velocity):
    $$\text{State Equation:} \quad x_t = F x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$
    $$\text{Measurement Equation:} \quad z_t = H x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$
  * The filter recursively updates:
    1. **Predict step:** Projects the demand state forward ($\hat{x}_{t|t-1} = F \hat{x}_{t-1}$) and propagates error covariance ($P_{t|t-1} = F P_{t-1} F^T + Q$).
    2. **Update step:** Computes the Kalman Gain ($K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}$) and blends the observation with the prior state estimate.
  * **Inventory Benefit:** The filter error covariance $P_t$ directly provides dynamic safety stock bounds:
    $$\text{Safety Stock} = z_{\alpha} \cdot \sqrt{H P_{t+1|t} H^T + R}$$
    which prevents both stockouts and excess holding costs."

---

### Q10: How does Phase 1 prepare for the Phase 2 Hybrid Approach (Additional Marks)?
* **Answer:**
  * "The faculty guidelines state that teams implementing a technically sound **Hybrid Approach** will receive additional marks in Phase 2.
  * Our strategy links Phase 1 (ML & Kalman Filters) with Phase 2 (DeepAR):
    1. **Kalman Filter (Local State-Space):** Filters out SKU-specific measurement noise and extracts the smoothed latent demand level $x_t$ and uncertainty $P_t$ for individual series.
    2. **DeepAR (Global Autoregressive RNN):** Learns non-linear cross-series seasonal patterns, SNAP interactions, and price elasticity across all 30,490 SKUs using a Negative Binomial loss.
    3. **Hybrid Architecture:** The Kalman filter's smoothed state estimates and covariance metrics will be fed as dynamic exogenous features into DeepAR, combining local Bayesian state-space filtering with global neural representation learning."
