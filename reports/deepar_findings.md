# AML Benchmark Report: Global DeepAR Probabilistic Forecaster
**Project 5:** Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models  
**Course:** Advanced Machine Learning (AML)  
**Evaluation Horizon:** `2016-03-28` to `2016-04-24` (28 days)  
**Series Pool:** 50 Product-Store Series across Walmart CA, TX, and WI stores  
**Generated At:** 2026-10-24T11:16:39Z  

---

## 📌 Executive Summary & Four-Way Benchmark

This report documents the rigorous formulation, implementation, and empirical evaluation of **Amazon's DeepAR (Global Autoregressive Recurrent Neural Network)**.
DeepAR is trained **globally across all product-store series simultaneously**, sharing statistical representations across sparse and high-velocity SKUs.
It is evaluated on the **exact same 50 series, identical validation dates, and the identical 28-day out-of-sample forecast horizon** as the Seasonal-Naive, GBDT, and Structural Kalman Filter models.

### Four-Way Model Performance Benchmark Table

| Model Architecture | MAE (Units) | RMSE (Units) | WAPE (%) | Mean Pinball Loss | 80% CI Coverage | Uncertainty Protocol |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Seasonal-Naive (7-day Persistence)** | 1.1143 | 2.6030 | 95.59% | - | - | Deterministic weekly persistence |
| **GBDT (Recursive Multi-Step LightGBM)** | 0.9512 | 2.2679 | 81.60% | - | - | Deterministic recursive rollout |
| **Kalman Filter per SKU (Local Level + Seasonality)** | 0.9333 | 2.3303 | 80.06% | - | 80.93% | Exact closed-form covariance $\mathbf{P}_t$ |
| **Global DeepAR (Autoregressive LSTM)** | **0.9928** | **2.2082** | **85.16%** | **0.3075** | **90.07%** | **Monte Carlo Quantile Rollout** |

---

## 📐 Mathematical Formulation of Global DeepAR

The Global DeepAR model formulates the joint probability distribution of the future trajectory $\mathbf{y}_{i, T+1:T+H}$ conditioned on historical observations $\mathbf{y}_{i, 1:T}$, static features $\mathbf{e}_i$, and dynamic exogenous covariates $\mathbf{x}_{i, 1:T+H}$:

### 1. Autoregressive Factorization
$$P(\mathbf{y}_{i, T+1:T+H} \mid \mathbf{y}_{i, 1:T}, \mathbf{x}_{i, 1:T+H}) = \prod_{h=1}^H p(y_{i, T+h} \mid \mathbf{h}_{i, T+h})$$

where $\mathbf{h}_{i, t}$ is the recurrent state computed by a multi-layer Long Short-Term Memory (LSTM) network:
$$\mathbf{h}_{i, t} = \text{LSTM}(\mathbf{h}_{i, t-1}, [\tilde{y}_{i, t-1}, \mathbf{e}_i, \mathbf{x}_{i, t}])$$

### 2. Item Scale Normalization
To prevent high-volume items from dominating gradient updates and ensure scale-invariance across sparse retail time series, each series is normalized by its historical scale factor:
$$\nu_i = 1.0 + \frac{1}{T} \sum_{t=1}^T y_{i, t}$$
$$\tilde{y}_{i, t} = \frac{y_{i, t}}{\nu_i}$$

### 3. Conditional Probability Distribution
At each step $t$, the recurrent state $\mathbf{h}_{i, t}$ is mapped to distribution parameters $\theta_{i, t}$:
- **Gaussian Likelihood (Normalized Space)**:
  $$\mu_{i, t} = \mathbf{W}_\mu^T \mathbf{h}_{i, t} + b_\mu$$
  $$\sigma_{i, t} = \text{softplus}(\mathbf{W}_\sigma^T \mathbf{h}_{i, t} + b_\sigma) + \epsilon$$
  $$y_{i, t} \sim \mathcal{N}(\mu_{i, t} \nu_i, (\sigma_{i, t} \nu_i)^2)$$

### 4. Non-Negative Demand Guarantee
Retail inventory demand cannot be physically negative ($y_{i, t} \ge 0$).
During Monte Carlo trajectory sampling, sampled values are strictly bounded at zero:
$$\hat{y}_{i, T+h}^{(s)} = \max(0.0, \tilde{y}_{i, T+h}^{(s)} \cdot \nu_i)$$
The clipped non-negative value is then scaled and recursively injected as the target input $\tilde{y}_{i, T+h}^{(s)} / \nu_i$ for step $h+1$.

---

## 🛡️ Zero Data Leakage Protocol

1. **Chronological Horizon Separation**:
   - The training set strictly ends at **2016-03-28** (minus 1 day).
   - Validation strictly covers the 28 subsequent calendar days (2016-03-28 to 2016-04-24).
2. **Causal Rollout**:
   - Validation actuals are strictly withheld during inference rollout.
   - For step $h=1$, the model conditions on training history up to origin $T$.
   - For subsequent steps $h \in [2, 28]$, the model feeds its own sampled predictions from previous steps.
3. **Safe Zero-Denominator Handling**:
   - WAPE is computed as $\frac{\sum |y - \hat{y}|}{\sum y}$, returning 0.0 if both sums are 0, avoiding divide-by-zero crashes.
4. **Pinball Loss Formula**:
   $$L_q(y, \hat{q}_q) = \frac{1}{N} \sum_{i=1}^N \max(q(y_i - \hat{q}_{q, i}), (1 - q)(\hat{q}_{q, i} - y_i))$$
   - Pinball Loss at $P_{10}$: **0.1126**
   - Pinball Loss at $P_{50}$: **0.4964**
   - Pinball Loss at $P_{90}$: **0.3136**
   - Mean Pinball Loss: **0.3075**
5. **Prediction Interval Calibration**:
   - 80% Confidence Interval ($[P_{10}, P_{90}]$) Empirical Coverage: **90.07%**
   - Average Interval Width: **2.87 units**

---

## 📊 Category-Level Performance Decomposition

| Category | Actual Demand | DeepAR MAE | DeepAR RMSE | DeepAR WAPE | 80% CI Coverage | Mean 80% Width |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FOODS** | 1179 units | 1.4373 | 3.0999 | 75.10% | **91.7%** | 4.47 units |
| **HOBBIES** | 180 units | 0.6574 | 1.2289 | 143.17% | **85.5%** | 1.35 units |
| **HOUSEHOLD** | 273 units | 0.6295 | 0.8967 | 90.39% | **92.1%** | 1.88 units |

---

## 🔬 Architectural Trade-Off Analysis: Local vs Global Models

| Dimension | Structural Kalman Filter per SKU | Global DeepAR Recurrent Network |
| :--- | :--- | :--- |
| **Model Scope** | Local: 50 independent state-space models | Global: 1 unified network trained across all 50 SKUs |
| **Statistical Sharing** | Zero cross-series sharing (isolated states) | High cross-series sharing via static embeddings & recurrent weights |
| **Uncertainty Quantification** | Exact closed-form covariance propagation $\mathbf{P}_t$ | Empirical Monte Carlo quantile estimation (200 sample paths) |
| **Exogenous Features** | Relies on structural dynamics (level + seasonality) | Ingests calendar cyclical signals, event markers, and SNAP schedules |
| **Intermittency Handling** | Linear continuous state update with zero preservation | Scale-normalized likelihood with non-negative clipping |

---

## 📈 Visual Artifacts Generated

The following publication-grade figures (300 DPI) have been rendered into `reports/figures/`:
1. `deepar_forecasts.png`: System-level daily tracking comparing actual sales against DeepAR predictions with an 80% prediction interval envelope.
2. `deepar_prediction_intervals.png`: Representative SKU time-series trajectories showing historical context, validation cutoff, point predictions, and [P10, P90] predictive intervals.
3. `16_four_way_model_comparison.png`: Comprehensive benchmark comparison across Seasonal-Naive, GBDT, Kalman Filter, and Global DeepAR.

---
*Report automatically generated by `src/run_deepar_pipeline.py`.*
