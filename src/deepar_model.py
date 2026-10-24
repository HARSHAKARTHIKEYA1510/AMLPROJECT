"""
deepar_model.py - Global DeepAR Probabilistic Forecasting Model for M5 Demand.

CSAI Student Project 5:
Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models.

This module implements a global probabilistic neural time-series model based on:
"DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"
(Salinas, Flunkert, Gasthaus, Januschowski, IJF 2020).

Key Architectural Components:
1. Global Parameter Sharing:
   A single autoregressive Recurrent Neural Network (LSTM) is trained globally
   across all product-store series, sharing statistical strength across sparse,
   intermittent, and high-velocity retail demand series.

2. Static Entity Embeddings:
   Low-dimensional continuous representations for store_id, cat_id, dept_id,
   state_id, and series_id allow the network to learn hierarchical relationships
   across the retail catalog.

3. Dynamic Exogenous Covariates:
   Known calendar signals (day_sin, day_cos, month_sin, month_cos, is_weekend),
   event flags (is_event_day, is_christmas), and policy schedules (snap_active)
   guide both historical sequence modeling and future 28-day forecasting.

4. Item Scale Normalization:
   Each series is normalized by its historical scale nu_i = 1.0 + mean(y_i),
   enabling scale-invariant training across items with orders-of-magnitude differences
   in demand volume.

5. Probabilistic Likelihood Heads:
   - Gaussian Likelihood (Default): Outputs parameters mu_t and sigma_t = softplus(.) + eps.
     During rollout, samples are drawn from N(mu, sigma^2), unscaled, and clipped at 0.0.
   - Negative Binomial Likelihood: Models count demand via mean mu_t and dispersion alpha_t.

6. Strictly Causal Multi-Step Rollout:
   During validation forecasting, actual validation sales are strictly withheld.
   Predictions are rolled out autoregressively for 28 steps: at each step, a sample
   is drawn from the predicted distribution, clipped non-negatively, and fed as the
   target input to the next step.

7. Uncertainty Quantification:
   Monte Carlo quantile estimation (P10, P50, P90), Pinball loss, and 80% prediction
   interval coverage and width.
"""

import sys
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_models import compute_forecasting_metrics
from src.kalman_filter_model import compute_prediction_interval_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("GlobalDeepAR")


def compute_pinball_loss(
    y_true: Union[np.ndarray, pd.Series],
    y_pred_quantile: Union[np.ndarray, pd.Series],
    quantile: float
) -> float:
    """
    Computes empirical pinball loss (quantile loss) for a specified quantile tau in (0, 1).
    
    Formula:
        L_tau(y, q) = (1 / N) * sum( max(tau * (y - q), (1 - tau) * (q - y)) )
    
    Special Cases:
        - tau = 0.50 corresponds to 0.5 * MAE.
        - Perfectly calibrated quantiles minimize pinball loss at that quantile level.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"Quantile must be in (0, 1), received {quantile}.")

    y_t = np.asarray(y_true, dtype=np.float64)
    q_p = np.asarray(y_pred_quantile, dtype=np.float64)

    if len(y_t) == 0 or len(q_p) == 0:
        raise ValueError("Cannot compute pinball loss on empty arrays.")
    if len(y_t) != len(q_p):
        raise ValueError(f"Shape mismatch: y_true ({len(y_t)}) vs y_pred_quantile ({len(q_p)}).")

    diff = y_t - q_p
    loss = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    return float(np.mean(loss))


def compute_deepar_probabilistic_metrics(
    y_true: Union[np.ndarray, pd.Series],
    p10: Union[np.ndarray, pd.Series],
    p50: Union[np.ndarray, pd.Series],
    p90: Union[np.ndarray, pd.Series],
    point_pred: Optional[Union[np.ndarray, pd.Series]] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of DeepAR point and probabilistic forecasts.
    
    Computes:
    1. Standard Point Forecast Metrics: MAE, RMSE, WAPE (using point_pred or p50 median).
    2. Pinball Losses: at tau = 0.10, tau = 0.50, tau = 0.90, and mean pinball loss.
    3. 80% Prediction Interval Calibration: Coverage and average interval width [P10, P90].
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    p10_arr = np.asarray(p10, dtype=np.float64)
    p50_arr = np.asarray(p50, dtype=np.float64)
    p90_arr = np.asarray(p90, dtype=np.float64)

    eval_point = np.asarray(point_pred, dtype=np.float64) if point_pred is not None else p50_arr

    # Standard point metrics
    point_metrics = compute_forecasting_metrics(y_t, eval_point)

    # Pinball losses
    pinball_10 = compute_pinball_loss(y_t, p10_arr, quantile=0.10)
    pinball_50 = compute_pinball_loss(y_t, p50_arr, quantile=0.50)
    pinball_90 = compute_pinball_loss(y_t, p90_arr, quantile=0.90)
    mean_pinball = float((pinball_10 + pinball_50 + pinball_90) / 3.0)

    # Prediction interval coverage and width (80% CI: [P10, P90])
    pi_metrics = compute_prediction_interval_metrics(y_t, lower_bound=p10_arr, upper_bound=p90_arr)

    return {
        **point_metrics,
        "pinball_loss": {
            "p10": round(pinball_10, 4),
            "p50": round(pinball_50, 4),
            "p90": round(pinball_90, 4),
            "mean": round(mean_pinball, 4),
        },
        "prediction_intervals": {
            "ci_80": {
                "coverage": pi_metrics["coverage"],
                "mean_width": pi_metrics["mean_width"],
                "inside_count": pi_metrics["inside_count"],
                "total_count": pi_metrics["total_count"],
            }
        }
    }


class DeepARDataProcessor:
    """
    Data Preprocessor and Batch Extractor for Global DeepAR.
    
    Responsibilities:
    1. Integer-encodes static categoricals (store_id, cat_id, dept_id, state_id, id).
    2. Standardizes / selects dynamic exogenous covariates.
    3. Computes series-specific historical scale factors nu_i = 1.0 + mean(y_i).
    4. Extracts sliding windows for teacher-forced sequence training.
    5. Extracts historical context windows and future dynamic covariates for inference.
    """

    DEFAULT_DYNAMIC_FEATURES = [
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "has_event_1",
        "has_event_2",
        "is_event_day",
        "is_christmas",
        "snap_active",
    ]

    STATIC_CATEGORICAL_COLS = ["store_id", "cat_id", "dept_id", "state_id", "id"]

    def __init__(
        self,
        dynamic_feature_cols: Optional[List[str]] = None,
        context_length: int = 56,
        prediction_length: int = 28,
        stride: int = 14,
    ):
        self.dynamic_feature_cols = dynamic_feature_cols or self.DEFAULT_DYNAMIC_FEATURES
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.window_length = context_length + prediction_length

        self.cat_mappings: Dict[str, Dict[str, int]] = {}
        self.cat_cardinalities: Dict[str, int] = {}
        self.series_scales: Dict[str, float] = {}
        self.series_metadata: Dict[str, Dict[str, Any]] = {}
        self.unique_series_ids: List[str] = []
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "DeepARDataProcessor":
        """
        Learns categorical encodings and item scale factors strictly from training data.
        """
        logger.info("Fitting DeepARDataProcessor on training observations...")
        df = train_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # 1. Fit categorical encodings
        for col in self.STATIC_CATEGORICAL_COLS:
            if col in df.columns:
                unique_vals = sorted(df[col].dropna().unique().tolist())
                self.cat_mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
                self.cat_cardinalities[col] = len(unique_vals)
            else:
                logger.warning(f"Categorical column '{col}' not found in training dataframe.")

        self.unique_series_ids = sorted(df["id"].unique().tolist())

        # 2. Compute series-level scale factors nu_i = 1.0 + mean(demand)
        for sid, grp in df.groupby("id", observed=True):
            mean_demand = float(grp["demand"].mean())
            self.series_scales[str(sid)] = float(1.0 + max(0.0, mean_demand))

            first_row = grp.iloc[0]
            self.series_metadata[str(sid)] = {
                "store_id": str(first_row.get("store_id", "")),
                "cat_id": str(first_row.get("cat_id", "")),
                "dept_id": str(first_row.get("dept_id", "")),
                "state_id": str(first_row.get("state_id", "")),
                "item_id": str(first_row.get("item_id", "")),
                "scale": self.series_scales[str(sid)],
            }

        self.is_fitted = True
        logger.info(
            f"Processor fitted: {len(self.unique_series_ids)} unique series, "
            f"{len(self.dynamic_feature_cols)} dynamic features, "
            f"mean scale: {np.mean(list(self.series_scales.values())):.2f} units."
        )
        return self

    def prepare_training_dataset(
        self,
        train_df: pd.DataFrame,
        batch_size: int = 64,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None
    ) -> DataLoader:
        """
        Extracts sliding sequence windows from historical training records.
        
        Returns a DataLoader yielding:
            target_scaled: Tensor of shape (batch, window_length)
            dynamic_feats: Tensor of shape (batch, window_length, n_dynamic)
            cat_feats: Tensor of shape (batch, n_static_cats)
            scale: Tensor of shape (batch, 1)
        """
        if not self.is_fitted:
            self.fit(train_df)

        df = train_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        all_target_scaled = []
        all_dynamic_feats = []
        all_cat_feats = []
        all_scales = []

        logger.info(
            f"Extracting training sequence windows (window_length={self.window_length}, "
            f"context={self.context_length}, horizon={self.prediction_length}, stride={self.stride})..."
        )

        for sid in self.unique_series_ids:
            s_df = df[df["id"] == sid].sort_values("date")
            n_obs = len(s_df)
            if n_obs < self.window_length:
                continue

            scale = self.series_scales[sid]
            demands = s_df["demand"].values.astype(np.float32)
            scaled_demands = demands / scale

            # Extract dynamic features
            dyn_feats = s_df[self.dynamic_feature_cols].values.astype(np.float32)

            # Static categorical vector for this series
            cat_vec = [
                self.cat_mappings[c][s_df[c].iloc[0]]
                for c in self.STATIC_CATEGORICAL_COLS
                if c in self.cat_mappings
            ]

            # Sliding window extraction across chronological time
            for start_idx in range(0, n_obs - self.window_length + 1, self.stride):
                end_idx = start_idx + self.window_length
                w_target = scaled_demands[start_idx:end_idx]
                w_dyn = dyn_feats[start_idx:end_idx]

                all_target_scaled.append(w_target)
                all_dynamic_feats.append(w_dyn)
                all_cat_feats.append(cat_vec)
                all_scales.append([scale])

        t_target = torch.tensor(np.array(all_target_scaled, dtype=np.float32))
        t_dynamic = torch.tensor(np.array(all_dynamic_feats, dtype=np.float32))
        t_cat = torch.tensor(np.array(all_cat_feats, dtype=np.int64))
        t_scale = torch.tensor(np.array(all_scales, dtype=np.float32))

        logger.info(
            f"Training dataset ready: {len(t_target):,} sequence windows extracted "
            f"(tensor shape: {t_target.shape})."
        )

        dataset = TensorDataset(t_target, t_dynamic, t_cat, t_scale)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            drop_last=False
        )

    def prepare_inference_inputs(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Prepares conditioning context (past) and future covariates for out-of-sample inference.
        
        Zero Data Leakage Guarantee:
        Validation demand is strictly omitted from the inference tensors.
        Only the last `context_length` days of training demand and future calendar/event
        covariates from validation_df are extracted.
        """
        if not self.is_fitted:
            self.fit(train_df)

        tr_df = train_df.copy()
        tr_df["date"] = pd.to_datetime(tr_df["date"])

        v_df = val_df.copy()
        v_df["date"] = pd.to_datetime(v_df["date"])

        context_targets = []
        context_dyn_feats = []
        future_dyn_feats = []
        cat_feats = []
        scales = []
        series_id_list = []

        for sid in self.unique_series_ids:
            s_train = tr_df[tr_df["id"] == sid].sort_values("date")
            s_val = v_df[v_df["id"] == sid].sort_values("date")

            if len(s_train) < self.context_length:
                # In case of short series, pad with zeros at the beginning
                pad_len = self.context_length - len(s_train)
                pad_demand = np.zeros(pad_len, dtype=np.float32)
                hist_demand = np.concatenate([pad_demand, s_train["demand"].values.astype(np.float32)])

                first_dyn = s_train[self.dynamic_feature_cols].iloc[0].values.astype(np.float32)
                pad_dyn = np.repeat(first_dyn[np.newaxis, :], pad_len, axis=0)
                hist_dyn = np.concatenate([pad_dyn, s_train[self.dynamic_feature_cols].values.astype(np.float32)], axis=0)
            else:
                s_context = s_train.tail(self.context_length)
                hist_demand = s_context["demand"].values.astype(np.float32)
                hist_dyn = s_context[self.dynamic_feature_cols].values.astype(np.float32)

            scale = self.series_scales[sid]
            context_scaled = hist_demand / scale

            # Future dynamic covariates (from validation horizon)
            fut_dyn = s_val[self.dynamic_feature_cols].values.astype(np.float32)

            # Categorical static indices
            cat_vec = [
                self.cat_mappings[c][s_train[c].iloc[0]]
                for c in self.STATIC_CATEGORICAL_COLS
                if c in self.cat_mappings
            ]

            context_targets.append(context_scaled)
            context_dyn_feats.append(hist_dyn)
            future_dyn_feats.append(fut_dyn)
            cat_feats.append(cat_vec)
            scales.append([scale])
            series_id_list.append(sid)

        return {
            "context_target": torch.tensor(np.array(context_targets, dtype=np.float32)),
            "context_dyn": torch.tensor(np.array(context_dyn_feats, dtype=np.float32)),
            "future_dyn": torch.tensor(np.array(future_dyn_feats, dtype=np.float32)),
            "cat_feats": torch.tensor(np.array(cat_feats, dtype=np.int64)),
            "scales": torch.tensor(np.array(scales, dtype=np.float32)),
            "series_ids": series_id_list,
        }


class GlobalDeepARNet(nn.Module):
    """
    Autoregressive Recurrent Neural Network for Global Probabilistic Forecasting.
    
    Structure:
    1. Static Categorical Embeddings: store_id, cat_id, dept_id, state_id, id.
    2. Dynamic Features Concatenation.
    3. Multi-Layer Recurrent Core: LSTM with dropout.
    4. Distribution Parameter Projection Head:
       - Gaussian Head: outputs (mu, sigma) with sigma = softplus(.) + 1e-4.
       - Negative Binomial Head: outputs (mu, alpha) with alpha = dispersion.
    """

    def __init__(
        self,
        cat_cardinalities: Dict[str, int],
        n_dynamic_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        distribution: str = "gaussian",
        embedding_dims: Optional[Dict[str, int]] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.distribution = distribution.lower()
        if self.distribution not in ["gaussian", "negative_binomial"]:
            raise ValueError(f"Unsupported distribution '{distribution}'. Choose 'gaussian' or 'negative_binomial'.")

        # Static categorical embeddings
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0

        default_embed_dims = {
            "store_id": 8,
            "cat_id": 4,
            "dept_id": 6,
            "state_id": 4,
            "id": 16,
        }
        if embedding_dims:
            default_embed_dims.update(embedding_dims)

        for cat_col, card in cat_cardinalities.items():
            dim = default_embed_dims.get(cat_col, max(2, int(math.ceil(math.log2(card + 1)))))
            self.embeddings[cat_col] = nn.Embedding(num_embeddings=card, embedding_dim=dim)
            total_embed_dim += dim

        # Input dimension: 1 (lagged target) + total_embed_dim + n_dynamic_features
        self.input_dim = 1 + total_embed_dim + n_dynamic_features

        # Recurrent Core (LSTM)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Distribution projection heads
        if self.distribution == "gaussian":
            self.fc_mu = nn.Linear(hidden_dim, 1)
            self.fc_sigma = nn.Linear(hidden_dim, 1)
        else:  # negative_binomial
            self.fc_mu = nn.Linear(hidden_dim, 1)
            self.fc_alpha = nn.Linear(hidden_dim, 1)

    def _get_static_embeddings(self, cat_feats: torch.Tensor) -> torch.Tensor:
        """
        Embeds integer categorical columns and concatenates into a single vector.
        """
        embeds = []
        for idx, (cat_col, emb_layer) in enumerate(self.embeddings.items()):
            col_indices = cat_feats[:, idx]
            embeds.append(emb_layer(col_indices))
        return torch.cat(embeds, dim=-1)

    def forward(
        self,
        target_seq: torch.Tensor,
        dynamic_feats: torch.Tensor,
        cat_feats: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with teacher forcing for sequence training.
        
        Args:
            target_seq: (batch, seq_len) - scaled demand sequence
            dynamic_feats: (batch, seq_len, n_dyn)
            cat_feats: (batch, n_cats)
            hidden: Optional (h_0, c_0)
            
        Returns:
            dist_params: (mu, sigma) or (mu, alpha), each shape (batch, seq_len, 1)
            hidden: updated (h_n, c_n)
        """
        batch_size, seq_len = target_seq.shape

        # Construct lagged target input: y_{t-1} with y_0 = 0
        lagged_target = torch.zeros_like(target_seq)
        lagged_target[:, 1:] = target_seq[:, :-1]
        lagged_target = lagged_target.unsqueeze(-1)  # (batch, seq_len, 1)

        # Static embeddings: (batch, embed_dim) -> broadcast to (batch, seq_len, embed_dim)
        static_embed = self._get_static_embeddings(cat_feats)
        static_broadcast = static_embed.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate inputs: (batch, seq_len, input_dim)
        x_in = torch.cat([lagged_target, static_broadcast, dynamic_feats], dim=-1)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(x_in, hidden)

        # Compute distribution parameters
        if self.distribution == "gaussian":
            mu = self.fc_mu(lstm_out)
            sigma = torch.nn.functional.softplus(self.fc_sigma(lstm_out)) + 1e-4
            return (mu, sigma), hidden
        else:
            mu = torch.nn.functional.softplus(self.fc_mu(lstm_out)) + 1e-2
            alpha = torch.nn.functional.softplus(self.fc_alpha(lstm_out)) + 1e-2
            return (mu, alpha), hidden

    def compute_loss(
        self,
        target: torch.Tensor,
        dist_params: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes Negative Log-Likelihood (NLL) loss.
        """
        t = target.unsqueeze(-1)  # (batch, seq_len, 1)

        if self.distribution == "gaussian":
            mu, sigma = dist_params
            # Gaussian NLL: 0.5 * log(2*pi*sigma^2) + 0.5 * (target - mu)^2 / sigma^2
            variance = sigma.pow(2)
            nll = 0.5 * torch.log(2 * math.pi * variance) + 0.5 * (t - mu).pow(2) / variance
            return nll.mean()
        else:
            mu, alpha = dist_params
            # Negative Binomial parameterization:
            # dispersion alpha > 0, total_count r = 1 / alpha, p = r / (r + mu)
            r = 1.0 / alpha
            p = r / (r + mu)
            # PyTorch NegativeBinomial: probs is probability of failure (p_fail = 1 - p)
            # where count = target
            nb_dist = torch.distributions.NegativeBinomial(total_count=r, probs=1.0 - p)
            nll = -nb_dist.log_prob(torch.clamp(t, min=0.0))
            return nll.mean()

    @torch.no_grad()
    def sample_rollout(
        self,
        context_target: torch.Tensor,
        context_dyn: torch.Tensor,
        future_dyn: torch.Tensor,
        cat_feats: torch.Tensor,
        scales: torch.Tensor,
        n_samples: int = 200,
        prediction_length: int = 28
    ) -> torch.Tensor:
        """
        Vectorized Autoregressive Monte Carlo Rollout.
        
        Guarantees zero validation target leakage by conditioning ONLY on the
        training context, then rolling out S sample paths for prediction_length steps.
        
        Returns:
            samples: Tensor of shape (batch, n_samples, prediction_length) unscaled and clipped >= 0.0
        """
        self.eval()
        device = context_target.device
        batch_size = context_target.shape[0]
        context_len = context_target.shape[1]

        # 1. Warm up the recurrent state using the historical training context
        lagged_ctx = torch.zeros_like(context_target)
        lagged_ctx[:, 1:] = context_target[:, :-1]
        lagged_ctx = lagged_ctx.unsqueeze(-1)

        static_embed = self._get_static_embeddings(cat_feats)
        static_ctx = static_embed.unsqueeze(1).expand(-1, context_len, -1)
        x_ctx = torch.cat([lagged_ctx, static_ctx, context_dyn], dim=-1)

        _, (h_T, c_T) = self.lstm(x_ctx)

        # 2. Expand states and inputs across n_samples for parallel vectorized rollout
        # Repeat interleave: [item_1, item_1, ..., item_B, item_B]
        total_paths = batch_size * n_samples
        h_curr = h_T.repeat_interleave(n_samples, dim=1)
        c_curr = c_T.repeat_interleave(n_samples, dim=1)

        static_rep = static_embed.repeat_interleave(n_samples, dim=0)
        future_dyn_rep = future_dyn.repeat_interleave(n_samples, dim=0)
        scales_rep = scales.repeat_interleave(n_samples, dim=0)

        # Initial autoregressive input: the final observed context demand
        y_prev = context_target[:, -1:].repeat_interleave(n_samples, dim=0)

        all_samples = torch.zeros(total_paths, prediction_length, device=device)

        # 3. Step-by-step autoregressive rollout
        for step in range(prediction_length):
            x_future_step = future_dyn_rep[:, step, :]
            z_step = torch.cat([y_prev, static_rep, x_future_step], dim=-1).unsqueeze(1)

            lstm_out, (h_curr, c_curr) = self.lstm(z_step, (h_curr, c_curr))
            out_squeeze = lstm_out.squeeze(1)

            if self.distribution == "gaussian":
                mu = self.fc_mu(out_squeeze)
                sigma = torch.nn.functional.softplus(self.fc_sigma(out_squeeze)) + 1e-4
                eps = torch.randn_like(mu)
                y_sampled_scaled = mu + sigma * eps
            else:
                mu = torch.nn.functional.softplus(self.fc_mu(out_squeeze)) + 1e-2
                alpha = torch.nn.functional.softplus(self.fc_alpha(out_squeeze)) + 1e-2
                r = 1.0 / alpha
                p = r / (r + mu)
                nb_dist = torch.distributions.NegativeBinomial(total_count=r, probs=1.0 - p)
                y_sampled_scaled = nb_dist.sample()

            # Unscale to physical inventory units
            y_unscaled = y_sampled_scaled * scales_rep

            # Enforce physical non-negative retail demand constraint: max(0.0, y)
            y_clipped = torch.clamp(y_unscaled, min=0.0)

            all_samples[:, step] = y_clipped.squeeze(-1)

            # Recursive feedback: prepare scaled prediction for next step input
            y_prev = (y_clipped / scales_rep)

        # Reshape to (batch_size, n_samples, prediction_length)
        return all_samples.view(batch_size, n_samples, prediction_length)


class GlobalDeepARForecaster:
    """
    High-level, production-ready Global DeepAR Forecaster for M5 Demand.
    
    Compatible with the existing AML baseline and Kalman pipelines.
    Provides scikit-learn style fit() and predict() interfaces.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 15,
        context_length: int = 56,
        prediction_length: int = 28,
        stride: int = 14,
        n_samples: int = 200,
        distribution: str = "gaussian",
        device: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.n_samples = n_samples
        self.distribution = distribution
        self.random_seed = random_seed

        # Device selection: MPS on Apple Silicon, CUDA on GPU, CPU fallback
        if device is None or device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.processor = DeepARDataProcessor(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            stride=self.stride
        )
        self.model: Optional[GlobalDeepARNet] = None
        self.training_losses: List[float] = []

    def _set_seeds(self):
        """Sets random seeds for reproducibility."""
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    def fit(self, train_df: pd.DataFrame) -> "GlobalDeepARForecaster":
        """
        Trains the global DeepAR model strictly on historical training observations.
        """
        start_time = time.time()
        self._set_seeds()
        logger.info("=" * 70)
        logger.info(f"  GLOBAL DEEPAR MODEL TRAINING (Device: {self.device})")
        logger.info("=" * 70)
        logger.info(
            f"Config: hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, "
            f"epochs={self.epochs}, batch_size={self.batch_size}, lr={self.learning_rate}, "
            f"context_length={self.context_length}, horizon={self.prediction_length}, "
            f"distribution={self.distribution}"
        )

        # 1. Prepare Dataset
        gen = torch.Generator().manual_seed(self.random_seed)
        train_loader = self.processor.prepare_training_dataset(
            train_df,
            batch_size=self.batch_size,
            shuffle=True,
            generator=gen
        )

        # 2. Build Model
        self.model = GlobalDeepARNet(
            cat_cardinalities=self.processor.cat_cardinalities,
            n_dynamic_features=len(self.processor.dynamic_feature_cols),
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            distribution=self.distribution
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        # 3. Training Loop
        self.training_losses = []
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for batch_target, batch_dyn, batch_cat, _ in train_loader:
                batch_target = batch_target.to(self.device)
                batch_dyn = batch_dyn.to(self.device)
                batch_cat = batch_cat.to(self.device)

                optimizer.zero_grad()
                dist_params, _ = self.model(batch_target, batch_dyn, batch_cat)
                loss = self.model.compute_loss(batch_target, dist_params)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_epoch_loss = epoch_loss / max(1, n_batches)
            self.training_losses.append(avg_epoch_loss)
            scheduler.step(avg_epoch_loss)

            if epoch % 5 == 0 or epoch == 1 or epoch == self.epochs:
                logger.info(
                    f"[Epoch {epoch:02d}/{self.epochs:02d}] NLL Loss: {avg_epoch_loss:.4f} "
                    f"(lr: {optimizer.param_groups[0]['lr']:.6f})"
                )

        train_dur = time.time() - start_time
        logger.info(f"Training completed in {train_dur:.2f} seconds. Final NLL: {self.training_losses[-1]:.4f}")
        return self

    @torch.no_grad()
    def predict(
        self,
        val_df: pd.DataFrame,
        train_df: pd.DataFrame,
        n_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generates 28-day out-of-sample probabilistic forecasts for each series.
        
        Zero Lookahead Guarantee:
        Validation sales are NEVER provided to the forecaster.
        Predictions are generated strictly via autoregressive Monte Carlo rollout.
        
        Returns:
            Dictionary containing:
            - point_forecast: P50 median point prediction
            - mean_forecast: Sample path mean prediction
            - p10: 10th percentile
            - p50: 50th percentile
            - p90: 90th percentile
            - samples_by_series: Dict mapping series_id to (n_samples, 28) numpy array
            - predictions_by_series: Dict mapping series_id to quantile series
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        # Set seed for reproducible Monte Carlo sampling
        self._set_seeds()

        samples_count = n_samples or self.n_samples
        logger.info(
            f"Generating out-of-sample forecasts for {self.prediction_length} days "
            f"using {samples_count} Monte Carlo paths per series..."
        )

        # Prepare inputs
        inputs = self.processor.prepare_inference_inputs(train_df, val_df)
        ctx_target = inputs["context_target"].to(self.device)
        ctx_dyn = inputs["context_dyn"].to(self.device)
        fut_dyn = inputs["future_dyn"].to(self.device)
        cat_feats = inputs["cat_feats"].to(self.device)
        scales = inputs["scales"].to(self.device)
        series_ids = inputs["series_ids"]

        # Run vectorized rollout
        samples_tensor = self.model.sample_rollout(
            context_target=ctx_target,
            context_dyn=ctx_dyn,
            future_dyn=fut_dyn,
            cat_feats=cat_feats,
            scales=scales,
            n_samples=samples_count,
            prediction_length=self.prediction_length
        )  # (batch, n_samples, horizon)

        samples_np = samples_tensor.cpu().numpy()

        p10_matrix = np.percentile(samples_np, 10, axis=1)  # (batch, horizon)
        p50_matrix = np.percentile(samples_np, 50, axis=1)  # (batch, horizon)
        p90_matrix = np.percentile(samples_np, 90, axis=1)  # (batch, horizon)
        mean_matrix = np.mean(samples_np, axis=1)           # (batch, horizon)

        # Map predictions to series_id
        pred_map = {}
        samples_map = {}
        for idx, sid in enumerate(series_ids):
            pred_map[sid] = {
                "point": p50_matrix[idx],
                "mean": mean_matrix[idx],
                "p10": p10_matrix[idx],
                "p50": p50_matrix[idx],
                "p90": p90_matrix[idx],
                "lower_80": p10_matrix[idx],
                "upper_80": p90_matrix[idx],
            }
            samples_map[sid] = samples_np[idx]

        return {
            "predictions_by_series": pred_map,
            "samples_by_series": samples_map,
            "series_ids": series_ids,
        }
