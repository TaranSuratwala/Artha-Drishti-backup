"""
multi_horizon_models.py — LightGBM & Multi-Task Neural Network Models with Uncertainty Quantification.

TIMING CONVENTION:
Features use data <= close of day t -> entry at OPEN t+1 -> exit at CLOSE t+k or earlier on barrier touch.
Barriers use high/low of t+1…t+k; same-day double touch -> stop first; open through a barrier -> fill at open.
Adjusted prices; point-in-time universe incl. delisted; reindex to master trading calendar (k = market days).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple, Union
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ====================================================================
# 1. LIGHTGBM MULTI-HORIZON BASELINE MODELS
# ====================================================================

class LightGBMMultiHorizon:
    """
    LightGBM Multi-Horizon Baseline Model:
    Trains dedicated models per horizon k in {3, 5, 7, 15, 21}.

    Supports two primary objectives:
      1. 'regression': Regresses on continuous Rank-Gauss excess return targets,
         minimizing RMSE (surrogate for Rank IC), weighted by average-uniqueness weights.
      2. 'lambdarank': Optimizes pairwise / listwise ranking on discrete quantile relevance bins
         grouped by trading date cross-section queries (NDCG@20).
    """

    def __init__(
        self,
        horizons: List[int] = [3, 5, 7, 15, 21],
        mode: str = 'regression',
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        min_child_samples: int = 50,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
        num_boost_round: int = 150
    ):
        self.horizons = horizons
        self.mode = mode
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.num_boost_round = num_boost_round
        self.models: Dict[int, lgb.Booster] = {}
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_dict: Dict[int, pd.Series],
        dates_train: Optional[pd.Series] = None,
        weights_dict: Optional[Dict[int, pd.Series]] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val_dict: Optional[Dict[int, pd.Series]] = None
    ) -> 'LightGBMMultiHorizon':
        """
        Fits LightGBM models across all specified horizons.
        """
        self.feature_names = list(X_train.columns)

        for k in self.horizons:
            if k not in y_train_dict:
                logger.warning(f"Target for horizon k={k} not found; skipping.")
                continue

            y_tr = y_train_dict[k]
            w_tr = weights_dict.get(k, None) if weights_dict is not None else None

            # Align valid rows
            valid_mask = y_tr.notna() & X_train.notna().all(axis=1)
            X_clean = X_train.loc[valid_mask]
            y_clean = y_tr.loc[valid_mask]
            w_clean = w_tr.loc[valid_mask].values if w_tr is not None else None

            if len(X_clean) < 100:
                logger.warning(f"Insufficient training samples ({len(X_clean)}) for horizon k={k}")
                continue

            if self.mode == 'regression':
                ds_train = lgb.Dataset(X_clean, label=y_clean, weight=w_clean, free_raw_data=False)
                valid_sets = [ds_train]
                valid_names = ['train']

                if X_val is not None and y_val_dict is not None and k in y_val_dict:
                    val_mask = y_val_dict[k].notna() & X_val.notna().all(axis=1)
                    if val_mask.any():
                        ds_val = lgb.Dataset(X_val.loc[val_mask], label=y_val_dict[k].loc[val_mask], free_raw_data=False)
                        valid_sets.append(ds_val)
                        valid_names.append('val')

                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'learning_rate': self.learning_rate,
                    'num_leaves': self.num_leaves,
                    'min_child_samples': self.min_child_samples,
                    'subsample': self.subsample,
                    'colsample_bytree': self.colsample_bytree,
                    'verbose': -1,
                    'random_state': 42
                }

                booster = lgb.train(
                    params,
                    ds_train,
                    num_boost_round=self.num_boost_round,
                    valid_sets=valid_sets,
                    valid_names=valid_names
                )

            elif self.mode == 'lambdarank':
                if dates_train is None:
                    raise ValueError("LambdaRank mode requires dates_train to construct query groups.")

                d_clean = dates_train.loc[valid_mask]
                # Sort by date for group query alignment
                sort_idx = np.argsort(d_clean.values)
                X_clean_sorted = X_clean.iloc[sort_idx]
                y_clean_sorted = y_clean.iloc[sort_idx].astype(int)
                d_clean_sorted = d_clean.iloc[sort_idx]

                # Group size per date
                group_sizes = d_clean_sorted.value_counts(sort=False).values

                ds_train = lgb.Dataset(X_clean_sorted, label=y_clean_sorted, group=group_sizes, free_raw_data=False)
                params = {
                    'objective': 'lambdarank',
                    'metric': 'ndcg',
                    'ndcg_eval_at': [10, 20],
                    'learning_rate': self.learning_rate,
                    'num_leaves': self.num_leaves,
                    'min_child_samples': self.min_child_samples,
                    'verbose': -1,
                    'random_state': 42
                }

                booster = lgb.train(params, ds_train, num_boost_round=self.num_boost_round)

            self.models[k] = booster

        return self

    def predict(self, X: pd.DataFrame, k: int) -> np.ndarray:
        """Generates raw model predictions for horizon k."""
        if k not in self.models:
            raise KeyError(f"Model for horizon k={k} not fitted.")
        return self.models[k].predict(X[self.feature_names])

    def predict_all_horizons(self, X: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Generates predictions across all fitted horizons."""
        return {k: self.predict(X, k) for k in self.models}


# ====================================================================
# 2. PYTORCH LOSS FUNCTIONS: STUDENT-T NLL & DIFFERENTIABLE IC
# ====================================================================

class StudentTNLLLoss(nn.Module):
    """
    Student-t Negative Log-Likelihood Loss with NaN-masked support:
    L_NLL = -ln Gamma((nu+1)/2) + ln Gamma(nu/2) + 0.5*ln(pi*nu) + ln(sigma) + ((nu+1)/2)*ln(1 + (y - mu)^2 / (nu * sigma^2))
    Handles fat-tailed non-Gaussian equity return distributions.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        df: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Enforce positive scale and df > 2 (finite variance)
        scale_pos = F.softplus(scale) + 1e-6
        df_pos = F.softplus(df) + 2.1
        var = scale_pos ** 2

        log_gamma_diff = torch.special.gammaln((df_pos + 1.0) / 2.0) - torch.special.gammaln(df_pos / 2.0)
        term1 = log_gamma_diff - 0.5 * torch.log(df_pos * math.pi) - torch.log(scale_pos)
        term2 = -((df_pos + 1.0) / 2.0) * torch.log1p(((target - loc) ** 2) / (df_pos * var))

        nll = -(term1 + term2)

        if mask is not None:
            valid_nll = nll[mask]
            if len(valid_nll) == 0:
                return torch.tensor(0.0, device=loc.device, requires_grad=True)
            return valid_nll.mean()
        else:
            return nll.mean()


class DifferentiableICLoss(nn.Module):
    """
    Differentiable Pearson / Rank IC Loss:
    Minimizes negative correlation: - Cov(y_hat, y) / (sigma_{y_hat} * sigma_y).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        if len(pred) < 3:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_c = pred - pred.mean()
        target_c = target - target.mean()

        cov = torch.sum(pred_c * target_c)
        denom = torch.sqrt(torch.sum(pred_c ** 2) * torch.sum(target_c ** 2)) + 1e-8
        ic = cov / denom
        return -ic


# ====================================================================
# 3. MULTI-TASK NEURAL NETWORK (TCN/MLP + STUDENT-T HEADS)
# ====================================================================

class MultiTaskNN(nn.Module):
    """
    Multi-Task Neural Network Architecture:
      - Shared Trunk: LayerNorm + Residual MLP with Dropout for joint representation learning.
      - Multi-Horizon Heads: For each k in {3, 5, 7, 15, 21}, predicts (loc, scale, df).
      - Learnable Task Weighting: Kendall homoscedastic uncertainty log-variance parameters.
    """

    def __init__(
        self,
        input_dim: int,
        horizons: List[int] = [3, 5, 7, 15, 21],
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.horizons = horizons
        self.num_tasks = len(horizons)

        # Shared Trunk with LayerNorm and Residual Connection
        self.norm0 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act3 = nn.GELU()

        # Dedicated Heads per Horizon: Output (loc, scale, df)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 3) for _ in range(self.num_tasks)
        ])

        # Kendall Uncertainty Parameters: log(sigma_k^2)
        self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))

    def forward(
        self,
        x: torch.Tensor,
        mc_dropout: bool = False
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        If mc_dropout=True, dropout layers remain active during inference for epistemic uncertainty.
        """
        if mc_dropout:
            self.train()

        h = self.norm0(x)
        h = self.drop1(self.act1(self.fc1(h)))
        # Residual connection
        h = h + self.drop2(self.act2(self.fc2(h)))
        features = self.act3(self.fc3(h))

        outputs = []
        for head in self.heads:
            out = head(features)
            loc = out[:, 0]
            scale = out[:, 1]
            df = out[:, 2]
            outputs.append((loc, scale, df))

        return outputs

    def fit_model(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_dict: Dict[int, Union[np.ndarray, pd.Series]],
        epochs: int = 12,
        lr: float = 1e-3,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        lambda_ic: float = 0.5,
        device: Optional[Union[str, torch.device]] = None
    ) -> 'MultiTaskNN':
        """
        Fits MultiTaskNN on multi-horizon targets using joint Student-t NLL loss
        and Differentiable IC loss with Kendall homoscedastic uncertainty weighting.
        """
        dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(dev, str):
            dev = torch.device(dev)
        self.to(dev)
        self.train()

        if isinstance(X, pd.DataFrame):
            X_mat = X.values.astype(np.float32)
        else:
            X_mat = np.asarray(X, dtype=np.float32)

        # Clean NaNs or Infs
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        X_tensor = torch.tensor(X_mat, dtype=torch.float32, device=dev)

        y_dict_tensors: Dict[int, torch.Tensor] = {}
        for k in self.horizons:
            if k in y_dict:
                y_val = y_dict[k].values if isinstance(y_dict[k], pd.Series) else y_dict[k]
                y_arr = np.asarray(y_val, dtype=np.float32)
                y_dict_tensors[k] = torch.tensor(y_arr, dtype=torch.float32, device=dev)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        nll_crit = StudentTNLLLoss()
        ic_crit = DifferentiableICLoss()

        n_samples = len(X_mat)
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]

                if len(batch_idx) < 8:
                    continue

                x_batch = X_tensor[batch_idx]
                optimizer.zero_grad()
                outputs = self(x_batch)

                batch_loss = torch.tensor(0.0, device=dev, requires_grad=True)
                for t_idx, k in enumerate(self.horizons):
                    if k not in y_dict_tensors:
                        continue

                    y_batch = y_dict_tensors[k][batch_idx]
                    loc, scale, df = outputs[t_idx]

                    mask = torch.isfinite(y_batch)
                    if mask.sum() < 4:
                        continue

                    loss_nll = nll_crit(loc, scale, df, y_batch, mask=mask)
                    loss_ic = ic_crit(loc, y_batch, mask=mask)
                    task_loss = loss_nll + lambda_ic * loss_ic

                    # Kendall homoscedastic uncertainty weighting: 0.5 * exp(-s) * L + 0.5 * s
                    s = self.log_vars[t_idx]
                    weighted_task_loss = 0.5 * torch.exp(-s) * task_loss + 0.5 * s
                    batch_loss = batch_loss + weighted_task_loss

                if batch_loss.requires_grad and batch_loss.item() != 0.0:
                    batch_loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
                    optimizer.step()
                    epoch_loss += batch_loss.item()
                    n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.debug(f"MultiTaskNN Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        self.eval()
        return self

    def predict_distribution(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        device: Optional[Union[str, torch.device]] = None
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Generates distribution parameters (mu, scale, df, var, std) for each horizon.
        """
        dev = device or next(self.parameters()).device
        if isinstance(dev, str):
            dev = torch.device(dev)

        self.eval()
        if isinstance(X, pd.DataFrame):
            X_mat = X.values.astype(np.float32)
        else:
            X_mat = np.asarray(X, dtype=np.float32)

        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        X_tensor = torch.tensor(X_mat, dtype=torch.float32, device=dev)

        results: Dict[int, Dict[str, np.ndarray]] = {}
        with torch.no_grad():
            outputs = self(X_tensor, mc_dropout=False)
            for t_idx, k in enumerate(self.horizons):
                loc, scale, df = outputs[t_idx]
                scale_pos = F.softplus(scale) + 1e-6
                df_pos = F.softplus(df) + 2.1
                var = (scale_pos ** 2) * (df_pos / (df_pos - 2.0))

                results[k] = {
                    "mu": loc.cpu().numpy(),
                    "scale": scale_pos.cpu().numpy(),
                    "df": df_pos.cpu().numpy(),
                    "var": var.cpu().numpy(),
                    "std": torch.sqrt(var).cpu().numpy(),
                }

        return results

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        k: int,
        device: Optional[Union[str, torch.device]] = None
    ) -> np.ndarray:
        """Returns point expectation mu for horizon k."""
        dist = self.predict_distribution(X, device=device)
        if k not in dist:
            raise KeyError(f"Horizon {k} not found in model horizons {self.horizons}")
        return dist[k]["mu"]



# ====================================================================
# 4. TRAINING HARNESS & DATE-WISE BATCHING
# ====================================================================

class MultiTaskNNTrainer:
    """
    Multi-Task Neural Network Training Harness:
    Supports date-wise cross-sectional mini-batches, multi-task Kendall loss,
    early stopping on validation Rank IC, and MC-Dropout uncertainty estimation.
    """

    def __init__(
        self,
        model: MultiTaskNN,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_ic: float = 0.5,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.lambda_ic = lambda_ic

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.nll_crit = StudentTNLLLoss()
        self.ic_crit = DifferentiableICLoss()

    def train_epoch(
        self,
        X_tensor: torch.Tensor,
        y_dict_tensors: Dict[int, torch.Tensor],
        date_indices: List[np.ndarray]
    ) -> float:
        """
        Trains for one epoch across date-wise batches.
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx in date_indices:
            if len(batch_idx) < 5:
                continue

            x_batch = X_tensor[batch_idx].to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(x_batch)

            batch_loss = 0.0
            for t_idx, k in enumerate(self.model.horizons):
                if k not in y_dict_tensors:
                    continue

                y_batch = y_dict_tensors[k][batch_idx].to(self.device)
                loc, scale, df = outputs[t_idx]

                mask = torch.isfinite(y_batch)
                if mask.sum() < 3:
                    continue

                loss_nll = self.nll_crit(loc, scale, df, y_batch, mask=mask)
                loss_ic = self.ic_crit(loc, y_batch, mask=mask)
                task_loss = loss_nll + self.lambda_ic * loss_ic

                # Kendall homoscedastic uncertainty weighting: 0.5 * exp(-s) * L + 0.5 * s
                s = self.model.log_vars[t_idx]
                weighted_task_loss = 0.5 * torch.exp(-s) * task_loss + 0.5 * s
                batch_loss = batch_loss + weighted_task_loss

            if isinstance(batch_loss, torch.Tensor) and batch_loss.requires_grad:
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
                total_loss += batch_loss.item()

        return total_loss / max(len(date_indices), 1)

    def predict_with_uncertainty(
        self,
        X_tensor: torch.Tensor,
        n_mc_samples: int = 20
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Inference with decomposition of Aleatoric and Epistemic uncertainty:
          - Epistemic: Variance of MC-Dropout predictions across n_mc_samples passes.
          - Aleatoric: Predicted variance from Student-t head: sigma^2 * (nu / (nu - 2)).
        """
        self.model.eval()
        x = X_tensor.to(self.device)

        mc_predictions = {k: [] for k in self.model.horizons}
        aleatoric_vars = {k: [] for k in self.model.horizons}

        with torch.no_grad():
            for _ in range(n_mc_samples):
                outputs = self.model(x, mc_dropout=True)
                for t_idx, k in enumerate(self.model.horizons):
                    loc, scale, df = outputs[t_idx]
                    scale_pos = F.softplus(scale) + 1e-6
                    df_pos = F.softplus(df) + 2.1
                    var_aleatoric = (scale_pos ** 2) * (df_pos / (df_pos - 2.0))

                    mc_predictions[k].append(loc.cpu().numpy())
                    aleatoric_vars[k].append(var_aleatoric.cpu().numpy())

        results = {}
        for k in self.model.horizons:
            preds_arr = np.array(mc_predictions[k])  # [mc, N]
            mean_pred = np.mean(preds_arr, axis=0)
            var_epistemic = np.var(preds_arr, axis=0)
            var_aleatoric = np.mean(np.array(aleatoric_vars[k]), axis=0)
            total_std = np.sqrt(var_epistemic + var_aleatoric)

            results[k] = {
                'pred': mean_pred,
                'std_epistemic': np.sqrt(var_epistemic),
                'std_aleatoric': np.sqrt(var_aleatoric),
                'std_total': total_std
            }

        return results


# ====================================================================
# 5. STACKED CROSS-HORIZON SIGNAL
# ====================================================================

class CrossHorizonSignalStacker:
    """
    Stacked Cross-Horizon Signal Generator:
    Combines predictions across multiple horizons k in {3, 5, 7, 15, 21} into
    a composite cross-sectional alpha score.
      S_stacked = sum_k w_k * Rank(y_hat_k)
    where w_k are proportional to out-of-sample ICIR or inverse volatility.
    """

    def __init__(self, horizons: List[int] = [3, 5, 7, 15, 21], weights: Optional[Dict[int, float]] = None):
        self.horizons = horizons
        if weights is not None:
            total_w = sum(weights.values())
            self.weights = {k: weights[k] / total_w for k in horizons if k in weights}
        else:
            # Equal weighting default
            self.weights = {k: 1.0 / len(horizons) for k in horizons}

    def fit_icir_weights(self, oos_icirs: Dict[int, float]) -> 'CrossHorizonSignalStacker':
        """Calibrates weights proportionally to positive out-of-sample ICIR."""
        pos_icirs = {k: max(oos_icirs.get(k, 0.0), 0.0) for k in self.horizons}
        total = sum(pos_icirs.values())
        if total > 0:
            self.weights = {k: pos_icirs[k] / total for k in self.horizons}
        else:
            self.weights = {k: 1.0 / len(self.horizons) for k in self.horizons}
        return self

    def stack(
        self,
        predictions_dict: Dict[int, Union[pd.Series, np.ndarray]],
        index: Optional[pd.Index] = None
    ) -> pd.Series:
        """
        Stacks horizon scores into a composite rank score.
        """
        stacked = None
        for k, w in self.weights.items():
            if k not in predictions_dict:
                continue

            pred = predictions_dict[k]
            if isinstance(pred, pd.Series):
                rank = pred.rank(pct=True)
            else:
                pred_s = pd.Series(pred, index=index)
                rank = pred_s.rank(pct=True)

            if stacked is None:
                stacked = w * rank
            else:
                stacked = stacked + w * rank

        return stacked


# ====================================================================
# 6. UNCERTAINTY QUANTIFICATION & CONFIDENCE FILTER
# ====================================================================

class UncertaintyFilter:
    """
    Uncertainty Filter & Coverage Evaluator:
    Filters predictions based on total predictive uncertainty (aleatoric + epistemic)
    and evaluates the coverage vs Rank IC performance curve.
    """

    def __init__(self, coverage_quantile: float = 0.75):
        self.coverage_quantile = coverage_quantile

    def filter_high_confidence(
        self,
        scores: np.ndarray,
        uncertainties: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selects only the top coverage_quantile predictions with lowest uncertainty.
        Returns: (filtered_scores, boolean_selection_mask)
        """
        threshold = np.quantile(uncertainties, self.coverage_quantile)
        mask = uncertainties <= threshold
        filtered = np.where(mask, scores, np.nan)
        return filtered, mask

    def evaluate_coverage_curve(
        self,
        scores: pd.Series,
        uncertainties: pd.Series,
        targets: pd.Series,
        quantiles: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    ) -> pd.DataFrame:
        """
        Evaluates Rank IC across varying confidence coverage thresholds.
        Verifies whether pruning uncertain predictions monotonically enhances Rank IC.
        """
        records = []
        for q in quantiles:
            thr = uncertainties.quantile(q)
            mask = uncertainties <= thr
            n_obs = mask.sum()
            if n_obs >= 20:
                ic = scores[mask].rank().corr(targets[mask].rank())
            else:
                ic = np.nan

            records.append({
                'coverage_pct': q * 100.0,
                'uncertainty_threshold': thr,
                'sample_count': n_obs,
                'rank_ic': ic
            })

        return pd.DataFrame(records)
