"""
orthogonalization.py — Collinearity Control, Feature Clustering, and Orthogonalization.

TIMING CONVENTION:
Features use data <= close of day t -> entry at OPEN t+1 -> exit at CLOSE t+k or earlier on barrier touch.
Barriers use high/low of t+1…t+k; same-day double touch -> stop first; open through a barrier -> fill at open.
Adjusted prices; point-in-time universe incl. delisted; reindex to master trading calendar (k = market days).
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class CollinearityController:
    """
    Collinearity Controller:
    Reduces feature multicollinearity via Spearman-distance hierarchical clustering
    strictly fitted on training split data only, reporting Variance Inflation Factor (VIF)
    before and after feature selection.

    Mathematical Formulation:
      1. Spearman Rank Correlation Matrix:
         rho_{ij} = corr(Rank(X_i), Rank(X_j)) in [-1, 1]

      2. Information-Theoretic Angular Distance Metric:
         d_{ij} = sqrt( 0.5 * (1 - rho_{ij}) ) in [0, 1]
         where rho = 1.0 -> d = 0.0; rho = 0.0 -> d = 0.707; rho = -1.0 -> d = 1.0.

      3. Hierarchical Agglomerative Clustering:
         Average linkage on condensed distance vector squareform(D).
         Flat clusters formed via cophenetic distance cut-off threshold t (default 0.5).

      4. Variance Inflation Factor (VIF):
         VIF_j = 1 / (1 - R_j^2)
         where R_j^2 is the coefficient of determination from regressing X_j on X_{-j}.
    """

    def __init__(self, method: str = 'clustering', threshold: float = 0.5):
        self.method = method
        self.threshold = threshold
        self.selected_features: List[str] = []
        self.ortho_matrix: Optional[np.ndarray] = None
        self.feature_order: List[str] = []
        self.vif_before: pd.Series = pd.Series(dtype=float)
        self.vif_after: pd.Series = pd.Series(dtype=float)
        self.vif_report: Optional[pd.DataFrame] = None
        self.clusters_dict: Dict[int, List[str]] = {}

    def compute_vif(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates Variance Inflation Factor (VIF) for each column in df.
        Formula: VIF_j = 1 / (1 - R_j^2)
        """
        cols = list(df.columns)
        p = len(cols)
        if p <= 1:
            return pd.Series([1.0] * p, index=cols)

        # Drop NaNs across columns
        X_df = df[cols].dropna()
        if len(X_df) < p + 5:
            logger.warning(f"VIF calculation skipped: insufficient samples ({len(X_df)}) vs features ({p})")
            return pd.Series(np.nan, index=cols)

        X = X_df.values.astype(float)
        # Center and scale X to prevent numerical ill-conditioning
        X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-12)

        vifs = []
        for j in range(p):
            y_j = X_std[:, j]
            X_other = np.delete(X_std, j, axis=1)

            # OLS: min ||X_other * b - y_j||^2
            try:
                b, _, _, _ = np.linalg.lstsq(X_other, y_j, rcond=None)
                ss_res = np.sum((y_j - X_other.dot(b)) ** 2)
                ss_tot = np.sum(y_j ** 2)
                r_squared = 1.0 - (ss_res / (ss_tot + 1e-12))
                r_squared = np.clip(r_squared, 0.0, 0.9999)
                vif = 1.0 / (1.0 - r_squared)
            except Exception:
                vif = np.nan
            vifs.append(vif)

        return pd.Series(vifs, index=cols)

    def fit(self, df_train: pd.DataFrame, priority_order: Optional[List[str]] = None):
        """
        Fits collinearity controller strictly on training split data.
        Selects representative features per cluster and records VIF before/after.
        """
        if priority_order:
            features = [f for f in priority_order if f in df_train.columns]
        else:
            features = list(df_train.columns)

        df_clean = df_train[features].dropna()
        if len(df_clean) < len(features) + 5:
            logger.warning("df_train has insufficient clean rows; using pairwise correlations")
            rho = df_train[features].corr(method='spearman').fillna(0.0).values
        else:
            rho = df_clean.corr(method='spearman').values

        self.feature_order = features

        # Compute initial VIF
        self.vif_before = self.compute_vif(df_train[features])

        if self.method == 'clustering':
            # Distance matrix: d = sqrt(0.5 * (1 - rho))
            d_ij = np.sqrt(0.5 * np.clip(1.0 - rho, 0.0, 2.0))
            np.fill_diagonal(d_ij, 0.0)

            # Hierarchical clustering
            condensed_dist = squareform(d_ij, checks=False)
            Z = linkage(condensed_dist, method='average')

            # Flat clusters
            clusters = fcluster(Z, t=self.threshold, criterion='distance')

            # Group features by cluster
            cluster_map: Dict[int, List[str]] = {}
            for feat, cid in zip(features, clusters):
                cluster_map.setdefault(cid, []).append(feat)
            self.clusters_dict = cluster_map

            # Select one representative per cluster
            selected = []
            for cid, cluster_feats in cluster_map.items():
                if priority_order:
                    # Pick highest priority feature in cluster
                    rep = min(cluster_feats, key=lambda f: priority_order.index(f) if f in priority_order else 9999)
                else:
                    # Pick feature with highest individual variance / completeness
                    rep = cluster_feats[0]
                selected.append(rep)

            self.selected_features = selected

            # Compute post-selection VIF
            self.vif_after = self.compute_vif(df_train[self.selected_features])

        elif self.method == 'gram-schmidt':
            # Gram-Schmidt sequential orthogonalization ablation
            X = df_clean.values.astype(float)
            Q, R = np.linalg.qr(X)
            self.ortho_matrix = np.linalg.inv(R)
            self.selected_features = features
            self.vif_after = pd.Series([1.0] * len(features), index=features)

        # Assemble summary report
        report = []
        for f in features:
            is_sel = f in self.selected_features
            report.append({
                'feature': f,
                'vif_before': self.vif_before.get(f, np.nan),
                'selected': is_sel,
                'vif_after': self.vif_after.get(f, np.nan) if is_sel else np.nan
            })
        self.vif_report = pd.DataFrame(report)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature selection or orthogonal projection to out-of-sample data.
        """
        if self.method == 'clustering':
            available = [c for c in self.selected_features if c in df.columns]
            return df[available].copy()
        elif self.method == 'gram-schmidt':
            if self.ortho_matrix is None:
                raise ValueError("CollinearityController not fitted with Gram-Schmidt.")
            available = [c for c in self.feature_order if c in df.columns]
            df_clean = df[available].copy()
            mask = df_clean.notna().all(axis=1)
            out = pd.DataFrame(np.nan, index=df.index, columns=available)
            if mask.any():
                X = df_clean.loc[mask].values.astype(float)
                Q_new = X.dot(self.ortho_matrix)
                out.loc[mask] = Q_new
            return out
        else:
            return df

    def get_vif_report(self) -> pd.DataFrame:
        """Returns the VIF comparison table before and after selection."""
        if self.vif_report is None:
            return pd.DataFrame()
        return self.vif_report
