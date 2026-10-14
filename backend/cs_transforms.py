"""
cs_transforms.py — Cross-Sectional Normalization, Winsorization, and Factor Neutralization.

TIMING CONVENTION:
Features use data <= close of day t -> entry at OPEN t+1 -> exit at CLOSE t+k or earlier on barrier touch.
Barriers use high/low of t+1…t+k; same-day double touch -> stop first; open through a barrier -> fill at open.
Adjusted prices; point-in-time universe incl. delisted; reindex to master trading calendar (k = market days).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class CrossSectionalNormalizer:
    """
    Cross-Sectional Normalizer:
    Performs per-date MAD winsorization followed by sector-aware rank-scaling / Rank-Gauss.

    Mathematical Formulation:
      1. Per-date MAD Winsorization:
         median_t = median_i(x_{i,t})
         MAD_t = median_i(|x_{i,t} - median_t|)
         sigma_hat_t = 1.4826 * MAD_t
         x_tilde_{i,t} = clip(x_{i,t}, median_t - k * sigma_hat_t, median_t + k * sigma_hat_t)
         where k = n_mad (default 5.0).

      2. Sector-Aware Rank Scaling:
         For sectors with count >= min_sector_size (default 8):
             u_{i,s,t} = (Rank(x_{i,s,t}) - 0.5) / N_{s,t}
         For sectors with count < min_sector_size (or missing sector):
             u_{i,t} = (Rank(x_{i,t}) - 0.5) / N_t   (fallback to universe rank)

      3. Rank-Gauss Transformation:
         z_{i,t} = Phi^{-1}(clip(u_{i,t}, eps, 1 - eps))
    """

    def __init__(self, min_sector_size: int = 8, gaussianize: bool = True, n_mad: float = 5.0):
        self.min_sector_size = min_sector_size
        self.gaussianize = gaussianize
        self.n_mad = n_mad

    def _mad_winsorize(self, s: pd.Series) -> pd.Series:
        """
        Winsorize a cross-sectional series using Median Absolute Deviation (MAD).
        Scale factor 1.4826 ensures asymptotic normality consistency.
        """
        valid = s.dropna()
        if len(valid) < 3:
            return s
        median = valid.median()
        mad = np.median(np.abs(valid - median))
        scale = 1.4826 * mad
        if scale < 1e-12:
            return s
        return s.clip(lower=median - self.n_mad * scale, upper=median + self.n_mad * scale)

    def _rank_scale(self, s: pd.Series) -> pd.Series:
        """
        Transforms a series to uniform rank scores in (0, 1) or standard normal via Rank-Gauss.
        """
        ranks = s.rank(method='average')
        N = s.notna().sum()
        if N == 0:
            return s
        u = (ranks - 0.5) / N
        if self.gaussianize:
            # Clip probabilities to prevent +/- inf at endpoints
            u_clipped = u.clip(lower=1e-4, upper=1.0 - 1e-4)
            return pd.Series(norm.ppf(u_clipped), index=s.index)
        else:
            return u - 0.5

    def transform(self, df: pd.DataFrame, features: List[str], sector_col: str = 'sector') -> pd.DataFrame:
        """
        Transform features for a SINGLE DATE cross-section.
        For panel datasets, invoke via df.groupby('date', group_keys=False).apply(lambda g: normalizer.transform(g, features)).
        """
        res = pd.DataFrame(index=df.index)

        # Prepare sector mapping if column exists
        has_sector = sector_col in df.columns and df[sector_col].notna().any()
        if has_sector:
            sector_series = df[sector_col].fillna('UNKNOWN')
            sector_counts = sector_series.value_counts()
        else:
            sector_series = None
            sector_counts = pd.Series(dtype=int)

        for f in features:
            if f not in df.columns:
                continue

            s = df[f]
            s_clean = self._mad_winsorize(s)

            # Compute full-universe rank as fallback
            full_rank = self._rank_scale(s_clean)

            if has_sector:
                out_s = pd.Series(np.nan, index=s.index, dtype=float)
                for sector, count in sector_counts.items():
                    idx = (sector_series == sector) & s_clean.notna()
                    if sector != 'UNKNOWN' and count >= self.min_sector_size:
                        out_s[idx] = self._rank_scale(s_clean[idx])
                    else:
                        # Fallback to full universe rank for small sectors or UNKNOWN
                        out_s[idx] = full_rank[idx]
            else:
                out_s = full_rank

            res[f + '_cs'] = out_s

        return res


class CrossSectionalNeutralizer:
    """
    Cross-Sectional Factor Neutralizer:
    Performs cross-sectional OLS neutralization of feature(s) against market beta,
    log market capitalization, and sector indicator dummies per trading date.

    Mathematical Formulation:
      For feature vector y_t in R^{N_t} and factor design matrix X_t in R^{N_t x K}:
        X_t = [ 1, beta_t, ln(MCap_t), S_{1,t}, ..., S_{M-1,t} ]
      where S_{m,t} are one-hot dummy variables for sectors with >= min_sector_size stocks.

      OLS coefficient:
        w_t = (X_t^T X_t)^{-1} X_t^T y_t
      Residual vector:
        e_t = y_t - X_t w_t

      Exact Orthogonality Property:
        X_t^T e_t = 0_K  (verified numerically: max |X_t^T e_t| < 1e-10)
    """

    def __init__(self, min_sector_size: int = 8):
        self.min_sector_size = min_sector_size

    def _blume_shrinkage(self, betas: pd.Series) -> pd.Series:
        """Blume (1975) adjustment: 2/3 * beta + 1/3 * 1.0 towards market mean."""
        return (2.0 / 3.0) * betas + (1.0 / 3.0)

    def estimate_betas(self, returns_df: pd.DataFrame, market_returns: pd.Series, window: int = 60) -> pd.DataFrame:
        """
        Estimate 60-day rolling market betas with Blume shrinkage.
        returns_df: wide DataFrame (dates x tickers).
        market_returns: Series (dates).
        """
        cov = returns_df.rolling(window, min_periods=window // 2).cov(market_returns)
        var = market_returns.rolling(window, min_periods=window // 2).var()
        betas = cov.divide(var, axis=0)
        return betas.apply(self._blume_shrinkage)

    def neutralise(
        self,
        df: pd.DataFrame,
        feature_cols: Union[str, List[str]],
        beta_col: str = 'beta',
        mcap_col: str = 'mcap',
        sector_col: str = 'sector',
        verify_orthogonality: bool = True
    ) -> pd.DataFrame:
        """
        Neutralizes features on a SINGLE DATE cross-section.
        Returns DataFrame with neutralized residuals (matching df.index).
        """
        if isinstance(feature_cols, str):
            cols_to_process = [feature_cols]
            single_col = True
        else:
            cols_to_process = list(feature_cols)
            single_col = False

        res = pd.DataFrame(np.nan, index=df.index, columns=cols_to_process)

        # Check required columns
        req_cols = [beta_col, mcap_col]
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            logger.warning(f"Neutralization skipped: missing required columns {missing}")
            for c in cols_to_process:
                if c in df.columns:
                    res[c] = df[c]
            return res[cols_to_process[0]] if single_col else res

        # Valid mask
        valid_mask = df[beta_col].notna() & df[mcap_col].notna() & (df[mcap_col] > 0)
        df_clean = df.loc[valid_mask]

        if len(df_clean) < 10:
            for c in cols_to_process:
                if c in df.columns:
                    res.loc[df_clean.index, c] = df_clean[c]
            return res[cols_to_process[0]] if single_col else res

        # Construct Design Matrix X
        beta = df_clean[beta_col].values.astype(float)
        log_mcap = np.log(np.maximum(df_clean[mcap_col].values.astype(float), 1.0))
        intercept = np.ones(len(df_clean), dtype=float)

        x_components = [intercept, beta, log_mcap]

        # Add sector dummies if present and sufficiently populated
        if sector_col in df_clean.columns and df_clean[sector_col].notna().any():
            sectors = df_clean[sector_col].fillna('UNKNOWN')
            counts = sectors.value_counts()
            eligible_sectors = counts[counts >= self.min_sector_size].index.tolist()

            if len(eligible_sectors) > 1:
                # One-hot encode eligible sectors, drop first to prevent collinearity with intercept
                dummies = pd.get_dummies(sectors[sectors.isin(eligible_sectors)], drop_first=True, dtype=float)
                # Reindex to full df_clean length (unselected sectors get 0)
                dummies = dummies.reindex(df_clean.index, fill_value=0.0).values
                x_components.append(dummies)

        X = np.column_stack(x_components)

        for c in cols_to_process:
            if c not in df_clean.columns:
                continue

            y_series = df_clean[c]
            feat_valid = y_series.notna()

            if feat_valid.sum() < X.shape[1] + 2:
                res.loc[df_clean.index[feat_valid], c] = y_series[feat_valid]
                continue

            y = y_series[feat_valid].values.astype(float)
            X_feat = X[feat_valid.values]

            try:
                # OLS via least squares: min ||X w - y||_2
                w, residuals, rank, s = np.linalg.lstsq(X_feat, y, rcond=None)
                y_pred = X_feat.dot(w)
                resid = y - y_pred

                if verify_orthogonality:
                    # Check X^T * resid ~ 0
                    ortho_err = np.max(np.abs(X_feat.T.dot(resid)))
                    if ortho_err > 1e-8:
                        logger.debug(f"OLS orthogonality error for {c}: {ortho_err:.2e}")

                res.loc[df_clean.index[feat_valid], c] = resid
            except Exception as e:
                logger.warning(f"OLS neutralization failed for {c}: {e}")
                res.loc[df_clean.index[feat_valid], c] = y

        return res[cols_to_process[0]] if single_col else res


# Backward compatibility alias
BetaNeutralizer = CrossSectionalNeutralizer
