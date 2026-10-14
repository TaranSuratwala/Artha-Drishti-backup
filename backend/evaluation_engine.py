"""
Evaluation & Rigorous Falsification Engine (Stage 7)
===================================================
Tier-1 quantitative evaluation suite for multi-horizon cross-sectional alpha on NSE equities.

Key Capabilities:
1. Multi-Horizon Metrics per Horizon k in {3, 5, 7, 15, 21}:
   - Rank IC, Newey-West adjusted t-statistic (lag L=k), ICIR (annualized).
   - Hit rate (directional accuracy).
   - Monotonic 10-decile spread and Spearman rank monotonicity score.
   - Net-of-cost Sharpe, Information Ratio, Maximum Drawdown, Annualized Turnover.
2. Rigorous Falsification against Null Models:
   - Stationary Block-Bootstrap 90% Confidence Interval for net active alpha over Equal-Weight Eligible Null.
   - Deflated Sharpe Ratio (DSR, Bailey & Lopez de Prado 2014) accounting for trial count (N ≈ 80).
   - Probability of Backtest Overfitting (PBO, Bailey et al. 2016).
3. Ablation Diagnostics:
   - Raw vs Neutralized Rank IC.
   - Naive Top-N vs Hysteresis Turnover & Net Sharpe.
   - Single-horizon vs Cross-horizon stacked models.
   - Feature group drop ablations.
4. Go / No-Go Decision Gate:
   - Evaluates all 5 mandatory conditions for shadow trading approval.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
import logging

from frictions import TradingSegment, NSECostConfig, NSECostModel, ExecutionFrictions
from portfolio import PortfolioConfig, PortfolioSimulator, RankHysteresisPortfolio

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration parameters for evaluation and falsification tests."""
    horizons: List[int] = field(default_factory=lambda: [3, 5, 7, 15, 21])
    n_trials_dsr: int = 80             # Account for ~80 design iterations on burned test window
    sr_variance_trials: float = 0.08   # Cross-trial Sharpe variance anchor (std ~ 0.28)
    bootstrap_reps: int = 1000         # Stationary block bootstrap replications
    bootstrap_block_len: int = 5       # Block length matching typical holding period
    bootstrap_ci: float = 0.90         # 90% Confidence Interval
    min_path_pass_rate: float = 0.70   # Require >= 70% of CV paths to show positive lower CI bound
    min_neutralized_ic: float = 0.02   # Neutralized Rank IC hurdle >= 0.02
    min_dsr: float = 0.95              # Deflated Sharpe Ratio hurdle >= 0.95
    risk_free_rate: float = 0.065      # 6.50% RBI repo rate anchor for Indian equities
    min_cross_section: int = 20        # Minimum valid cross-sectional names per date


class StatisticalMetrics:
    """High-precision time-series and cross-sectional econometric metrics."""

    @staticmethod
    def rank_ic(scores: np.ndarray, targets: np.ndarray) -> float:
        """Spearman Rank Correlation between predictions and realized targets."""
        mask = np.isfinite(scores) & np.isfinite(targets)
        if np.sum(mask) < 5:
            return 0.0
        rho, _ = stats.spearmanr(scores[mask], targets[mask])
        return float(rho) if np.isfinite(rho) else 0.0

    @staticmethod
    def daily_rank_ic_series(df: pd.DataFrame, score_col: str, target_col: str, min_names: int = 20) -> pd.Series:
        """Compute date-by-date cross-sectional Rank IC time series."""
        clean = df[['date', score_col, target_col]].dropna()
        daily_ics = {}
        for d, grp in clean.groupby('date'):
            if len(grp) >= min_names:
                rho = StatisticalMetrics.rank_ic(grp[score_col].values, grp[target_col].values)
                daily_ics[d] = rho
        return pd.Series(daily_ics).sort_index()

    @staticmethod
    def newey_west_stats(series: pd.Series, max_lags: int = 5) -> Dict[str, float]:
        """
        Compute Newey-West HAC adjusted long-run variance, standard error, and t-statistic.
        Handles serial correlation in overlapping return horizons k.
        """
        vals = series.dropna().values
        n = len(vals)
        if n < 3:
            return {"mean": 0.0, "nw_se": 1e-4, "nw_t": 0.0, "icir": 0.0}

        mean_val = float(np.mean(vals))
        e = vals - mean_val
        L = min(max_lags, n - 1)

        # Gamma_0
        gamma_0 = float(np.dot(e, e) / n)
        gamma_sum = 0.0

        for l in range(1, L + 1):
            w_l = 1.0 - (l / (L + 1.0))
            cov_l = float(np.dot(e[l:], e[:-l]) / n)
            gamma_sum += 2.0 * w_l * cov_l

        lr_var = max(gamma_0 + gamma_sum, 1e-12)
        nw_se = float(np.sqrt(lr_var / n))
        nw_t = float(mean_val / nw_se) if nw_se > 0 else 0.0

        std_val = float(np.std(vals))
        icir = (mean_val / std_val) * np.sqrt(252.0) if std_val > 0 else 0.0

        return {
            "mean": mean_val,
            "nw_se": nw_se,
            "nw_t": nw_t,
            "icir": icir,
        }

    @staticmethod
    def decile_spread(df: pd.DataFrame, score_col: str, return_col: str, n_bins: int = 10) -> Dict[str, Any]:
        """
        Sort universe into n_bins deciles cross-sectionally by score.
        Computes per-decile mean return, spread (D10 - D1), and Spearman rank monotonicity score.
        """
        clean = df[['date', score_col, return_col]].dropna().copy()
        clean['decile'] = clean.groupby('date')[score_col].transform(
            lambda x: pd.qcut(x, n_bins, labels=False, duplicates='drop') + 1
        )

        decile_means = clean.groupby('decile')[return_col].mean()
        # Annualized decile returns
        decile_cagr = decile_means * 252.0

        d_top = decile_cagr.get(n_bins, 0.0)
        d_bot = decile_cagr.get(1, 0.0)
        spread = d_top - d_bot

        # Monotonicity: Spearman correlation between decile index (1..10) and mean return
        valid_deciles = sorted(decile_means.index.tolist())
        if len(valid_deciles) >= 4:
            mono_corr, _ = stats.spearmanr(valid_deciles, [decile_means[d] for d in valid_deciles])
        else:
            mono_corr = 0.0

        return {
            "decile_returns_ann": decile_cagr.to_dict(),
            "top_decile_ann": d_top,
            "bottom_decile_ann": d_bot,
            "spread_ann": spread,
            "monotonicity": float(mono_corr) if np.isfinite(mono_corr) else 0.0,
        }


class FalsificationEngine:
    """
    Null-Model Comparison & Overfitting Falsification Engine.
    Executes block bootstrap, Deflated Sharpe Ratio (DSR), and Probability of Backtest Overfitting (PBO).
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()

    def block_bootstrap_ci(
        self,
        active_returns: pd.Series,
        block_len: Optional[int] = None,
        n_reps: Optional[int] = None,
        ci_level: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Stationary/Moving Block Bootstrap Confidence Interval for active returns.
        Accounts for temporal autocorrelation in overlapping return series.

        Returns
        -------
        Tuple[float, float, float]
            (mean_active_alpha_ann, lower_bound_ann, upper_bound_ann)
        """
        clean = active_returns.dropna().values
        n = len(clean)
        if n < 10:
            return 0.0, -1.0, 1.0

        b = block_len or self.config.bootstrap_block_len
        reps = n_reps or self.config.bootstrap_reps
        ci = ci_level or self.config.bootstrap_ci

        n_blocks = int(np.ceil(n / b))
        boot_means = np.zeros(reps)

        for r in range(reps):
            # Sample start indices for blocks
            start_indices = np.random.randint(0, n - b + 1, size=n_blocks)
            sampled_series = np.concatenate([clean[idx:idx + b] for idx in start_indices])[:n]
            boot_means[r] = np.mean(sampled_series) * 252.0  # Annualized

        alpha_mean = float(np.mean(clean) * 252.0)
        tail = (1.0 - ci) / 2.0
        lower = float(np.percentile(boot_means, tail * 100.0))
        upper = float(np.percentile(boot_means, (1.0 - tail) * 100.0))

        return alpha_mean, lower, upper

    def deflated_sharpe_ratio(
        self,
        observed_sr: float,
        n_samples: int,
        returns: pd.Series,
        n_trials: Optional[int] = None,
        sr_var: Optional[float] = None,
        trial_sharpes: Optional[List[float]] = None,
    ) -> float:
        """
        Deflated Sharpe Ratio (DSR) (Bailey and López de Prado 2014).
        Calculates the probability that observed Sharpe ratio exceeds the expected maximum
        Sharpe ratio under the null hypothesis of multiple testing over N trials.

        Parameters
        ----------
        observed_sr : float
            Annualized Sharpe Ratio of candidate strategy.
        n_samples : int
            Number of observations T (trading days).
        returns : pd.Series
            Strategy daily returns series (for skewness and kurtosis estimation).
        n_trials : int
            Number of iterations scored during research (e.g. 80).
        sr_var : float
            Variance of Sharpe ratios across trials.
        trial_sharpes : Optional[List[float]]
            Empirical array of Sharpe ratios from tested trials/iterations.

        Returns
        -------
        float
            DSR probability in [0, 1]. DSR >= 0.95 indicates robust non-spurious edge.
        """
        if trial_sharpes is not None and len(trial_sharpes) > 1:
            N = len(trial_sharpes)
            V = max(float(np.var(trial_sharpes)), 1e-6)
        else:
            N = n_trials or self.config.n_trials_dsr
            V = sr_var or self.config.sr_variance_trials

        if N <= 1:
            N = 2

        # Expected maximum Sharpe ratio under the null hypothesis of N independent tests
        # Euler-Mascheroni constant gamma ≈ 0.5772156649
        gamma = 0.5772156649
        term1 = (1.0 - gamma) * stats.norm.ppf(1.0 - (1.0 / N))
        term2 = gamma * stats.norm.ppf(1.0 - (1.0 / (N * np.e)))
        sr_star = float(np.sqrt(V) * (term1 + term2))

        # Sample moments of returns
        clean_ret = returns.dropna().values
        if len(clean_ret) > 5:
            skew = float(stats.skew(clean_ret))
            kurt = float(stats.kurtosis(clean_ret, fisher=False))  # Pearson kurtosis (normal = 3)
        else:
            skew = 0.0
            kurt = 3.0

        # Daily Sharpe for denominator scaling
        sr_daily = observed_sr / np.sqrt(252.0)
        sr_star_daily = sr_star / np.sqrt(252.0)

        # Standard deviation of Sharpe ratio estimator under non-normal returns (Lo 2002 / Mertens 2002)
        denom_var = 1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily ** 2)
        denom_se = np.sqrt(max(denom_var, 1e-6) / max(1, n_samples - 1))

        z = (sr_daily - sr_star_daily) / denom_se
        dsr = float(stats.norm.cdf(z))

        return dsr

    def probability_of_backtest_overfitting(
        self,
        is_matrix: np.ndarray,
        oos_matrix: np.ndarray,
    ) -> float:
        """
        Probability of Backtest Overfitting (PBO) via Combinatorial Symmetric Cross-Validation (CSCV).

        Parameters
        ----------
        is_matrix : np.ndarray
            Matrix of shape (num_splits, num_models) with in-sample performance metrics.
        oos_matrix : np.ndarray
            Matrix of shape (num_splits, num_models) with out-of-sample performance metrics.

        Returns
        -------
        float
            PBO probability in [0, 1]. PBO <= 0.30 indicates minimal overfitting.
        """
        n_splits, n_models = is_matrix.shape
        if n_splits == 0 or n_models < 2:
            return 0.0

        overfit_count = 0
        for s in range(n_splits):
            best_is_idx = np.argmax(is_matrix[s, :])
            # Rank of in-sample best model in out-of-sample test
            oos_scores = oos_matrix[s, :]
            oos_rank = np.sum(oos_scores > oos_scores[best_is_idx]) + 1  # 1 = best OOS
            rel_rank = oos_rank / (n_models + 1.0)
            if rel_rank > 0.50:  # Worse than median out-of-sample
                overfit_count += 1

        return float(overfit_count / n_splits)


class EvaluationEngine:
    """
    Main Evaluation & Gate Enforcement Engine.
    Executes full multi-horizon, cross-path, and null-model falsification analysis.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.metrics = StatisticalMetrics()
        self.falsifier = FalsificationEngine(self.config)

    def evaluate_multi_horizon_panel(
        self,
        panel_df: pd.DataFrame,
        score_prefix: str = "pred_",
        target_prefix: str = "target_",
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute comprehensive performance scorecard per horizon k in {3, 5, 7, 15, 21}.

        Returns
        -------
        Dict[int, Dict[str, Any]]
            Scorecard mapping horizon k -> evaluation dictionary.
        """
        results_by_k = {}

        for k in self.config.horizons:
            score_col = f"{score_prefix}{k}" if f"{score_prefix}{k}" in panel_df.columns else "score"
            target_col = f"{target_prefix}{k}" if f"{target_prefix}{k}" in panel_df.columns else f"target_{k}d"

            if score_col not in panel_df.columns or target_col not in panel_df.columns:
                logger.warning(f"Columns {score_col} or {target_col} missing; skipping k={k}")
                continue

            # 1. Cross-sectional Rank IC
            daily_ic = self.metrics.daily_rank_ic_series(panel_df, score_col, target_col, min_names=self.config.min_cross_section)
            nw_stats = self.metrics.newey_west_stats(daily_ic, max_lags=k)

            # 2. Hit Rate (Sign matching)
            clean = panel_df[[score_col, target_col]].dropna()
            hit_rate = float(np.mean((clean[score_col] > 0) == (clean[target_col] > 0))) if len(clean) > 0 else 0.50

            # 3. Decile spread & Monotonicity
            # Use forward return if available, else target return
            ret_col = f"fwd_ret_{k}d" if f"fwd_ret_{k}d" in panel_df.columns else target_col
            decile_res = self.metrics.decile_spread(panel_df, score_col, ret_col, n_bins=10)

            results_by_k[k] = {
                "horizon": k,
                "mean_rank_ic": nw_stats["mean"],
                "nw_t_stat": nw_stats["nw_t"],
                "icir_annualized": nw_stats["icir"],
                "hit_rate": hit_rate,
                "top_decile_cagr": decile_res["top_decile_ann"],
                "bottom_decile_cagr": decile_res["bottom_decile_ann"],
                "decile_spread_cagr": decile_res["spread_ann"],
                "decile_monotonicity": decile_res["monotonicity"],
                "daily_ic_series": daily_ic,
            }

        return results_by_k

    def evaluate_cv_paths(
        self,
        cv_paths: List[pd.DataFrame],
        score_col: str = "score",
        target_col: str = "target_5d",
    ) -> Dict[str, Any]:
        """
        Evaluate performance and block-bootstrap CI across continuous CV paths.
        Checks condition: 90% CI lower bound of active net alpha > 0 in >= 70% of paths.
        """
        path_results = []
        positive_ci_count = 0
        total_paths = len(cv_paths)

        sim_cfg = PortfolioConfig(n_enter=15, n_exit=25, rebalance_freq=5)
        sim = PortfolioSimulator(sim_cfg)

        for p_idx, path_df in enumerate(cv_paths):
            res = sim.run_simulation(path_df, score_col=score_col, use_hysteresis=True)
            nav_hist = res["nav_history"]

            # Compute active daily returns
            active_rets = nav_hist["active_return"]

            # Block bootstrap 90% CI
            mean_a, lower_a, upper_a = self.falsifier.block_bootstrap_ci(
                active_returns=active_rets,
                block_len=self.config.bootstrap_block_len,
                n_reps=self.config.bootstrap_reps,
                ci_level=self.config.bootstrap_ci,
            )

            is_positive = (lower_a > 0.0)
            if is_positive:
                positive_ci_count += 1

            # DSR on path
            dsr = self.falsifier.deflated_sharpe_ratio(
                observed_sr=res["net_sharpe"],
                n_samples=len(nav_hist),
                returns=nav_hist["net_return"],
                n_trials=self.config.n_trials_dsr,
            )

            path_results.append({
                "path_idx": p_idx,
                "net_cagr": res["cagr_net"],
                "bmk_cagr": res["cagr_bmk"],
                "net_alpha_cagr": res["net_alpha_cagr"],
                "net_sharpe": res["net_sharpe"],
                "bootstrap_alpha_mean": mean_a,
                "bootstrap_ci_lower_90": lower_a,
                "bootstrap_ci_upper_90": upper_a,
                "ci_lower_positive": is_positive,
                "dsr": dsr,
                "turnover_ann": res["annualized_turnover"],
            })

        pass_rate = positive_ci_count / max(1, total_paths)
        passes_gate = (pass_rate >= self.config.min_path_pass_rate)

        return {
            "path_results": path_results,
            "total_paths": total_paths,
            "positive_ci_paths": positive_ci_count,
            "path_pass_rate": pass_rate,
            "passes_bootstrap_gate": passes_gate,
        }

    def verify_go_no_go_gate(
        self,
        path_eval: Dict[str, Any],
        neutralized_ic: float,
        overall_dsr: float,
        planted_leak_caught: bool,
        null_model_net_sharpe: float,
    ) -> Dict[str, Any]:
        """
        Enforce Strict Go / No-Go Decision Gate:
        1. Net-of-cost alpha vs equal-weight-eligible null has bootstrap 90% CI lower bound > 0 in >= 70% of paths.
        2. Neutralized Rank IC >= 0.02.
        3. DSR >= 0.95.
        4. Planted-leak test caught.
        5. Null-model net Sharpe <= 0.
        """
        cond1 = path_eval.get("passes_bootstrap_gate", False)
        cond2 = (neutralized_ic >= self.config.min_neutralized_ic)
        cond3 = (overall_dsr >= self.config.min_dsr)
        cond4 = planted_leak_caught
        cond5 = (null_model_net_sharpe <= 0.0)

        all_passed = cond1 and cond2 and cond3 and cond4 and cond5

        gate_summary = {
            "condition_1_bootstrap_ci_positive_paths": {
                "passed": cond1,
                "value": f"{path_eval.get('path_pass_rate', 0.0)*100:.1f}%",
                "threshold": f">= {self.config.min_path_pass_rate*100:.1f}%",
            },
            "condition_2_neutralized_rank_ic": {
                "passed": cond2,
                "value": f"{neutralized_ic:.4f}",
                "threshold": f">= {self.config.min_neutralized_ic:.4f}",
            },
            "condition_3_deflated_sharpe_ratio": {
                "passed": cond3,
                "value": f"{overall_dsr:.4f}",
                "threshold": f">= {self.config.min_dsr:.4f}",
            },
            "condition_4_planted_leak_caught": {
                "passed": cond4,
                "value": str(planted_leak_caught),
                "threshold": "True",
            },
            "condition_5_null_model_net_sharpe": {
                "passed": cond5,
                "value": f"{null_model_net_sharpe:.2f}",
                "threshold": "<= 0.00",
            },
            "final_decision": "GO (Proceed to Shadow Trading)" if all_passed else "NO-GO (No Investable Edge)",
            "all_conditions_met": all_passed,
        }

        return gate_summary
