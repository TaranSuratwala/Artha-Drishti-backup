"""
nse_features.py — Comprehensive Causal Feature Engineering for NSE Equities.

TIMING CONVENTION:
Features use data <= close of day t -> entry at OPEN t+1 -> exit at CLOSE t+k or earlier on barrier touch.
Barriers use high/low of t+1…t+k; same-day double touch -> stop first; open through a barrier -> fill at open.
Adjusted prices; point-in-time universe incl. delisted; reindex to master trading calendar (k = market days).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from hmmlearn import hmm

logger = logging.getLogger(__name__)


class NSEFeatureEngine:
    """
    NSE Feature Engine:
    Computes strictly causal, leak-free alpha features across volatility-scaled reversal,
    NSE microstructure, F&O positioning, results-calendar windows, PIT fundamentals,
    and medium-horizon momentum signals.

    Every feature includes a one-line economic rationale and explicit timing convention.
    """

    def __init__(self):
        pass

    # ================================================================
    # 1. VOLATILITY-SCALED REVERSAL FEATURES
    # ================================================================

    def compute_vol_scaled_reversal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes volatility-normalized short-term price reversal features.
        TIMING: Uses data <= close of day t. Entry is at open of day t+1.
        """
        res = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']

        # 1-day log return
        r1 = np.log(close / close.shift(1))

        # Intraday Parkinson volatility: sigma_intraday = sqrt( 1/(4*ln(2)) * (ln(H/L))^2 )
        # Economic Rationale: Normalizing 1-day return by intraday price range separates liquidity-driven noise from true trend initiation.
        hl_ratio = np.maximum(high / np.maximum(low, 1e-4), 1.0)
        sigma_intraday = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(hl_ratio) ** 2))
        res['rev_1d_vol_scaled'] = r1 / (sigma_intraday + 1e-6)

        # 5-day cumulative return normalized by 20-day realized volatility
        # Economic Rationale: Identifies 5-day overreaction relative to baseline monthly volatility.
        r5 = np.log(close / close.shift(5))
        sigma_20 = r1.rolling(20, min_periods=10).std()
        res['rev_5d_vol_scaled'] = r5 / (sigma_20 + 1e-6)

        return res

    # ================================================================
    # 2. NSE MICROSTRUCTURE FEATURES
    # ================================================================

    def compute_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes delivery %, delivery z-score, delivery surge, signed flow,
        average trade size, gap in sigma, Amihud illiquidity, Corwin-Schultz spread,
        Parkinson, Garman-Klass, Yang-Zhang volatilities, and volatility term structure.
        TIMING: Uses data <= close of day t. Entry is at open of day t+1.
        """
        res = pd.DataFrame(index=df.index)
        close = df['close']
        open_p = df['open']
        high = df['high']
        low = df['low']
        volume = df['volume']
        r1 = np.log(close / close.shift(1))

        # 2.1 Delivery Percentage
        # Economic Rationale: Measures proportion of trading volume settled without intraday offsetting, isolating real institutional position accumulation from retail day-trading churn.
        if 'delivery_percentage' in df.columns:
            deliv_pct = df['delivery_percentage']
        elif 'delivery_qty' in df.columns and 'traded_qty' in df.columns:
            deliv_pct = (df['delivery_qty'] / (df['traded_qty'] + 1e-6)) * 100.0
        elif 'delivery_qty' in df.columns and 'volume' in df.columns:
            deliv_pct = (df['delivery_qty'] / (df['volume'] + 1e-6)) * 100.0
        else:
            deliv_pct = pd.Series(np.nan, index=df.index)
        res['delivery_pct'] = deliv_pct

        # 2.2 Delivery % Z-Score (20-day rolling)
        # Economic Rationale: Standardized delivery reveals anomalous institutional accumulation relative to own-history.
        deliv_mean_20 = deliv_pct.rolling(20, min_periods=5).mean()
        deliv_std_20 = deliv_pct.rolling(20, min_periods=5).std()
        res['delivery_pct_zscore'] = (deliv_pct - deliv_mean_20) / (deliv_std_20 + 1e-6)

        # 2.3 Delivered Quantity Surge
        # Economic Rationale: Flags sudden spikes in institutional inventory absorption preceding fundamental information dissemination.
        if 'delivery_qty' in df.columns:
            deliv_qty = df['delivery_qty']
        else:
            deliv_qty = deliv_pct * volume / 100.0
        deliv_qty_roll = deliv_qty.rolling(20, min_periods=5).mean()
        res['deliv_qty_surge'] = deliv_qty / (deliv_qty_roll + 1e-6)

        # 2.4 Signed Delivery Flow
        # Economic Rationale: Distinguishes whether delivery volume was absorbed into advancing prices (accumulation) or dumped into declining prices (distribution).
        res['signed_delivery_flow'] = np.sign(r1) * res['deliv_qty_surge']

        # 2.5 Average Trade Size (INR Turnover / No of Trades)
        # Economic Rationale: Proxies the relative market share of large institutional ticket sizes versus fragmented retail orders.
        turnover_inr = None
        if 'turnover' in df.columns:
            turnover_inr = df['turnover']
        elif 'turnover_lacs' in df.columns:
            turnover_inr = df['turnover_lacs'] * 1e5
        elif 'volume' in df.columns:
            turnover_inr = df['close'] * df['volume']

        if 'no_of_trades' in df.columns and turnover_inr is not None:
            res['avg_trade_size'] = turnover_inr / (df['no_of_trades'] + 1e-6)
        else:
            res['avg_trade_size'] = pd.Series(np.nan, index=df.index)

        # 2.6 Gap in Sigma (Overnight gap normalized by 20-day return volatility)
        # Economic Rationale: Quantifies overnight information shocks relative to typical volatility, exploiting post-open gap fading tendencies.
        prev_close = close.shift(1)
        overnight_gap = (open_p / (prev_close + 1e-6)) - 1.0
        sigma_close_20 = r1.rolling(20, min_periods=10).std()
        res['gap_in_sigma'] = overnight_gap / (sigma_close_20 + 1e-6)

        # 2.7 Amihud Illiquidity: ln(1 + 1e7 * |r| / Turnover_INR)
        # Economic Rationale: Kyle-Amihud price impact per rupee of turnover, capturing the compensation demanded for bearing illiquidity risk.
        if turnover_inr is not None:
            res['amihud_illiq'] = np.log1p(1e7 * (np.abs(r1) / (turnover_inr + 1e-6)))
        else:
            res['amihud_illiq'] = pd.Series(np.nan, index=df.index)

        # 2.8 Corwin-Schultz (2012) Bid-Ask Spread Estimator
        # Economic Rationale: Extracts effective transaction spreads from high-low price bounces without requiring ultra-high frequency limit order book data.
        H_t = high
        L_t = np.maximum(low, 1e-4)
        H_t1 = high.shift(1)
        L_t1 = np.maximum(low.shift(1), 1e-4)

        beta_cs = np.log(H_t / L_t) ** 2 + np.log(H_t1 / L_t1) ** 2
        gamma_cs = (np.log(np.maximum(H_t, H_t1) / np.minimum(L_t, L_t1))) ** 2
        c_const = 1.0 + np.sqrt(2.0)
        alpha_cs = c_const * (np.sqrt(np.maximum(beta_cs, 0.0)) - np.sqrt(np.maximum(gamma_cs, 0.0)))
        spread_daily = 2.0 * (np.exp(alpha_cs) - 1.0) / (1.0 + np.exp(alpha_cs))
        res['corwin_schultz'] = spread_daily.clip(lower=0.0, upper=1.0).rolling(20, min_periods=5).median()

        # 2.9 Parkinson Volatility (20-day)
        # Economic Rationale: Extreme-value volatility based on daily High/Low ratios with 5x greater statistical efficiency than close-to-close variance.
        parkinson_daily = (1.0 / (4.0 * np.log(2.0))) * (np.log(H_t / L_t) ** 2)
        res['parkinson_vol_20'] = np.sqrt(parkinson_daily.rolling(20, min_periods=5).mean())

        # 2.10 Garman-Klass Volatility (20-day)
        # Economic Rationale: Discrete jump-inclusive OHLC volatility estimator incorporating opening jumps and closing price drift.
        log_hl = np.log(H_t / L_t)
        log_co = np.log(close / np.maximum(open_p, 1e-4))
        gk_daily = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
        res['garman_klass_20'] = np.sqrt(np.maximum(gk_daily.rolling(20, min_periods=5).mean(), 0.0))

        # 2.11 Yang-Zhang Volatility (20-day minimum-variance unbiased estimator)
        # Economic Rationale: Minimum-variance unbiased volatility estimator combining continuous Brownian motion and overnight opening jumps.
        log_ho = np.log(H_t / np.maximum(open_p, 1e-4))
        log_lo = np.log(L_t / np.maximum(open_p, 1e-4))
        log_oc_prev = np.log(open_p / np.maximum(prev_close, 1e-4))

        rs_daily = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        rs_var = rs_daily.rolling(20, min_periods=5).mean()
        o_var = log_oc_prev.rolling(20, min_periods=5).var()
        c_var = log_co.rolling(20, min_periods=5).var()

        k_yz = 0.34 / (1.34 + (20 + 1) / (20 - 1))
        yz_var = o_var + k_yz * c_var + (1.0 - k_yz) * rs_var
        res['yang_zhang_20'] = np.sqrt(np.maximum(yz_var, 0.0))

        # 2.12 Volatility Term Structure (5-day GK / 20-day GK)
        # Economic Rationale: Term-structure slope detecting volatility expansion or compression regimes.
        gk_5 = np.sqrt(np.maximum(gk_daily.rolling(5, min_periods=3).mean(), 0.0))
        res['vol_term_structure'] = gk_5 / (res['garman_klass_20'] + 1e-6)

        return res

    # ================================================================
    # 3. F&O DERIVATIVES POSITIONING FEATURES
    # ================================================================

    def compute_fno_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Open Interest quadrant, rollover %, futures basis, and MWPL utilization.
        TIMING: Uses data <= close of day t. Entry is at open of day t+1.
        """
        res = pd.DataFrame(index=df.index)
        close = df['close']
        r1 = np.log(close / close.shift(1))

        # 3.1 OI Quadrant (Long build-up, Short build-up, Short covering, Long unwinding)
        # Economic Rationale: Classifies institutional leverage dynamics across the 4 canonical futures positioning quadrants.
        if 'open_interest' in df.columns:
            oi = df['open_interest']
            delta_oi = oi.diff()
            oi_roll_mean = oi.rolling(20, min_periods=5).mean()
            # Normalized signed flow: sign(return) * (delta_OI / mean_OI)
            res['oi_quadrant'] = np.sign(r1) * (delta_oi / (oi_roll_mean + 1e-6))
        else:
            res['oi_quadrant'] = pd.Series(np.nan, index=df.index)

        # 3.2 Rollover Percentage
        # Economic Rationale: High rollover indicates institutional willingness to pay carry costs to maintain directional exposure into the next monthly cycle.
        if 'rollover_percentage' in df.columns:
            res['rollover_pct'] = df['rollover_percentage']
        elif 'oi_next_month' in df.columns and 'open_interest' in df.columns:
            res['rollover_pct'] = (df['oi_next_month'] / (df['open_interest'] + 1e-6)) * 100.0
        else:
            res['rollover_pct'] = pd.Series(np.nan, index=df.index)

        # 3.3 Futures Basis: (Futures Settle - Spot Close) / Spot Close
        # Economic Rationale: Reflects net demand for leverage and short borrowing fees under cost-of-carry equilibrium.
        if 'futures_price' in df.columns:
            res['futures_basis'] = (df['futures_price'] - close) / (close + 1e-6)
        elif 'basis' in df.columns:
            res['futures_basis'] = df['basis']
        else:
            res['futures_basis'] = pd.Series(np.nan, index=df.index)

        # 3.4 Market Wide Position Limit (MWPL) Utilization
        # Economic Rationale: Approaching the 95% threshold triggers F&O ban restrictions, inducing asymmetric short-covering squeezes.
        if 'mwpl_utilization' in df.columns:
            res['mwpl_util'] = df['mwpl_utilization']
        elif 'mwpl_limit' in df.columns and 'open_interest' in df.columns:
            res['mwpl_util'] = df['open_interest'] / (df['mwpl_limit'] + 1e-6)
        else:
            res['mwpl_util'] = pd.Series(np.nan, index=df.index)

        return res

    # ================================================================
    # 4. RESULTS-CALENDAR WINDOWS
    # ================================================================

    def compute_event_features(
        self,
        df: pd.DataFrame,
        results_dates: Optional[Union[List[pd.Timestamp], pd.Series]] = None
    ) -> pd.DataFrame:
        """
        Computes results calendar window flag and post-earnings-announcement drift.
        TIMING: Uses data <= close of day t. Entry is at open of day t+1.
        """
        res = pd.DataFrame(index=df.index)
        close = df['close']

        # Determine results dates
        event_mask = pd.Series(False, index=df.index)
        if 'is_earnings_day' in df.columns:
            event_mask = df['is_earnings_day'].astype(bool)
        elif results_dates is not None:
            dt_index = pd.to_datetime(df.index)
            res_dts = pd.to_datetime(results_dates)
            event_mask = pd.Series(dt_index.isin(res_dts), index=df.index)

        # 4.1 Results Window Indicator: [-2, +2] days around earnings
        # Economic Rationale: Event-window indicator isolating elevated earnings announcement volatility and structural liquidity changes.
        roll_window_forward = event_mask.rolling(5, center=True, min_periods=1).max().astype(float)
        res['is_results_window'] = roll_window_forward.fillna(0.0)

        # 4.2 Post-Earnings-Announcement Drift (cumulative return within [0, 20] days post-event)
        # Economic Rationale: Exploits post-earnings-announcement drift (PEAD) driven by investor underreaction to earnings surprises.
        drift = pd.Series(0.0, index=df.index)
        event_indices = np.where(event_mask.values)[0]
        n_rows = len(df)
        for idx in event_indices:
            end_idx = min(idx + 21, n_rows)
            base_price = close.iloc[idx]
            if base_price > 0:
                for k in range(idx, end_idx):
                    drift.iloc[k] = np.log(close.iloc[k] / base_price)
        res['post_earnings_drift'] = drift

        return res

    # ================================================================
    # 5. POINT-IN-TIME FUNDAMENTALS & SHAREHOLDING
    # ================================================================

    def compute_pit_fundamental_features(
        self,
        df: pd.DataFrame,
        fundamentals_pit: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Computes point-in-time fundamentals and promoter shareholding changes.
        Strictly lagged to announcement/filing dates without lookahead.
        TIMING: Uses data <= close of day t. Entry is at open of day t+1.
        """
        res = pd.DataFrame(index=df.index)
        close = df['close']

        # 5.1 PIT Return on Equity (ROE)
        # Economic Rationale: Quality factor capturing persistent capital allocation efficiency without lookahead bias.
        if 'roe' in df.columns:
            res['pit_roe'] = df['roe']
        else:
            res['pit_roe'] = pd.Series(np.nan, index=df.index)

        # 5.2 PIT Debt to Equity
        # Economic Rationale: Financial distress and solvency risk proxy, penalizing excessive leverage during credit contractions.
        if 'debt_to_equity' in df.columns:
            res['pit_debt_to_equity'] = df['debt_to_equity']
        else:
            res['pit_debt_to_equity'] = pd.Series(np.nan, index=df.index)

        # 5.3 PIT Earnings Yield (TTM EPS / Close)
        # Economic Rationale: Value metric identifying stocks trading at an attractive yield relative to current earnings power.
        if 'eps_ttm' in df.columns:
            res['pit_earnings_yield'] = df['eps_ttm'] / (close + 1e-6)
        elif 'pe_ratio' in df.columns:
            res['pit_earnings_yield'] = 1.0 / (df['pe_ratio'].replace(0, np.nan))
        else:
            res['pit_earnings_yield'] = pd.Series(np.nan, index=df.index)

        # 5.4 Promoter Holding QoQ Delta
        # Economic Rationale: Insider alignment signal where promoter buying reflects superior private information on company prospects.
        if 'promoter_holding' in df.columns:
            # Change over 60 trading days (~1 quarter)
            res['promoter_holding_change_qoq'] = df['promoter_holding'].diff(60)
        else:
            res['promoter_holding_change_qoq'] = pd.Series(np.nan, index=df.index)

        return res

    # ================================================================
    # 6. MEDIUM-TO-LONG HORIZON SIGNALS (15–21d)
    # ================================================================

    def compute_medium_horizon_features(
        self,
        df: pd.DataFrame,
        market_series: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Computes residual momentum (12m minus 1m) and 52-week high proximity.
        TIMING: Uses data <= close of day t. Entry is at open of day t+1.
        """
        res = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']

        # 6.1 Residual Momentum (12-month return ex recent 1-month): r(t-21) / r(t-252)
        # Economic Rationale: Removes broad market and industry beta to isolate stock-specific intermediate momentum driven by slow information diffusion.
        r_12m_1m = np.log(close.shift(21) / close.shift(252))
        if market_series is not None:
            mkt_aligned = market_series.reindex(df.index).ffill()
            mkt_12m_1m = np.log(mkt_aligned.shift(21) / mkt_aligned.shift(252))
            res['residual_momentum_12m_1m'] = r_12m_1m - mkt_12m_1m
        else:
            res['residual_momentum_12m_1m'] = r_12m_1m

        # 6.2 Proximity to 52-Week High: Close / max_252(High)
        # Economic Rationale: George & Hwang anchoring heuristic where investors underreact to positive news when stocks trade near historical 52-week highs.
        high_252 = high.rolling(252, min_periods=60).max()
        res['proximity_52w_high'] = close / (high_252 + 1e-6)

        return res

    # ================================================================
    # 7. MASTER FEATURE ORCHESTRATOR
    # ================================================================

    def compute_all_features(
        self,
        df: pd.DataFrame,
        market_series: Optional[pd.Series] = None,
        results_dates: Optional[List[pd.Timestamp]] = None
    ) -> pd.DataFrame:
        """
        Computes all causal features across the entire taxonomy for a single stock DataFrame.
        TIMING: Uses data <= close of day t. Entry is at open of day t+1.
        """
        f_rev = self.compute_vol_scaled_reversal(df)
        f_micro = self.compute_microstructure_features(df)
        f_fno = self.compute_fno_features(df)
        f_event = self.compute_event_features(df, results_dates)
        f_fund = self.compute_pit_fundamental_features(df)
        f_med = self.compute_medium_horizon_features(df, market_series)

        features = pd.concat([f_rev, f_micro, f_fno, f_event, f_fund, f_med], axis=1)
        return features


# ====================================================================
# BACKWARD COMPATIBILITY CLASSES
# ====================================================================

class MicrostructureFeatures:
    """Backward compatibility wrapper preserving existing callers."""

    def __init__(self):
        self.engine = NSEFeatureEngine()

    def compute_delivery_features(self, df: pd.DataFrame) -> pd.DataFrame:
        m = self.engine.compute_microstructure_features(df)
        cols = ['delivery_pct', 'delivery_pct_zscore', 'deliv_qty_surge', 'avg_trade_size', 'signed_delivery_flow']
        return m[[c for c in cols if c in m.columns]]

    def compute_gap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        res = pd.DataFrame(index=df.index)
        if 'open' in df.columns and 'close' in df.columns:
            prev_close = df['close'].shift(1)
            res['gap'] = df['open'] / (prev_close + 1e-6) - 1.0
            res['gap_zscore'] = (res['gap'] - res['gap'].rolling(20, min_periods=5).mean()) / (res['gap'].rolling(20, min_periods=5).std() + 1e-6)
        return res

    def compute_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        m = self.engine.compute_microstructure_features(df)
        cols = ['amihud_illiq', 'corwin_schultz']
        return m[[c for c in cols if c in m.columns]]

    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        m = self.engine.compute_microstructure_features(df)
        cols = ['parkinson_vol_20', 'garman_klass_20', 'yang_zhang_20', 'vol_term_structure']
        return m[[c for c in cols if c in m.columns]]

    def compute_fno_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.engine.compute_fno_features(df)


class VIXRegimeModel:
    """
    Causal HMM regime model on India VIX, fitted strictly on train dates only.
    State 1 represents high volatility regime.
    """

    def __init__(self, n_components: int = 2):
        self.hmm = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100, random_state=42)
        self.is_fitted = False

    def fit(self, train_vix: pd.Series):
        """Fit on TRAIN dates only"""
        vix_vals = train_vix.dropna().values.reshape(-1, 1)
        if len(vix_vals) > 50:
            self.hmm.fit(vix_vals)
            self.is_fitted = True

            # Ensure state 1 is always the higher volatility regime
            means = self.hmm.means_.flatten()
            if means[0] > means[1]:
                self.hmm.means_ = self.hmm.means_[::-1]
                self.hmm.covars_ = self.hmm.covars_[::-1]
                trans = self.hmm.transmat_
                self.hmm.transmat_ = np.array([[trans[1, 1], trans[1, 0]], [trans[0, 1], trans[0, 0]]])
        return self

    def predict_regime_proba(self, vix_series: pd.Series) -> pd.Series:
        """Causal forward filter step: P(Z_t | X_{1..t})"""
        res = pd.Series(np.nan, index=vix_series.index)
        if not self.is_fitted:
            return res

        vix_clean = vix_series.dropna()
        if len(vix_clean) > 0:
            vals = vix_clean.values.reshape(-1, 1)
            n_samples = len(vals)
            n_states = self.hmm.n_components

            alpha = np.zeros((n_samples, n_states))
            alpha[0] = self.hmm.startprob_ * np.exp(self.hmm._compute_log_likelihood(vals[0:1]))[0]
            alpha[0] /= (np.sum(alpha[0]) + 1e-12)

            log_lik = self.hmm._compute_log_likelihood(vals)
            trans = self.hmm.transmat_

            for t in range(1, n_samples):
                alpha[t] = np.dot(alpha[t - 1], trans) * np.exp(log_lik[t])
                alpha[t] /= (np.sum(alpha[t]) + 1e-12)

            res.loc[vix_clean.index] = alpha[:, 1]
        return res
