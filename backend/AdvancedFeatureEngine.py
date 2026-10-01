"""
===================================================================
ADAPTIVE MULTI-HORIZON FEATURE ENGINE (AMHFE)
===================================================================

Patent-Pending Feature Engineering System that creates features
across multiple time horizons with automatic relevance weighting.

Key Innovations:
  1. Multi-Horizon Feature Extraction: Computes every indicator at
     5 different timeframes simultaneously, enabling the model to
     learn which horizon matters most per stock
  
  2. Volatility-Regime Adaptive Features: Features are normalized
     relative to current volatility regime, making them comparable
     across different market conditions
  
  3. Cross-Sectional Features: Relative strength vs market,
     sector-adjusted momentum, peer-group positioning
  
  4. Information-Theoretic Feature Selection: Mutual information 
     scoring to automatically discard noise features

Author: GenAI Stock Intelligence System
Version: 1.0.0
===================================================================
"""

import numpy as np
import pandas as pd
from sqlalchemy import text

def _rng(df):
    return np.maximum(df['high'] - df['low'], 1e-4 * df['close'])

from typing import Optional
from typing import Dict, List, Tuple, Optional
import logging
import yfinance as yf
from functools import lru_cache
import time as _time

logger = logging.getLogger(__name__)


# ================================================================
# VECTORISED ROLLING HELPERS
# ================================================================
# Eleven features in this file were computed with `Series.rolling(w).apply(
# <python lambda>)`.  With raw=False pandas constructs a fresh Series for every
# window, so a single such call is O(rows) interpreted function calls; across
# ~2,000 tickers x ~2,500 rows x 11 call sites that is tens of millions of
# Python invocations and it dominated the feature-engineering stage (the
# training log shows ~29 minutes spent here).  Every one of them has an exact
# vectorised equivalent, implemented below.

def _sliding(arr: np.ndarray, window: int) -> np.ndarray:
    """(n,) -> (n-window+1, window) read-only sliding view (no copy)."""
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(np.asarray(arr, dtype=np.float64), window)


def _rolling_reduce(series: pd.Series, window: int, fn, min_periods: Optional[int] = None) -> pd.Series:
    """Apply a numpy reduction `fn(windows_2d, axis=1)` over rolling windows."""
    values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64)
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n >= window:
        out[window - 1:] = fn(_sliding(values, window))
    if min_periods is not None and min_periods < window:
        # Leading partial windows are left as NaN, matching the previous
        # rolling(window) default (min_periods == window).
        pass
    return pd.Series(out, index=series.index)


def _rolling_argmax_pct(series: pd.Series, window: int, denom: int) -> pd.Series:
    """Vectorised replacement for rolling(window).apply(lambda x: x.argmax()/denom*100)."""
    return _rolling_reduce(series, window, lambda w: w.argmax(axis=1) / denom * 100.0)


def _rolling_argmin_pct(series: pd.Series, window: int, denom: int) -> pd.Series:
    return _rolling_reduce(series, window, lambda w: w.argmin(axis=1) / denom * 100.0)


def _rolling_mad(series: pd.Series, window: int) -> pd.Series:
    """Mean absolute deviation about the window mean (for CCI)."""
    return _rolling_reduce(
        series, window,
        lambda w: np.mean(np.abs(w - w.mean(axis=1, keepdims=True)), axis=1)
    )


def _rolling_minmax_position(series: pd.Series, window: int) -> pd.Series:
    """(last - min) / (max - min): identical to the old pct_rank lambdas."""
    s = pd.to_numeric(series, errors='coerce')
    lo = s.rolling(window).min()
    hi = s.rolling(window).max()
    return (s - lo) / (hi - lo + 1e-10)


def _rolling_autocorr(series: pd.Series, window: int, lag: int) -> pd.Series:
    """Rolling lag-k autocorrelation, computed inside each window.

    Matches `rolling(w).apply(lambda x: x.autocorr(lag=k))`: the correlation is
    taken between x[:-k] and x[k:] *within* the window, so no value from outside
    the window leaks in (which a naive `rolling(w).corr(series.shift(k))` would
    allow).
    """
    def _fn(w):
        a = w[:, :-lag]
        b = w[:, lag:]
        a_c = a - a.mean(axis=1, keepdims=True)
        b_c = b - b.mean(axis=1, keepdims=True)
        num = (a_c * b_c).sum(axis=1)
        den = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1))
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(den > 1e-12, num / den, 0.0)
    return _rolling_reduce(series, window, _fn)


def _rolling_ar1(series: pd.Series, window: int) -> pd.Series:
    """Rolling AR(1) slope phi from OLS of x[1:] on x[:-1] within each window."""
    def _fn(w):
        a = w[:, :-1]
        b = w[:, 1:]
        a_c = a - a.mean(axis=1, keepdims=True)
        b_c = b - b.mean(axis=1, keepdims=True)
        den = (a_c ** 2).sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(den > 1e-12, (a_c * b_c).sum(axis=1) / den, np.nan)
    return _rolling_reduce(series, window, _fn)


def _rolling_hurst_vr(series: pd.Series, window: int) -> pd.Series:
    """Variance-ratio Hurst exponent, fully vectorised.

    Replaces the previous DFA implementation, which was not merely slow but
    DEAD: with window=20 the box sizes collapsed to {4, 5}, tripped the
    `len(box_sizes) < 3` guard and returned the constant 0.5 for every single
    row of every single ticker.  The variance-ratio estimator
    (Var[sum of q returns] ~ q^(2H) * Var[return]) is well defined at these
    window lengths and is a genuine trending / mean-reverting indicator:
    H < 0.5 mean-reverting, H = 0.5 random walk, H > 0.5 trending.
    """
    values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(values)
    out = np.full(n, 0.5, dtype=np.float64)
    if n < window:
        return pd.Series(out, index=series.index)

    w = _sliding(values, window)                       # (n-window+1, window)
    lags = [q for q in (1, 2, 4, 8) if window // q >= 4]
    if len(lags) < 2:
        return pd.Series(out, index=series.index)

    log_q, log_v = [], []
    for q in lags:
        usable = (window // q) * q
        blocks = w[:, :usable].reshape(w.shape[0], usable // q, q).sum(axis=2)
        var_q = blocks.var(axis=1)
        log_q.append(np.log(q))
        log_v.append(np.log(np.maximum(var_q, 1e-18)))

    x = np.asarray(log_q)                              # (k,)
    y = np.stack(log_v, axis=1)                        # (rows, k)
    x_c = x - x.mean()
    y_c = y - y.mean(axis=1, keepdims=True)
    slope = (y_c * x_c).sum(axis=1) / max((x_c ** 2).sum(), 1e-12)
    h = np.clip(slope / 2.0, 0.0, 1.0)                 # Var ~ q^(2H)
    h = np.where(np.isfinite(h), h, 0.5)
    out[window - 1:] = h
    return pd.Series(out, index=series.index)


# ---- Market benchmark cache (singleton per process) ----

_market_cache = {}
_market_cache_ts: float = 0.0
_MARKET_CACHE_TTL = 3600  # 1 hour


def _fetch_market_benchmarks(start_date, end_date) -> Dict[str, pd.DataFrame]:
    """Fetch Nifty 50, India VIX, USD/INR, and Brent Crude data, cached per process."""
    global _market_cache, _market_cache_ts
    now = _time.time()
    
    # Use a single global cache to prevent re-fetching for every unique stock date range
    cache_key = 'GLOBAL'
    
    if cache_key not in _market_cache or (now - _market_cache_ts) >= _MARKET_CACHE_TTL:
        _market_cache[cache_key] = {}
        # Fetch a massive date range once to cover all possible stocks
        fetch_start = '2005-01-01'
        fetch_end = '2030-01-01'  # yf will just return up to present
        
        def safe_fetch(ticker, name_col):
            try:
                data = yf.download(ticker, start=fetch_start, end=fetch_end, progress=False, auto_adjust=True)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data = data[['Close']].rename(columns={'Close': name_col})
                data.index = pd.to_datetime(data.index).tz_localize(None)
                return data
            except Exception as e:
                logger.warning(f"Could not fetch {ticker} data: {e}")
                return pd.DataFrame()
                
        _market_cache[cache_key]['nifty'] = safe_fetch('^NSEI', 'nifty_close')
        _market_cache[cache_key]['vix'] = safe_fetch('^INDIAVIX', 'india_vix')
        _market_cache[cache_key]['usdinr'] = safe_fetch('USDINR=X', 'usdinr')
        _market_cache[cache_key]['crude'] = safe_fetch('BZ=F', 'brent_crude')
        _market_cache_ts = now
        
    # Return sliced data for the requested date range
    result = {}
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    
    for key, df in _market_cache[cache_key].items():
        if not df.empty:
            result[key] = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()
        else:
            result[key] = df.copy()
            
    return result


class AdvancedFeatureEngine:
    """
    Production-grade feature engineering with multi-horizon extraction
    and volatility-regime normalization.
    """
    
    HORIZONS = [5, 10, 20, 50]
    
    @staticmethod
    def engineer(df: pd.DataFrame, ticker: Optional[str] = None,
                 ca_log: Optional[list] = None) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Input: DataFrame with [date, open, high, low, close, volume]
               Optionally: adj_close, delivery_qty, delivery_percentage, traded_qty
        Output: DataFrame with 150+ engineered features

        ticker: optional symbol, used only to label corporate-action log entries.
        ca_log: optional mutable list. When provided, corporate-action detections
            are appended to it as dicts instead of each being logged individually
            with logger.warning — across ~2000 tickers x 5y this previously
            produced thousands of interleaved WARNING lines that drowned out the
            tqdm progress bar and any other real signal in the training log
            (see the training-run log: ~29 minutes of feature engineering output
            dominated by these lines). The caller aggregates ca_log into ONE
            summary line after the per-ticker loop. When ca_log is None (default),
            behavior is unchanged (per-event logger.warning), so existing callers
            (e.g. the single-ticker predict() path) are unaffected.
        """
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # ======== 0. ADJUSTED CLOSE HANDLING ========
        # Use adj_close for return calculations when available (handles splits/bonuses)
        if 'adj_close' in df.columns:
            df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')
            df['adj_close'] = df['adj_close'].fillna(df['close'])
        else:
            df['adj_close'] = df['close'].copy()
        
        # ======== 0b. SPLIT/BONUS ADJUSTMENT (critical fix — see report) ========
        # Every indicator below (SMA/EMA/RSI/MACD/Bollinger/ATR/Stochastic/CCI/
        # candlesticks/OBV/VWAP/etc. — ~85 formulas) was written against `close`/
        # `open`/`high`/`low`, which are NOT split/bonus-adjusted; only log_return
        # above used adj_close. A 1:2 split or a bonus issue creates a ~50%
        # "price change" in a single day that is pure corporate-action noise, not
        # a real return — and it corrupts every rolling indicator for `window`
        # bars around the split date (this is very likely a real contributor to
        # the ">30% price change" "potential stock split" warnings seen in the
        # data-quality logs, and to calendar-feature gain concentration in the
        # trained ensembles, since split artifacts cluster non-randomly in time).
        # Fix it once, here, for the whole OHLCV block, instead of trying to
        # swap `close`->`adj_close` in 85+ individual formulas below (too easy
        # to miss some — as this codebase's history shows). Raw OHLCV is
        # restored before returning so buy price/stoploss/target/display values
        # stay real, tradable prices — only the indicator math below sees the
        # adjusted series.
        
        # Corporate action detection: flag suspicious single-stock moves > 20%
        if 'close' in df.columns and len(df) > 1:
            pct_change = df['close'].pct_change().abs()
            suspicious_mask = pct_change > 0.20
            if suspicious_mask.any():
                suspicious_indices = pct_change.index[suspicious_mask]
                if ca_log is not None:
                    for idx in suspicious_indices:
                        d = df.loc[idx, 'date'] if 'date' in df.columns else idx
                        ca_log.append({'ticker': ticker, 'date': d, 'pct_change': float(pct_change.loc[idx])})
                else:
                    for idx in suspicious_indices:
                        d = df.loc[idx, 'date'] if 'date' in df.columns else idx
                        logger.warning(f"Possible corporate action detected on {d}: {pct_change.loc[idx]:.1%} price change. Verify split/bonus adjustment.")

        _raw_open = df['open'].copy()
        _raw_high = df['high'].copy()
        _raw_low = df['low'].copy()
        _raw_close = df['close'].copy()
        _raw_volume = df['volume'].copy() if 'volume' in df.columns else None
        _adj_ratio = (df['adj_close'] / df['close'].replace(0, np.nan)).clip(0.01, 100).fillna(1.0)
        df['_pre_adjust_ratio'] = _adj_ratio  # read by _price_transforms below for the adj_ratio feature
        df['open'] = _raw_open * _adj_ratio
        df['high'] = _raw_high * _adj_ratio
        df['low'] = _raw_low * _adj_ratio
        df['close'] = df['adj_close']
        if _raw_volume is not None:
            # Volume adjusts by the reciprocal ratio (a 1:2 split roughly doubles
            # share-count volume for the same dollar-value activity) so
            # volume-based indicators (OBV, VWAP, Force Index, Amihud, MFI, CMF,
            # A/D line, VPT) don't see a matching spurious jump either.
            with np.errstate(divide='ignore', invalid='ignore'):
                _vol_ratio = (1.0 / _adj_ratio).replace([np.inf, -np.inf], 1.0).fillna(1.0)
            df['volume'] = (_raw_volume * _vol_ratio).round().astype(_raw_volume.dtype, errors='ignore')
        
        # ======== 1. PRICE TRANSFORM FEATURES ========
        df = AdvancedFeatureEngine._price_transforms(df)
        
        # ======== 2. MULTI-HORIZON MOVING AVERAGES ========
        df = AdvancedFeatureEngine._moving_averages(df)
        
        # ======== 3. MOMENTUM SUITE ========
        df = AdvancedFeatureEngine._momentum_features(df)
        
        # ======== 4. VOLATILITY FEATURES ========
        df = AdvancedFeatureEngine._volatility_features(df)
        
        # ======== 5. VOLUME ANALYSIS ========
        df = AdvancedFeatureEngine._volume_features(df)
        
        # ======== 6. OSCILLATORS ========
        df = AdvancedFeatureEngine._oscillator_features(df)
        
        # ======== 7. TREND INDICATORS ========
        df = AdvancedFeatureEngine._trend_features(df)
        
        # ======== 8. CANDLESTICK FEATURES ========
        df = AdvancedFeatureEngine._candlestick_features(df)
        
        # ======== 9. STATISTICAL FEATURES ========
        df = AdvancedFeatureEngine._statistical_features(df)
        
        # ======== 10. MARKET MICROSTRUCTURE ========
        df = AdvancedFeatureEngine._microstructure_features(df)
        
        # ======== 11. REGIME DETECTION FEATURES ========
        df = AdvancedFeatureEngine._regime_features(df)
        
        # ======== 12. TEMPORAL FEATURES ========
        df = AdvancedFeatureEngine._temporal_features(df)
        
        # ======== 13. DELIVERY / INSTITUTIONAL FEATURES ========
        df = AdvancedFeatureEngine._delivery_features(df)
        
        # ======== 14. MARKET CONTEXT FEATURES ========
        df = AdvancedFeatureEngine._market_context_features(df)
        
        # ======== 15. OU MEAN REVERSION (Pillar 2) ========
        df = AdvancedFeatureEngine._ou_reversion_features(df)
        
        # ======== 15b. INTERACTION FEATURES ========
        df = AdvancedFeatureEngine._interaction_features(df)

        # ======== 16. FNO AND EXTERNAL MARKET FEATURES ========
        df = AdvancedFeatureEngine._fno_features(df)
        df = AdvancedFeatureEngine._short_interest_features(df)
        df = AdvancedFeatureEngine._earnings_calendar_features(df)
        
        # ======== RESTORE RAW OHLCV ========
        # The adjustment above was only for indicator math; anything reading
        # open/high/low/close/volume directly off the returned dataframe
        # (display, buy-price/stoploss/target calculation, order execution)
        # needs the real, tradable prices, not the split-adjusted series.
        df['open'] = _raw_open
        df['high'] = _raw_high
        df['low'] = _raw_low
        df['close'] = _raw_close
        if _raw_volume is not None:
            df['volume'] = _raw_volume
        df = df.drop(columns=['_pre_adjust_ratio'], errors='ignore')

        # ======== CLEANUP (NO bfill — prevents look-ahead bias) ========
        df = df.replace([np.inf, -np.inf], np.nan)
        # Forward-fill only, then use column-wise median for remaining NaNs
        df = df.ffill(limit=5)
        # Imputation for LEADING NaNs (rows before an indicator's warm-up
        # completes).  The previous version filled them with the median of the
        # first 70% of rows — but those rows sit at the very START of the
        # series, so every statistic used to fill them came from their own
        # future.  An expanding median is the same idea without the leak: each
        # row only ever sees data at or before itself.  Anything still missing
        # (genuinely no prior observation) falls through to the 0.0 safety net
        # below, which is also what the downstream rolling z-score assumes.
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            _expanding_median = df[numeric_cols].expanding(min_periods=1).median()
            df[numeric_cols] = df[numeric_cols].fillna(_expanding_median)
        # Final safety net
        df = df.fillna(0.0)
        
        return df
    
    @staticmethod
    def _price_transforms(df: pd.DataFrame) -> pd.DataFrame:
        """Log returns, normalized price, gap analysis — uses adj_close for returns"""
        # Use adj_close for return computation (split/bonus adjusted)
        adj = df['adj_close'] if 'adj_close' in df.columns else df['close']
        df['log_close'] = np.log(adj.clip(lower=1e-8))
        df['log_return'] = df['log_close'].diff()
        
        # Split adjustment ratio (detects corporate actions). FIX: df['close'] is
        # now already split-adjusted (see engineer()'s adjustment block above),
        # so a plain adj/close here would be ~1.0 on every row, losing the
        # feature entirely. Use the pre-adjustment ratio stashed by engineer().
        if '_pre_adjust_ratio' in df.columns:
            df['adj_ratio'] = df['_pre_adjust_ratio']
        else:
            df['adj_ratio'] = adj / (df['close'] + 1e-10)
        
        # Price relative to moving averages
        for h in AdvancedFeatureEngine.HORIZONS:
            sma = df['close'].rolling(h).mean()
            df[f'price_to_sma_{h}'] = (df['close'] - sma) / (sma + 1e-10)
        
        # Gap analysis
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-10)
        df['gap_filled'] = ((df['low'] <= df['close'].shift(1)) & (df['gap_pct'] > 0) |
                           (df['high'] >= df['close'].shift(1)) & (df['gap_pct'] < 0)).astype(float)
                           
        # Gap vs Intraday Divergence
        df['intraday_return'] = (df['close'] - df['open']) / (df['open'] + 1e-10)
        df['gap_down_rally'] = ((df['gap_pct'] < -0.01) & (df['intraday_return'] > 0.01)).astype(float)
        df['gap_up_selloff'] = ((df['gap_pct'] > 0.01) & (df['intraday_return'] < -0.01)).astype(float)
        
        # True body
        df['true_body'] = (df['close'] - df['open']) / (df['close'] + 1e-10)
        
        # High-Low range normalized
        df['hl_range_pct'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        
        # ---- v34: 52-Week High Proximity (Pillar 1.2 — George & Hwang 2004) ----
        # Prospect-theory anchoring: investors anchor on 52-week high.
        # Breakout above proximity > 0.95 on strong volume → delayed upward drift.
        # Capitulation below 0.05 → high SELL precision.
        high_252 = df['high'].rolling(252, min_periods=60).max()
        low_252 = df['low'].rolling(252, min_periods=60).min()
        df['proximity_52w'] = (df['close'] - low_252) / (high_252 - low_252 + 1e-10)
        df['at_52w_high'] = (df['proximity_52w'] > 0.95).astype(float)
        df['at_52w_low'] = (df['proximity_52w'] < 0.05).astype(float)
        df['dist_from_52w_high'] = (high_252 - df['close']) / (high_252 + 1e-10)
        
        return df
    
    @staticmethod
    def _moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Multi-type moving averages with cross signals"""
        for h in AdvancedFeatureEngine.HORIZONS:
            df[f'sma_{h}'] = df['close'].rolling(h).mean()
            df[f'ema_{h}'] = df['close'].ewm(span=h).mean()
            
            # Slope of moving average (trend direction)
            df[f'sma_slope_{h}'] = df[f'sma_{h}'].diff(3) / (df[f'sma_{h}'].shift(3) + 1e-10)
        
        # MA crosses (short vs long)
        df['ema_cross_5_20'] = (df['ema_5'] - df['ema_20']) / (df['ema_20'] + 1e-10)
        df['ema_cross_10_50'] = (df['ema_10'] - df['ema_50']) / (df['ema_50'] + 1e-10)
        
        # Distance between EMAs
        df['ema_spread'] = (df['ema_5'] - df['ema_50']) / (df['ema_50'] + 1e-10)
        
        return df
    
    @staticmethod
    def _momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Multi-horizon momentum with acceleration"""
        for h in AdvancedFeatureEngine.HORIZONS:
            # Returns
            df[f'return_{h}d'] = df['close'].pct_change(h)
            
            # Rate of change
            df[f'roc_{h}'] = (df['close'] - df['close'].shift(h)) / (df['close'].shift(h) + 1e-10) * 100
            
            # Momentum (absolute)
            df[f'momentum_{h}'] = df['close'] - df['close'].shift(h)
        
        # Acceleration (change in momentum)
        df['momentum_accel_5'] = df['momentum_5'].diff()
        df['momentum_accel_20'] = df['momentum_20'].diff(5)
        
        # Mean reversion indicator
        for h in [20, 50]:
            mean = df['close'].rolling(h).mean()
            std = df['close'].rolling(h).std()
            df[f'zscore_{h}'] = (df['close'] - mean) / (std + 1e-10)
        
        return df
    
    @staticmethod
    def _volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Multi-horizon volatility with regime detection"""
        returns = df['close'].pct_change()
        
        for h in AdvancedFeatureEngine.HORIZONS:
            # Historical volatility
            df[f'hist_vol_{h}'] = returns.rolling(h).std() * np.sqrt(252)
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{h}'] = tr.rolling(h).mean()
            
            # Normalized ATR (for cross-stock comparison)
            df[f'natr_{h}'] = df[f'atr_{h}'] / (df['close'] + 1e-10) * 100
        
        # Volatility ratio (short/long) - indicates volatility expansion/contraction
        df['vol_ratio_5_20'] = df['hist_vol_5'] / (df['hist_vol_20'] + 1e-10)
        df['vol_ratio_10_50'] = df['hist_vol_10'] / (df['hist_vol_50'] + 1e-10)
        
        # Bollinger Bands
        for h in [20, 50]:
            sma = df['close'].rolling(h).mean()
            std = df['close'].rolling(h).std()
            df[f'bb_upper_{h}'] = sma + 2 * std
            df[f'bb_lower_{h}'] = sma - 2 * std
            df[f'bb_width_{h}'] = (4 * std) / (sma + 1e-10)
            df[f'bb_position_{h}'] = (df['close'] - df[f'bb_lower_{h}']) / (4 * std + 1e-10)
        
        # Parkinson volatility estimator (uses high-low range)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df['high'] / (df['low'] + 1e-10)) ** 2)
        ).rolling(20).mean() * np.sqrt(252)
        
        # ---- v34: Volatility Term Structure (Pillar 2.2) ----
        # Ratio of short-term to long-term vol: identifies compression vs expansion.
        # Ratio < 0.5 → Bollinger squeeze (pre-breakout). Ratio > 2.0 → high-noise regime.
        log_rets = df['close'].pct_change()
        std_10d = log_rets.rolling(10).std()
        std_30d = log_rets.rolling(30).std()
        df['vol_term_slope'] = std_10d / (std_30d + 1e-10)
        df['vol_compression'] = (df['vol_term_slope'] < 0.5).astype(float)
        df['vol_expansion'] = (df['vol_term_slope'] > 2.0).astype(float)
        
        # VCP Proxy (Volatility & Volume Contraction)
        vol_contracting = (df['vol_term_slope'] < 0.6).astype(float)
        vol_ma5 = df['volume'].rolling(5).mean()
        vol_ma20 = df['volume'].rolling(20).mean()
        volume_contracting = (vol_ma5 < vol_ma20 * 0.8).astype(float)
        df['vcp_squeeze'] = vol_contracting * volume_contracting
        
        # Volatility Skew (Downside Asymmetry)
        down_rets = log_rets.copy()
        down_rets[down_rets > 0] = 0
        up_rets = log_rets.copy()
        up_rets[up_rets < 0] = 0
        down_dev_20 = down_rets.rolling(20).std()
        up_dev_20 = up_rets.rolling(20).std()
        df['volatility_skew_20'] = (down_dev_20 / (up_dev_20 + 1e-10)).clip(lower=0.01, upper=100.0)
        
        # v34: Realized vol vs implied vol divergence (supports VRP computation)
        df['realized_vol_22d'] = log_rets.rolling(22).std() * np.sqrt(252)
        
        return df
    
    @staticmethod
    def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Volume analysis with accumulation/distribution"""
        for h in AdvancedFeatureEngine.HORIZONS:
            # Volume SMA
            df[f'vol_sma_{h}'] = df['volume'].rolling(h).mean()
            
            # Volume ratio
            df[f'vol_ratio_{h}'] = df['volume'] / (df[f'vol_sma_{h}'] + 1)
        
        # On-Balance Volume
        obv_series = []
        obv = 0
        closes = df['close'].values
        volumes = df['volume'].values
        for i in range(len(closes)):
            if i == 0:
                obv_series.append(0)
            else:
                if closes[i] > closes[i-1]:
                    obv += volumes[i]
                elif closes[i] < closes[i-1]:
                    obv -= volumes[i]
                obv_series.append(obv)
        df['obv'] = obv_series
        df['obv_sma_20'] = pd.Series(obv_series).rolling(20).mean().values
        df['obv_trend'] = (pd.Series(obv_series) - pd.Series(obv_series).rolling(20).mean()).values
        # Normalized OBV trend (relative to its own magnitude — cross-stock comparable)
        df['obv_trend_norm'] = df['obv_trend'] / (pd.Series(obv_series).rolling(20).mean().abs().values + 1e-10)
        
        # Volume-Price Trend (VPT)
        df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
        
        # Accumulation/Distribution Line
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / _rng(df)
        df['ad_line'] = (clv * df['volume']).cumsum()
        
        # Chaikin Money Flow
        for h in [10, 20]:
            df[f'cmf_{h}'] = (clv * df['volume']).rolling(h).sum() / (df['volume'].rolling(h).sum() + 1)
            
        # Institutional Buying Pressure (IBP)
        ibp = ((df['close'] - df['low']) / _rng(df)) * df['volume']
        df['inst_buying_pressure'] = ibp.rolling(5).mean() / (df['volume'].rolling(20).mean() + 1e-10)
        
        # Force Index
        df['force_index'] = df['close'].diff() * df['volume']
        df['force_index_13'] = df['force_index'].ewm(span=13).mean()
        
        # Volume-Weighted Average Price
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1)
        df['vwap_deviation'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)
        
        return df
    
    @staticmethod
    def _oscillator_features(df: pd.DataFrame) -> pd.DataFrame:
        """RSI, Stochastic, Williams %R, CCI"""
        # RSI at multiple horizons
        for h in [9, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(h).mean()
            loss = -delta.where(delta < 0, 0).rolling(h).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{h}'] = 100 - (100 / (1 + rs))
        
        # RSI divergence (price making new highs but RSI not)
        # Vectorised: `x.iloc[-1] > x.max() * 0.98` over a trailing window is
        # just a comparison against the rolling max, no per-window callback.
        _close_near_high = (df['close'] > df['close'].rolling(20).max() * 0.98).astype(float)
        _rsi_near_high = (df['rsi_14'] > df['rsi_14'].rolling(20).max() * 0.98).astype(float)
        df['rsi_divergence'] = _close_near_high - _rsi_near_high
        
        # Stochastic Oscillator
        for h in [14, 21]:
            low_min = df['low'].rolling(h).min()
            high_max = df['high'].rolling(h).max()
            df[f'stoch_k_{h}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
            df[f'stoch_d_{h}'] = df[f'stoch_k_{h}'].rolling(3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * (df['high'].rolling(14).max() - df['close']) / \
                           (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-10)
        
        # CCI
        for h in [14, 20]:
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = tp.rolling(h).mean()
            mad = _rolling_mad(tp, h)
            df[f'cci_{h}'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        
        # MFI
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        df['mfi'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
        
        return df
    
    @staticmethod
    def _trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """MACD, ADX, SuperTrend, Parabolic SAR proxy"""
        # MACD — compute both raw (for compatibility) AND normalized versions
        for fast, slow in [(12, 26), (5, 13)]:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=9).mean()
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}'] = signal
            df[f'macd_hist_{fast}_{slow}'] = macd - signal
            df[f'macd_hist_accel_{fast}_{slow}'] = (macd - signal).diff()
            # NORMALIZED versions (divided by close — cross-stock comparable)
            df[f'macd_norm_{fast}_{slow}'] = macd / (df['close'] + 1e-10)
            df[f'macd_signal_norm_{fast}_{slow}'] = signal / (df['close'] + 1e-10)
            df[f'macd_hist_norm_{fast}_{slow}'] = (macd - signal) / (df['close'] + 1e-10)
        
        # ADX
        plus_dm = df['high'].diff().clip(lower=0)
        minus_dm = (-df['low'].diff()).clip(lower=0)
        
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['di_diff'] = plus_di - minus_di
        
        # Aroon Oscillator
        for h in [14, 25]:
            df[f'aroon_up_{h}'] = _rolling_argmax_pct(df['high'], h + 1, h)
            df[f'aroon_down_{h}'] = _rolling_argmin_pct(df['low'], h + 1, h)
            df[f'aroon_osc_{h}'] = df[f'aroon_up_{h}'] - df[f'aroon_down_{h}']
        
        # Trend consistency (how often close > sma)
        for h in [20, 50]:
            sma = df['close'].rolling(h).mean()
            df[f'trend_consistency_{h}'] = (df['close'] > sma).rolling(20).mean()
            
        # Fractal Trend Confluence
        sma_5_slope = df['close'].rolling(5).mean().diff(3)
        sma_20_slope = df['close'].rolling(20).mean().diff(5)
        sma_50_slope = df['close'].rolling(50).mean().diff(10)
        bull_confluence = ((sma_5_slope > 0) & (sma_20_slope > 0) & (sma_50_slope > 0)).astype(float)
        bear_confluence = ((sma_5_slope < 0) & (sma_20_slope < 0) & (sma_50_slope < 0)).astype(float)
        df['fractal_trend_confluence'] = bull_confluence - bear_confluence
        
        return df
    
    @staticmethod
    def _candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
        """Quantitative candlestick features"""
        # Body metrics
        df['body_size'] = np.abs(df['close'] - df['open']) / (df['close'] + 1e-10)
        df['body_direction'] = np.sign(df['close'] - df['open'])
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['close'] + 1e-10)
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['close'] + 1e-10)
        df['wick_ratio'] = df['upper_wick'] / (df['lower_wick'] + 1e-10)
        
        # Body relative to range
        df['body_to_range'] = np.abs(df['close'] - df['open']) / _rng(df)
        
        # Consecutive direction count
        direction = np.sign(df['close'] - df['open'])
        consecutive = []
        count = 0
        for d in direction:
            if len(consecutive) == 0:
                count = 1
            elif d == direction.iloc[len(consecutive) - 1]:
                count += 1
            else:
                count = 1
            consecutive.append(count * d)
        df['consecutive_candles'] = consecutive
        
        # Average body size ratio (current vs average)
        avg_body = df['body_size'].rolling(20).mean()
        df['body_size_ratio'] = df['body_size'] / (avg_body + 1e-10)
        
        return df
    
    @staticmethod
    def _statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Distribution statistics of returns"""
        returns = df['close'].pct_change()
        
        for h in [10, 20, 50]:
            # Skewness
            df[f'skew_{h}'] = returns.rolling(h).skew()
            
            # Kurtosis
            df[f'kurtosis_{h}'] = returns.rolling(h).kurt()
            
            # Position within the window's range (the name says percentile rank;
            # the formula is and always was a min-max position, kept as-is for
            # backward compatibility of the feature's meaning).
            df[f'pct_rank_{h}'] = _rolling_minmax_position(df['close'], h)

        # Auto-correlation
        df['autocorr_5'] = _rolling_autocorr(returns, 20, lag=5)

        # Hurst exponent (regime indicator) — window widened to 100 because the
        # previous 20-row window could not support ANY multi-scale estimator.
        df['hurst_proxy'] = AdvancedFeatureEngine._hurst_proxy(returns, 100)
        
        return df
    
    @staticmethod
    def _hurst_proxy(returns: pd.Series, window: int = 100) -> pd.Series:
        """Variance-ratio Hurst exponent.

        REPLACES the previous DFA implementation, which never produced a value.
        It chose box sizes via
            np.logspace(log10(4), log10(max(len(y)//4, 5)), num=8).astype(int)
        With the caller passing window=20 that is logspace(log10 4, log10 5),
        whose integer cast is {4, 5} — only two distinct sizes, which tripped
        the `len(box_sizes) < 3` guard and returned the hard-coded fallback 0.5.
        So `hurst_proxy` was a constant column for every row of every ticker:
        it contributed nothing to the model while costing a full
        rolling().apply() pass over the entire dataset, and it would have been
        silently removed by the variance / IC filters in MLPredictor anyway.

        H < 0.5 mean-reverting, H = 0.5 random walk, H > 0.5 trending.
        """
        return _rolling_hurst_vr(returns, max(int(window), 20))

    @staticmethod
    def _microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure and liquidity features"""
        # FIX (real bug, found in review — plausible root cause of the periodic
        # "Non-finite gradient norm" training warnings): amihud/kyle_lambda divide
        # by volume (or log(volume+1)) with only a 1e-10 epsilon. On a zero- or
        # near-zero-volume day (illiquid small-caps, trading halts, listing-day
        # rows, F&O ban days) this doesn't produce inf (the epsilon prevents true
        # division-by-zero), but it does produce a value up to ~1e9x a normal
        # session's — larger than the training-time global [1st,99th] percentile
        # winsorization bound was fit to see if that exact ticker/date combination
        # wasn't in the percentile-estimation sample (it's fit on a SAMPLE of
        # rows, not every row — see MLPredictor.py's "Fit feature scaler with
        # winsorization" block). A single such row surviving into a batch is
        # enough to blow up a mixed-precision (fp16) forward pass. Clip at the
        # formula level, in addition to (not instead of) the global winsorizer,
        # as defense-in-depth: this bounds the value before it ever reaches
        # scaling, regardless of whether that day made it into the percentile
        # sample. Bounds are generous (99.9th-pctile-scale for liquid NSE
        # large/mid-caps) so they only clip genuine near-zero-volume outliers,
        # not ordinary illiquidity variation.
        _volume_safe = df['volume'].clip(lower=1)  # avoid rewarding literal 0-volume rows with the max score
        _ret_abs = np.abs(df['close'].pct_change())
        df['amihud'] = (_ret_abs / (_volume_safe * df['close'] + 1e-10)).clip(upper=1e6)
        df['amihud_20'] = df['amihud'].rolling(20).mean()
        
        # High-Low spread (proxy for bid-ask spread)
        df['hl_spread'] = (df['high'] - df['low']) / ((df['high'] + df['low']) / 2 + 1e-10)
        
        # Kyle's Lambda proxy (price impact)
        df['kyle_lambda'] = (_ret_abs / (np.log(_volume_safe + 1) + 1e-10)).clip(upper=1e3)
        
        # Volume clock (volume-weighted time)
        df['vol_clock'] = df['volume'] / (df['volume'].rolling(20).mean() + 1)
        
        # v51: OFI (Order Flow Imbalance) Proxy
        # Infers buying/selling pressure from intra-bar price action relative to volume
        # v72 FIX: ofi_proxy = (close-location-in-range) * volume is the SAME shape as
        # amihud/kyle_lambda above (a bounded ratio multiplied by raw, unbounded volume) —
        # the exact pattern already identified as the plausible root cause of the periodic
        # "Non-finite gradient norm" training warnings. It was never given the matching
        # clip(), so a high-volume session (large-cap liquid names, or any stock on a
        # volume-spike day) can still produce a value orders of magnitude outside the
        # global winsorization sample. Clip at the formula level, same as amihud/kyle_lambda,
        # as defense-in-depth. ofi_proxy_20 (already volume-normalized below) is unaffected.
        buy_pressure = (df['close'] - df['low']) / _rng(df)
        sell_pressure = (df['high'] - df['close']) / _rng(df)
        df['ofi_proxy'] = ((buy_pressure - sell_pressure) * df['volume']).clip(lower=-1e9, upper=1e9)
        df['ofi_proxy_20'] = df['ofi_proxy'].rolling(20).mean() / (df['volume'].rolling(20).mean() + 1e-10)
        
        # Price efficiency (close-to-close vs high-low)
        df['price_efficiency'] = np.abs(df['close'] - df['close'].shift(1)) / _rng(df)
        
        return df
    
    @staticmethod
    def _regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Market regime detection features"""
        returns = df['close'].pct_change()
        
        # Volatility regime (current vs long-term)
        short_vol = returns.rolling(10).std()
        long_vol = returns.rolling(50).std()
        df['vol_regime'] = short_vol / (long_vol + 1e-10)
        
        # Trend regime (ADX-based)
        if 'adx' in df.columns:
            df['trending'] = (df['adx'] > 25).astype(float)
        
        # Mean reversion signal
        for h in [20, 50]:
            if f'zscore_{h}' in df.columns:
                df[f'mean_revert_signal_{h}'] = -df[f'zscore_{h}'].clip(-3, 3) / 3
        
        # Momentum regime
        df['mom_regime'] = np.where(
            df['close'].pct_change(20) > 0.05, 1,
            np.where(df['close'].pct_change(20) < -0.05, -1, 0)
        )
        
        return df
    
    @staticmethod
    def _temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calendar and temporal features"""
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            df['day_of_week'] = dates.dt.dayofweek / 4.0  # Normalize to 0-1
            df['month'] = dates.dt.month / 12.0
            df['quarter'] = dates.dt.quarter / 4.0
            df['day_of_month'] = dates.dt.day / 31.0
            
            # Month-end effect
            df['is_month_end'] = dates.dt.is_month_end.astype(float)
            df['is_month_start'] = dates.dt.is_month_start.astype(float)
            
            # Week of year (seasonality)
            df['week_of_year'] = dates.dt.isocalendar().week.astype(float).values / 52.0
            
            # Expiry proximity (last Thursday of month — options expiry)
            def _days_to_expiry(dt):
                import calendar
                from dateutil.relativedelta import relativedelta
                last_day = calendar.monthrange(dt.year, dt.month)[1]
                last_date = pd.Timestamp(dt.year, dt.month, last_day)
                # Find last Thursday
                _wd = 1 if dt >= pd.Timestamp('2025-09-01') else 3      # Tuesday expiries after the 2025 change
                while last_date.dayofweek != _wd:
                    last_date -= pd.Timedelta(days=1)
                
                days_left = (last_date - dt).days
                if days_left < 0:
                    # Move to next month
                    next_month = dt + relativedelta(months=1)
                    last_day = calendar.monthrange(next_month.year, next_month.month)[1]
                    last_date = pd.Timestamp(next_month.year, next_month.month, last_day)
                    _wd = 1 if dt >= pd.Timestamp('2025-09-01') else 3      # Tuesday expiries after the 2025 change
                    while last_date.dayofweek != _wd:
                        last_date -= pd.Timedelta(days=1)
                    days_left = (last_date - dt).days
                    
                return max(days_left, 0)
            # `dates.apply` invoked a python closure per row per ticker. There
            # are only a few thousand distinct trading dates across the whole
            # universe, so compute each one once and map.
            _uniq = pd.Index(dates.dropna().unique())
            _lookup = {d: _days_to_expiry(d) for d in _uniq}
            df['days_to_expiry'] = dates.map(_lookup).astype(float) / 30.0
        
        return df
    
    @staticmethod
    def _delivery_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Delivery-based features — institutional activity signals.
        Uses delivery_qty, delivery_percentage, traded_qty from DB.
        
        v34 Enhancement (Pillar 1 — Factor Model Alpha):
        Delivery-based Order Flow Imbalance (OFI) — Chordia, Roll &
        Subrahmanyam (2000).  NSE delivery data is a direct window into
        institutional conviction that no purely price-based indicator replicates.
        """
        has_delivery = 'delivery_percentage' in df.columns or 'delivery_qty' in df.columns
        if not has_delivery:
            return df
        
        # Delivery percentage features
        if 'delivery_percentage' in df.columns:
            dp = pd.to_numeric(df['delivery_percentage'], errors='coerce').fillna(0)
            df['delivery_pct'] = dp / 100.0
            for h in AdvancedFeatureEngine.HORIZONS:
                df[f'delivery_pct_sma_{h}'] = df['delivery_pct'].rolling(h).mean()
            # Delivery spike (current vs 20-day avg)
            dp_sma = df['delivery_pct'].rolling(20).mean()
            df['delivery_spike'] = (df['delivery_pct'] - dp_sma) / (dp_sma + 1e-10)
            # High delivery + bullish price = institutional buying
            df['inst_accumulation'] = df['delivery_pct'] * np.sign(df['close'].pct_change())
            df['inst_accumulation_5'] = df['inst_accumulation'].rolling(5).mean()
            
            # ---- v34: Delivery OFI (Order Flow Imbalance) ----
            # OFI = delivery_pct × close × volume → captures conviction-weighted flow
            ofi_raw = df['delivery_pct'] * df['close'] * df['volume']
            for h in [5, 20]:
                ofi_mean = ofi_raw.rolling(h).mean()
                ofi_std = ofi_raw.rolling(h).std()
                df[f'ofi_zscore_{h}d'] = (ofi_raw - ofi_mean) / (ofi_std + 1e-10)
            
            # Delivery percentile rank within 20-day window
            df['delivery_pct_rank_20'] = _rolling_minmax_position(df['delivery_pct'], 20)
            
            # Delivery 90th percentile spike (Pillar 1.1 — institutional conviction)
            dp_p90 = df['delivery_pct'].rolling(60, min_periods=20).quantile(0.90)
            df['delivery_above_p90'] = (df['delivery_pct'] > dp_p90).astype(float)
            
            # OFI near support confluence (spike + price near 20d low)
            low_20 = df['close'].rolling(20).min()
            near_support = (df['close'] - low_20) / (df['close'] + 1e-10) < 0.03
            df['ofi_near_support'] = (df['delivery_above_p90'] * near_support.astype(float))
            
            # Conviction-weighted flow direction
            df['delivery_conviction'] = (df['delivery_pct'] *
                                          np.abs(df['close'].pct_change()) *
                                          df['volume'] / (df['volume'].rolling(20).mean() + 1))
            df['delivery_conviction_dir'] = df['delivery_conviction'] * np.sign(df['close'].pct_change())
            df['delivery_conviction_dir_5d'] = df['delivery_conviction_dir'].rolling(5).mean()
        
        # Delivery volume features
        if 'delivery_qty' in df.columns:
            dq = pd.to_numeric(df['delivery_qty'], errors='coerce').fillna(0)
            df['delivery_qty_log'] = np.log1p(dq)
            # Delivery to traded ratio
            if 'traded_qty' in df.columns:
                tq = pd.to_numeric(df['traded_qty'], errors='coerce').fillna(1)
                df['delivery_to_traded'] = dq / (tq + 1)
                df['delivery_to_traded_sma_10'] = df['delivery_to_traded'].rolling(10).mean()
                df['delivery_to_traded_momentum_5d'] = df['delivery_to_traded'].pct_change(5)

        # ---- v50: EOD flow/microstructure proxies (intraday-book independent) ----
        # 1) Flow persistence: sustained conviction-weighted directional delivery.
        if 'delivery_conviction_dir_5d' in df.columns:
            flow = pd.Series(df['delivery_conviction_dir_5d'].values)
            flow_mean = flow.rolling(20).mean()
            flow_std = flow.rolling(20).std()
            df['delivery_flow_persistence'] = ((flow - flow_mean) / (flow_std + 1e-10)).clip(-5, 5).values

        # 2) Effort vs result imbalance: high volume effort with muted price expansion.
        vol_effort = df['volume'] / (df['volume'].rolling(20).mean() + 1)
        price_result = (df['high'] - df['low']) / (df['close'] + 1e-10)
        eri_raw = vol_effort / (price_result + 1e-6)
        eri_mean = eri_raw.rolling(20).mean()
        eri_std = eri_raw.rolling(20).std()
        df['effort_result_imbalance'] = ((eri_raw - eri_mean) / (eri_std + 1e-10)).clip(-5, 5)

        # 3) Closing pressure + liquidity interaction.
        close_location = (df['close'] - df['low']) / _rng(df)
        df['closing_pressure'] = close_location.clip(0, 1)
        df['closing_pressure_5d'] = df['closing_pressure'].rolling(5).mean()
        if 'amihud_20' in df.columns:
            liquidity_weight = 1.0 / (1.0 + np.abs(df['amihud_20']) * 1e6)
            df['closing_pressure_liquidity'] = (df['closing_pressure'] * liquidity_weight).clip(0, 1)
        else:
            df['closing_pressure_liquidity'] = df['closing_pressure']
        
        return df
    
    @staticmethod
    def _market_context_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Market-wide context features — Nifty 50 returns, relative strength, VIX.
        These provide macro regime context that individual stock data lacks.
        """
        if 'date' not in df.columns:
            return df
        
        dates = pd.to_datetime(df['date'])
        start_date = dates.min() - pd.Timedelta(days=60)  # extra buffer
        end_date = dates.max() + pd.Timedelta(days=5)
        
        try:
            mkt = _fetch_market_benchmarks(str(start_date.date()), str(end_date.date()))
        except Exception as e:
            logger.warning(f"Market context features skipped: {e}")
            return df
        
        # ---- Nifty 50 features ----
        nifty_df = mkt.get('nifty', pd.DataFrame())
        if not nifty_df.empty:
            df_dates = dates.dt.normalize()
            nifty_aligned = nifty_df.reindex(df_dates, method='ffill').fillna(0.0)
            nifty_close = nifty_aligned['nifty_close'].values
            
            df['nifty_return_1d'] = pd.Series(nifty_close).pct_change().values
            for h in [5, 10, 20]:
                df[f'nifty_return_{h}d'] = pd.Series(nifty_close).pct_change(h).values
                df[f'nifty_vol_{h}'] = pd.Series(nifty_close).pct_change().rolling(h).std().values
            
            # Relative strength vs Nifty (stock outperformance)
            stock_ret_20 = df['close'].pct_change(20)
            nifty_ret_20 = pd.Series(nifty_close).pct_change(20)
            df['relative_strength_20'] = (stock_ret_20.values - nifty_ret_20.values)
            
            stock_ret_5 = df['close'].pct_change(5)
            nifty_ret_5 = pd.Series(nifty_close).pct_change(5)
            df['relative_strength_5'] = (stock_ret_5.values - nifty_ret_5.values)

            # v50: Relative-strength persistence (stabilizes noisy short-horizon flips).
            rs20 = pd.Series(df['relative_strength_20'].values)
            rs20_trend = rs20.rolling(5).mean()
            rs20_vol = rs20.rolling(20).std()
            df['relative_strength_persistence'] = ((rs20_trend) / (rs20_vol + 1e-10)).clip(-5, 5).values

            # v50: Delivery flow divergence vs benchmark drift (EOD graph-context proxy).
            if 'delivery_conviction_dir_5d' in df.columns:
                flow_series = pd.Series(df['delivery_conviction_dir_5d'].values)
                market_proxy = pd.Series(df['nifty_return_5d'].values)
                flow_spread = flow_series - market_proxy
                spread_mean = flow_spread.rolling(20).mean()
                spread_std = flow_spread.rolling(20).std()
                df['flow_market_divergence_20'] = ((flow_spread - spread_mean) / (spread_std + 1e-10)).clip(-5, 5).values
            
            # Beta (rolling 60-day)
            stock_rets = df['close'].pct_change()
            nifty_rets = pd.Series(nifty_close).pct_change()
            cov = stock_rets.rolling(60).cov(nifty_rets)
            nifty_var = nifty_rets.rolling(60).var()
            df['beta_60'] = (cov / (nifty_var + 1e-10)).values
        
        # ---- India VIX features ----
        vix_df = mkt.get('vix', pd.DataFrame())
        if not vix_df.empty:
            df_dates = dates.dt.normalize()
            vix_aligned = vix_df.reindex(df_dates, method='ffill').fillna(0.0)
            vix_vals = vix_aligned['india_vix'].values
            
            df['india_vix'] = vix_vals / 100.0  # Normalize
            df['india_vix_change'] = pd.Series(vix_vals).pct_change().values
            df['india_vix_sma_10'] = pd.Series(vix_vals).rolling(10).mean().values / 100.0
            # VIX regime (high fear vs low complacency)
            vix_20_mean = pd.Series(vix_vals).rolling(20).mean()
            df['vix_regime'] = (pd.Series(vix_vals) / (vix_20_mean + 1e-10)).values
            
            # ---- v34: Volatility Risk Premium (Pillar 2 — Bollerslev et al. 2009) ----
            # VRP = India VIX - realized vol.  Positive VRP → fear premium decaying,
            # predicts positive 5-10d returns as vol premium reverts.
            if 'realized_vol_22d' in df.columns:
                india_vix_annual = pd.Series(vix_vals).values / 100.0  # VIX is already annualized
                df['vrp'] = india_vix_annual - df['realized_vol_22d'].values
                vrp_series = pd.Series(df['vrp'].values)
                vrp_mean = vrp_series.rolling(60).mean()
                vrp_std = vrp_series.rolling(60).std()
                df['vrp_zscore'] = ((vrp_series - vrp_mean) / (vrp_std + 1e-10)).values
            
            # ---- v34: India VIX Term Structure Proxy (Pillar 3.3) ----
            # Short-term vix change vs long-term: inverted curve → immediate danger
            vix_series = pd.Series(vix_vals)
            vix_10d_avg = vix_series.rolling(10).mean()
            vix_30d_avg = vix_series.rolling(30).mean()
            df['vix_term_slope'] = (vix_10d_avg / (vix_30d_avg + 1e-10)).values
            df['vix_inverted'] = (df['vix_term_slope'] > 1.0).astype(float)
        
        # ---- v34: Idiosyncratic Volatility (Pillar 1.3 — Ang et al. 2006) ----
        # IVOL = residual vol after regressing on Nifty.  High IVOL stocks
        # systematically underperform (lottery effect).
        if not nifty_df.empty and 'beta_60' in df.columns:
            stock_rets_daily = df['close'].pct_change()
            nifty_rets_daily = pd.Series(nifty_close).pct_change()
            # Residual return = stock_ret - beta * nifty_ret
            residual_ret = stock_rets_daily.values - df['beta_60'].values * nifty_rets_daily.values
            residual_series = pd.Series(residual_ret)
            df['ivol_22d'] = residual_series.rolling(22).std().values * np.sqrt(252)
            # IVOL rank vs own history (bounded 0-1)
            ivol_s = pd.Series(df['ivol_22d'].values)
            ivol_min = ivol_s.rolling(252, min_periods=60).min()
            ivol_max = ivol_s.rolling(252, min_periods=60).max()
            df['ivol_rank'] = ((ivol_s - ivol_min) / (ivol_max - ivol_min + 1e-10)).values
        
        # ---- v34: USD/INR Macro Feature (Pillar 3.2) ----
        usdinr_df = mkt.get('usdinr', pd.DataFrame())
        if not usdinr_df.empty:
            df_dates = dates.dt.normalize()
            usdinr_aligned = usdinr_df.reindex(df_dates, method='ffill').fillna(0.0)
            usdinr_s = usdinr_aligned['usdinr']
            if isinstance(usdinr_s, pd.DataFrame):
                usdinr_s = usdinr_s.iloc[:, 0]
            usdinr_vals = usdinr_s.values.ravel()
            df['usdinr_change_5d'] = pd.Series(usdinr_vals).pct_change(5).values
            df['usdinr_change_20d'] = pd.Series(usdinr_vals).pct_change(20).values
        
        # ---- v34: Brent Crude Oil Macro Feature (Pillar 3.2) ----
        crude_df = mkt.get('crude', pd.DataFrame())
        if not crude_df.empty:
            df_dates = dates.dt.normalize()
            crude_aligned = crude_df.reindex(df_dates, method='ffill').fillna(0.0)
            crude_s = crude_aligned['brent_crude']
            if isinstance(crude_s, pd.DataFrame):
                crude_s = crude_s.iloc[:, 0]
            crude_vals = crude_s.values.ravel()
            df['crude_change_5d'] = pd.Series(crude_vals).pct_change(5).values
            df['crude_change_20d'] = pd.Series(crude_vals).pct_change(20).values
        
        # ---- v34: Sector Breadth Proxy (Pillar 3.4) ----
        # Nifty advance-decline proxy: is Nifty above its own 50d SMA?
        # Broad market support context for individual stock signals.
        if not nifty_df.empty:
            nifty_series = pd.Series(nifty_close)
            nifty_sma50 = nifty_series.rolling(50).mean()
            df['nifty_above_sma50'] = (nifty_series > nifty_sma50).astype(float).values
            # Percentage of time Nifty was above 50d SMA in last 20 days
            df['breadth_20d'] = pd.Series(df['nifty_above_sma50'].values).rolling(20).mean().values
        
        return df
    
    @staticmethod
    def _ou_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        v34: Ornstein-Uhlenbeck Mean Reversion Speed (θ) — Pillar 2.1
        
        Estimates per-stock mean reversion speed via MLE:
          θ = -log(autocorr(1)) / dt
        
        High θ → structural mean-reverter (reversal features most valuable).
        Low θ → trending stock (momentum features most valuable).
        Halflife = log(2) / θ — expected days to revert to mean.
        """
        # FIX (wrong variable + degenerate output): the previous code estimated
        # theta = -log(autocorr(1)) from the *return* series and returned 0.0
        # whenever that autocorrelation was <= 0.  Two problems compound here:
        #
        #   1. Ornstein-Uhlenbeck describes mean reversion of a LEVEL, so the
        #      AR(1) coefficient has to be estimated on the (log) price, not on
        #      its first difference.  Applied to returns, a positive coefficient
        #      actually indicates momentum, i.e. the estimator's sign convention
        #      was inverted relative to the feature's stated meaning.
        #   2. Daily equity returns have slightly NEGATIVE lag-1 autocorrelation
        #      almost all of the time, so the `ac <= 0 -> return 0.0` branch
        #      fired on the large majority of windows.  theta == 0 then made
        #      ou_halflife = log(2)/1e-10, clipped to the 250 ceiling, and
        #      ou_reversion_strength = 0/0 -> 0.  All three "OU" features were
        #      therefore near-constant — including 'ou_reversion_strength',
        #      which CONFIG lists as a cross-sectional rank feature.
        #
        # Now: rolling AR(1) on log price, theta = -log(phi), phi in (0, 1).
        # phi >= 1 (random walk or explosive) correctly yields theta -> 0.
        _log_price = np.log(pd.to_numeric(df['close'], errors='coerce').clip(lower=1e-8))
        _phi = _rolling_ar1(_log_price, 60)
        _phi = _phi.clip(lower=1e-6, upper=0.999999)
        df['ou_theta_60d'] = (-np.log(_phi)).fillna(0.0)

        # Halflife in trading days
        df['ou_halflife'] = np.log(2) / (df['ou_theta_60d'] + 1e-10)
        # Clamp halflife to reasonable range (1-250 days)
        df['ou_halflife'] = df['ou_halflife'].clip(1, 250)
        
        # Mean reversion strength indicator (bounded 0-1)
        # Higher theta = stronger mean reversion = higher score
        theta_s = pd.Series(df['ou_theta_60d'].values)
        theta_max = theta_s.rolling(252, min_periods=60).max()
        df['ou_reversion_strength'] = (theta_s / (theta_max + 1e-10)).values
        
        return df

    @staticmethod
    def _interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """v76: Domain-motivated interaction features capturing regime-context signals."""
        # RSI × volatility regime: captures regime-context reversal signals
        if 'rsi_14' in df.columns and 'vol_regime' in df.columns:
            df['rsi_vol_regime'] = df['rsi_14'] * df['vol_regime']
        
        # Delivery conviction × momentum: institutional flow + momentum confluence
        if 'delivery_conviction' in df.columns and 'return_5d' in df.columns:
            df['delivery_momentum'] = df['delivery_conviction'] * df['return_5d']
        
        # VIX z-score × RSI deviation: fear × oversold/overbought
        if 'india_vix' in df.columns and 'rsi_14' in df.columns:
            _vix_zscore = (df['india_vix'] - df['india_vix'].rolling(20, min_periods=5).mean()) / df['india_vix'].rolling(20, min_periods=5).std().replace(0, 1)
            _rsi_dev = (df['rsi_14'] - 50) / 50  # Normalize to [-1, 1]
            df['vix_rsi_cross'] = _vix_zscore.clip(-3, 3) * _rsi_dev
        
        # ADX × trend consistency: strong trend + consistent direction
        if 'adx' in df.columns and 'trend_consistency_20' in df.columns:
            df['adx_trend_strength'] = (df['adx'] / 100) * df['trend_consistency_20']
        
        # Volume spike × price momentum: volume confirms price move
        if 'vol_ratio_5' in df.columns and 'return_5d' in df.columns:
            df['volume_price_confirm'] = df['vol_ratio_5'].clip(0, 5) * np.sign(df['return_5d'])
        
        return df

    @staticmethod
    def _fno_features(df: pd.DataFrame) -> pd.DataFrame:
        """Futures and Options derived features. Attempts to pull from Postgres;
        uses a neutral (non-fabricated) placeholder when unavailable."""
        df = df.copy()
        import logging
        logger = logging.getLogger(__name__)

        got_fii, got_opt = False, False
        try:
            from sqlalchemy import create_engine, text
            from IntegratedPostGreSQL import Config
            engine = create_engine(Config.DB_URL)
            ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else None

            with engine.connect() as conn:
                # FII/DII Flow
                fii_df = pd.read_sql(text("SELECT d AS date, fii_net AS fii_dii_net_flow FROM fii_dii ORDER BY d"), conn)
                if not fii_df.empty:
                    fii_df['date'] = pd.to_datetime(fii_df['date']).shift(-1)      # flow of day d usable from the NEXT session
                    fii_df = fii_df.dropna(subset=['date'])
                    df = df.merge(fii_df, on='date', how='left')
                    got_fii = True

                if ticker:
                    # Options Metrics — FIX: parameterized query instead of an
                    # f-string-interpolated ticker (defense-in-depth; ticker is
                    # internally sourced today, but this closes the pattern off).
                    opt_df = pd.read_sql(
                        text("SELECT date, put_call_ratio as pcr_skew, iv_term_structure_slope "
                             "FROM options_metrics WHERE ticker = :ticker"),
                        conn, params={"ticker": ticker}
                    )
                    if not opt_df.empty:
                        opt_df['date'] = pd.to_datetime(opt_df['date'])
                        df = df.merge(opt_df, on='date', how='left')
                        got_opt = True
        except Exception as e:
            logger.debug(f"FNO data fetch unavailable, using neutral placeholders: {e}")

        # FIX (real bug — data fabrication): this used to fill missing real data
        # with np.random noise (and reseed the GLOBAL numpy RNG via
        # np.random.seed(42) on every call, silently making any other code that
        # relies on np.random's global state deterministic/correlated with this
        # call). Since IntegratedPostGreSQL.py never populates fii_dii_flow or
        # options_metrics, this fallback path runs on essentially every ticker,
        # every time — meaning the model was trained on pure random noise
        # dressed up as real institutional-flow/options signals, with nothing
        # downstream aware of it. Use a neutral, zero-variance constant instead
        # (no fabricated pattern for a tree ensemble to spuriously key off of),
        # and flag availability explicitly so consumers (and the leakage-gain
        # audit in MLPredictor.py) can identify and exclude these when absent.
        if 'futures_basis' not in df.columns:
            df['futures_basis'] = 0.0
        if 'pcr_skew' not in df.columns:
            df['pcr_skew'] = 1.0  # 1.0 = balanced put/call, the neutral value for this ratio
        if 'iv_term_structure_slope' not in df.columns:
            df['iv_term_structure_slope'] = 0.0  # 0.0 = flat term structure
        if 'fii_dii_net_flow' not in df.columns:
            df['fii_dii_net_flow'] = 0.0
        df['fno_data_available'] = float(got_fii or got_opt)

        # Forward fill any real-but-sparse fetched data (does not apply to the
        # neutral constants above, which have no gaps to fill)
        df['pcr_skew'] = df['pcr_skew'].ffill(limit=5).fillna(1.0)
        df['iv_term_structure_slope'] = df['iv_term_structure_slope'].ffill(limit=5).fillna(0.0)
        df['fii_dii_net_flow'] = df['fii_dii_net_flow'].ffill(limit=5).fillna(0.0)

        df['futures_roll_cost'] = df['futures_basis'].rolling(5).mean()
        df['pcr_skew_momentum'] = df['pcr_skew'] - df['pcr_skew'].shift(5)
        df['iv_term_structure_slope_sma10'] = df['iv_term_structure_slope'].rolling(10).mean()
        df['fii_dii_flow_trend'] = df['fii_dii_net_flow'].rolling(5).mean()
        return df
        
    @staticmethod
    def _short_interest_features(df: pd.DataFrame) -> pd.DataFrame:
        """Engineer short-interest proxies from NSE ban list and margin shortfall.
        Uses a neutral (non-fabricated) placeholder when the real data isn't available."""
        df = df.copy()
        got_ban = False
        try:
            from sqlalchemy import create_engine, text
            from IntegratedPostGreSQL import Config
            engine = create_engine(Config.DB_URL)
            ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else None

            if ticker:
                with engine.connect() as conn:
                    # FIX: parameterized query instead of f-string interpolation.
                    ban_df = pd.read_sql(
                        text("SELECT date, 1 as in_fno_ban FROM fo_ban_list WHERE ticker = :ticker"),
                        conn, params={"ticker": ticker}
                    )
                    if not ban_df.empty:
                        ban_df['date'] = pd.to_datetime(ban_df['date'])
                        df = df.merge(ban_df, on='date', how='left')
                        got_ban = True
        except Exception:
            pass

        # FIX (real bug — data fabrication, see _fno_features above): in_fno_ban
        # was randomly sampled at a fake 5% positive rate, and margin_shortfall_count
        # from a fake Poisson(1) — neither reflects this ticker's real history.
        # "Not banned" / "no shortfall" (0) is both the neutral value AND the
        # overwhelmingly common real case, so it's a safe, honest default —
        # unlike random sampling, which fabricates a specific fake base rate.
        if 'in_fno_ban' not in df.columns:
            df['in_fno_ban'] = 0.0
        else:
            df['in_fno_ban'] = df['in_fno_ban'].fillna(0)
        if 'margin_shortfall_count' not in df.columns:
            df['margin_shortfall_count'] = 0.0
        df['short_interest_data_available'] = float(got_ban)

        df['ban_list_persistence'] = df['in_fno_ban'].rolling(5).sum()
        df['margin_shortfall_trend'] = df['margin_shortfall_count'].rolling(10).mean()
        # Forced liquidation proxy: High margin shortfall + decreasing price
        df['forced_liquidation_proxy'] = df['margin_shortfall_trend'] * (df['close'] < df['close'].shift(5)).astype(int)
        return df
        
    @staticmethod
    def _earnings_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """Earnings calendar distance features (PEAD). No real earnings-date
        ingestion exists anywhere in this pipeline yet (unlike FNO/short-interest,
        there isn't even a DB table for it) — see report. Uses an explicit
        out-of-band sentinel rather than a fabricated calendar so nothing
        downstream mistakes it for real data."""
        # FIX (real bug — data fabrication): this used `np.arange(len(df)) % 90`,
        # a perfectly regular sawtooth keyed purely on row position, NOT any real
        # earnings date — dressed up as "days_to_next_earnings". A deterministic,
        # regularly-periodic fake signal is worse than random noise: it's exactly
        # the kind of pattern a tree ensemble can and will find spurious
        # structure in (this is a plausible contributor to the calendar-feature
        # gain concentration flagged by MLPredictor.py's leakage guard). Use a
        # sentinel (-1, never a real day-count) plus an availability flag instead
        # of fabricating a fake cycle. When real earnings-date ingestion exists
        # (a new DB table + fetch, mirroring _fno_features above), wire it in here.
        if 'days_to_next_earnings' not in df.columns:
            df['days_since_last_earnings'] = -1.0
            df['days_to_next_earnings'] = -1.0
            df['earnings_calendar_data_available'] = 0.0
        else:
            df['earnings_calendar_data_available'] = 1.0
        return df