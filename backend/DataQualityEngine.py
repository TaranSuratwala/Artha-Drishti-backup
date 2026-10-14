import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, List, Any
from collections import defaultdict
import json
import os
from scipy import stats

logger = logging.getLogger('artha_drishti.data_quality')

class DataQualityEngine:
    """
    Pre-training data quality pipeline.
    
    Validates raw OHLCV market data and applies cleaning steps:
    1. Corporate action detection & flagging (splits, bonuses, suspensions)
    2. Outlier winsorization (per-ticker, per-feature, IQR-based)
    3. Stale/insufficient data removal  
    4. Cross-sectional consistency checks
    5. Feature stability reporting (PSI between rolling windows)
    """
    
    def __init__(self, 
                 min_trading_days: int = 252,
                 min_price: float = 5.0,
                 max_daily_return: float = 0.25,
                 max_volume_zscore: float = 4.0,
                 iqr_multiplier: float = 3.0,
                 max_consecutive_identical: int = 3,
                 max_missing_volume_pct: float = 20.0,
                 psi_warning_threshold: float = 0.25,
                 gap_threshold_pct: float = 20.0,
                 volume_drop_threshold: float = 0.05,
                 volume_drop_days: int = 5,
                 clip_ohlc: bool = False,
                 metrics_dir: str = 'unified_metrics'):
        self.min_trading_days = min_trading_days
        self.min_price = min_price
        self.max_daily_return = max_daily_return
        self.max_volume_zscore = max_volume_zscore
        self.iqr_multiplier = iqr_multiplier
        self.max_consecutive_identical = max_consecutive_identical
        self.max_missing_volume_pct = max_missing_volume_pct
        self.psi_warning_threshold = psi_warning_threshold
        self.gap_threshold_pct = gap_threshold_pct
        self.volume_drop_threshold = volume_drop_threshold
        self.volume_drop_days = volume_drop_days
        # clip_ohlc: per-ticker FULL-SAMPLE IQR clipping of O/H/L/C uses future quantiles (look-ahead), clips each
        # column independently (can invert high<low) and distorts true tail returns. Disabled by default.
        self.clip_ohlc = clip_ohlc
        self.metrics_dir = metrics_dir
        
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def validate_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Main entry point. Runs all validation and cleaning steps.
        Returns cleaned DataFrame and a detailed report dict."""
        
        if df.empty:
            logger.warning("Empty dataframe provided to DataQualityEngine.")
            return df, {}
            
        required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(f"Missing required columns: {missing}")
        
        df = df.copy()
        
        # Sort values just in case
        if 'date' in df.columns and 'ticker' in df.columns:
            df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
            
        report = {
            'initial_rows': len(df),
            'initial_tickers': df['ticker'].nunique() if 'ticker' in df.columns else 0,
        }
        
        # 1. Stale Data Removal
        df, stale_report = self._remove_stale_data(df)
        report['stale_tickers'] = stale_report
        
        # 2. Corporate Actions
        if not df.empty:
            df, ca_report = self._detect_corporate_actions(df)
            report['corporate_actions_detected'] = ca_report
            
        # 3. Outliers
        if not df.empty:
            df, out_report = self._winsorize_outliers(df)
            report['outliers_clipped'] = out_report
            
        # 4. Cross Sectional Consistency
        if not df.empty:
            xs_report = self._check_cross_sectional_consistency(df)
            report['cross_sectional_warnings'] = xs_report
            
        # 5. Feature Stability
        if not df.empty:
            fs_report = self._compute_feature_stability(df)
            report['feature_stability'] = fs_report
            
        report['final_rows'] = len(df)
        report['final_tickers'] = df['ticker'].nunique() if 'ticker' in df.columns else 0
        report['rows_cleaned'] = report['initial_rows'] - report['final_rows']
        
        logger.info(f"Data quality check complete. Cleaned {report['rows_cleaned']} rows. "
                    f"Remaining tickers: {report['final_tickers']}/{report['initial_tickers']}")
                    
        return df, report

    def _detect_corporate_actions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Detect splits/bonuses (>20% gap in close), suspensions (volume drop >95% for >5 days).
        Flag but don't remove — let downstream decide."""
        ca_report = defaultdict(list)
        df['is_split_bonus'] = False
        df['is_suspended'] = False
        
        for ticker, group in df.groupby('ticker'):
            # Splits / Bonuses
            returns = group['close'].pct_change().abs()
            split_idx = returns > (self.gap_threshold_pct / 100.0)
            if split_idx.any():
                dates = group.loc[split_idx, 'date'].astype(str).tolist()
                ca_report['splits_bonuses'].append({'ticker': ticker, 'dates': dates})
                df.loc[split_idx[split_idx].index, 'is_split_bonus'] = True   # FIX: was split_idx.index => flagged EVERY row of the ticker (incl. pre-event dates = look-ahead)
                
            # Suspensions
            if 'volume' in group.columns:
                # rolling median volume over previous 20 days
                roll_med_vol = group['volume'].rolling(20, min_periods=1).median().shift(1)
                vol_drop = group['volume'] < (self.volume_drop_threshold * roll_med_vol)
                # Find consecutive drops
                is_susp = vol_drop.rolling(self.volume_drop_days).sum() >= self.volume_drop_days
                if is_susp.any():
                    dates = group.loc[is_susp, 'date'].astype(str).tolist()
                    ca_report['suspensions'].append({'ticker': ticker, 'dates': dates})
                    df.loc[is_susp[is_susp].index, 'is_suspended'] = True
                
        return df, dict(ca_report)

    def _winsorize_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Per-ticker IQR-based winsorization for OHLCV and returns.
        - Returns: clip at ±max_daily_return (no stock moves 25% without halt)
        - Volume: log-transform then clip at max_volume_zscore σ
        - OHLC: clip at Q1 - iqr_mult*IQR, Q3 + iqr_mult*IQR per ticker"""
        out_report = defaultdict(int)
        
        def winsorize_group(g):
            nonlocal out_report
            # OHLC
            for col in (['open', 'high', 'low', 'close'] if self.clip_ohlc else []):
                if col in g.columns:
                    q1 = g[col].quantile(0.25)
                    q3 = g[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - self.iqr_multiplier * iqr
                    upper = q3 + self.iqr_multiplier * iqr
                    
                    clip_mask = (g[col] < lower) | (g[col] > upper)
                    out_report[f'{col}_clipped'] += clip_mask.sum()
                    g[col] = g[col].clip(lower=lower, upper=upper)
                    
            # Volume: CAUSAL rolling statistics (strictly <= t, no full-history leak)
            if 'volume' in g.columns:
                log_v = np.log1p(g['volume'].clip(lower=0))
                roll_med = log_v.rolling(60, min_periods=10).median()
                roll_mad = (log_v - roll_med).abs().rolling(60, min_periods=10).median()
                roll_scale = 1.4826 * roll_mad.replace(0, np.nan).fillna(1.0)
                upper_log = roll_med + self.max_volume_zscore * roll_scale
                upper_vol = np.expm1(upper_log.bfill().fillna(log_v)) if False else np.expm1(upper_log.fillna(log_v))
                clip_mask = (g['volume'] > upper_vol) & upper_vol.notna()
                out_report['volume_clipped'] += clip_mask.sum()
                g['volume'] = np.where(clip_mask, upper_vol, g['volume'])
            return g
            
        df = df.groupby('ticker', group_keys=False).apply(winsorize_group)
        
        # Calculate returns if needed or just clip them if they exist
        if 'return' in df.columns:
            clip_mask = (df['return'] > self.max_daily_return) | (df['return'] < -self.max_daily_return)
            out_report['return_clipped'] += clip_mask.sum()
            df['return'] = df['return'].clip(lower=-self.max_daily_return, upper=self.max_daily_return)

        if 'close' in df.columns:
            df['_daily_return'] = df.groupby('ticker')['close'].pct_change()
            clip_mask = (df['_daily_return'] > self.max_daily_return) | (df['_daily_return'] < -self.max_daily_return)
            out_report['daily_return_clipped'] += clip_mask.sum()
            df = df.drop(columns=['_daily_return'])
            
        return df, dict(out_report)

    def _remove_stale_data(self, df: pd.DataFrame, point_in_time_mode: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Causally filter stale or non-tradable rows without full-sample survivorship bias.
        
        In point_in_time_mode (default):
        - Masks per-row non-tradable bars (e.g. price < min_price at date t, zero volume at date t).
        - Preserves historical bars of delisted or suspended companies up to their last trading date.
        
        If point_in_time_mode=False:
        - Legacy ticker-level exclusion mode (for training dataset pre-filtering).
        """
        if point_in_time_mode:
            removal_reasons = defaultdict(int)
            # Minimum history filter
            short_tickers = set()
            for ticker, grp in df.groupby('ticker'):
                if len(grp) < self.min_trading_days:
                    short_tickers.add(ticker)
            if short_tickers:
                removal_reasons['insufficient_history_tickers'] = len(short_tickers)
                df = df[~df['ticker'].isin(short_tickers)].copy()

            # 1. Causal price filter (penny stock at date t)
            penny_mask = (df['close'] < self.min_price) if 'close' in df.columns else pd.Series(False, index=df.index)
            removal_reasons['penny_stock_rows'] = int(penny_mask.sum())
            
            # 2. Causal zero volume on active bar
            zero_vol_mask = (df['volume'] <= 0) if 'volume' in df.columns else pd.Series(False, index=df.index)
            removal_reasons['zero_volume_rows'] = int(zero_vol_mask.sum())
            
            # Filter rows without dropping entire ticker history
            clean_df = df[~(penny_mask | zero_vol_mask)].copy()
            return clean_df, dict(removal_reasons)

        tickers_to_remove = set()
        removal_reasons = defaultdict(list)
        
        for ticker, group in df.groupby('ticker'):
            if len(group) < self.min_trading_days:
                tickers_to_remove.add(ticker)
                removal_reasons[ticker].append('min_trading_days')
                continue
                
            if group['close'].median() < self.min_price:
                tickers_to_remove.add(ticker)
                removal_reasons[ticker].append('min_price')
                continue
                
            vol_zero_pct = (group['volume'] <= 0).mean() * 100
            if vol_zero_pct > self.max_missing_volume_pct:
                tickers_to_remove.add(ticker)
                removal_reasons[ticker].append('missing_volume')
                continue
                
            is_same = group['close'].diff() == 0
            consecutive = is_same.groupby((~is_same).cumsum()).sum()
            if consecutive.max() > self.max_consecutive_identical:
                tickers_to_remove.add(ticker)
                removal_reasons[ticker].append('consecutive_identical')
                
        clean_df = df[~df['ticker'].isin(tickers_to_remove)].copy()
        return clean_df, dict(removal_reasons)

    def _check_cross_sectional_consistency(self, df: pd.DataFrame) -> Dict:
        """Verify date alignment, flag outlier return distributions (KS test vs universe).
        Returns warnings dict — does NOT modify data."""
        warnings = defaultdict(list)
        
        # Check date alignment
        date_counts = df['date'].value_counts()
        if date_counts.std() > 0.1 * date_counts.mean():
            warnings['date_alignment'].append("High variance in number of tickers per date. Possible missing cross-sections.")
            
        # Outlier return distributions
        if 'close' in df.columns:
            # calculate a simplified return for consistency check
            returns = df.groupby('ticker')['close'].pct_change().dropna()
            if not returns.empty:
                universe_returns = returns.values
                
                for ticker, group in df.groupby('ticker'):
                    tick_returns = group['close'].pct_change().dropna().values
                    if len(tick_returns) > 30 and len(universe_returns) > 30:
                        stat, p_val = stats.ks_2samp(tick_returns, universe_returns)
                        if p_val < 0.01:
                            warnings['outlier_distributions'].append(ticker)
                            
        return dict(warnings)

    def _compute_feature_stability(self, df: pd.DataFrame) -> Dict:
        """Compute PSI between first-half and second-half of training data for numeric features.
        Features with PSI > threshold are flagged as unstable."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ignore_cols = ['date', 'is_split_bonus', 'is_suspended']
        numeric_cols = [c for c in numeric_cols if c not in ignore_cols]
        
        if len(df) < 100:
            return {}
            
        # Split data chronologically
        dates = sorted(df['date'].unique())
        mid_idx = len(dates) // 2
        mid_date = dates[mid_idx]
        
        first_half = df[df['date'] < mid_date]
        second_half = df[df['date'] >= mid_date]
        
        stability_report = {}
        
        for col in numeric_cols:
            base = first_half[col].dropna().values
            current = second_half[col].dropna().values
            
            if len(base) > 0 and len(current) > 0:
                psi_val = self._psi(base, current)
                status = "unstable" if psi_val > self.psi_warning_threshold else "stable"
                stability_report[col] = {
                    'psi': float(psi_val),
                    'status': status
                }
                
        # Save to metrics dir
        try:
            out_path = os.path.join(self.metrics_dir, 'feature_stability.json')
            with open(out_path, 'w') as f:
                json.dump(stability_report, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save feature stability report: {e}")
            
        return stability_report

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        """Population Stability Index between two distributions."""
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
            
        # Create bins based on expected
        bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        # ensure unique bins
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
            
        # add tiny amount to last bin to include max value
        bins[-1] += 1e-8
        bins[0] -= 1e-8
        
        expected_pct = np.histogram(expected, bins)[0] / len(expected)
        actual_pct = np.histogram(actual, bins)[0] / len(actual)
        
        # Replace 0 with small value to avoid division by zero
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return psi_value