"""
labeling.py — Multi-Horizon Cross-Sectional Targets, Vectorized Triple-Barrier Labeling, and Uniqueness Weights.

TIMING CONVENTION:
Features use data <= close of day t -> entry at OPEN t+1 -> exit at CLOSE t+k or earlier on barrier touch.
Barriers use high/low of t+1…t+k; same-day double touch -> stop first; open through a barrier -> fill at open.
Adjusted prices; point-in-time universe incl. delisted; reindex to master trading calendar (k = market days).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from numpy.lib.stride_tricks import sliding_window_view
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class TripleBarrierLabeler:
    """
    Triple Barrier Event-Driven Labeler:
    Implements vectorized and brute-force event-driven labeling across horizons k in {3, 5, 7, 15, 21}.

    Rules & Edge-Case Handling:
      1. Entry: at Open of day t+1 (price O_{t+1}).
      2. Dynamic Barrier Width: W_{t, k} = m * sigma_{t} * sqrt(k), where sigma_t is known at close of day t.
      3. Profit Target (Upper Barrier): U_{t, k} = O_{t+1} * exp(pt * W_{t, k}).
      4. Stop Loss (Lower Barrier): L_{t, k} = O_{t+1} * exp(-sl * W_{t, k}).
      5. Monitoring Span: Daily bars tau in {t+1, ..., t+k}.
      6. Open Through Barrier: If O_{tau} >= U, fill at O_{tau}; if O_{tau} <= L, fill at O_{tau}.
      7. Same-Day Double Touch: If both H_{tau} >= U and L_{tau} <= L occur on the same day, stop-loss hits first.
      8. Vertical Barrier: If neither barrier touched by close of day t+k, exit at Close C_{t+k} (label 0).
    """

    def __init__(
        self,
        k_horizons: List[int] = [3, 5, 7, 15, 21],
        pt: float = 1.0,
        sl: float = 1.0,
        m: float = 1.0
    ):
        self.k_horizons = k_horizons
        self.pt = pt
        self.sl = sl
        self.m = m

    def get_events(self, df: pd.DataFrame, k: int, method: str = 'vectorized') -> pd.DataFrame:
        """
        Computes barrier events for horizon k.
        df must contain: 'open', 'high', 'low', 'close', and 'volatility' (e.g. 20d realized or Parkinson).
        method: 'vectorized' (default) or 'brute_force'.
        """
        if method == 'brute_force':
            return self.get_events_brute_force(df, k)

        n = len(df)
        labels = np.full(n, 0, dtype=int)
        returns = np.full(n, np.nan, dtype=float)
        fill_prices = np.full(n, np.nan, dtype=float)
        touch_types = np.array(['none'] * n, dtype=object)

        t0_dates = pd.Series(pd.NaT, index=df.index)
        t1_dates = pd.Series(pd.NaT, index=df.index)

        if n <= k + 1:
            return pd.DataFrame({
                f'label_{k}': labels,
                f'ret_{k}': returns,
                f'fill_{k}': fill_prices,
                f't0_{k}': t0_dates,
                f't1_{k}': t1_dates,
                f'touch_type_{k}': touch_types
            }, index=df.index)

        valid_n = n - k

        opens = df['open'].values.astype(float)
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        closes = df['close'].values.astype(float)
        vols = df['volatility'].values.astype(float)
        dt_index = df.index.values

        # Entry is Open at t+1
        entry_p = opens[1:valid_n + 1]
        sigma = vols[:valid_n]

        # Valid mask for prediction point t
        is_valid = (
            np.isfinite(entry_p) & (entry_p > 0) &
            np.isfinite(sigma) & (sigma > 0)
        )

        width = self.m * sigma * np.sqrt(k)
        ub = np.full((valid_n, 1), np.nan)
        lb = np.full((valid_n, 1), np.nan)
        ub[is_valid, 0] = entry_p[is_valid] * np.exp(self.pt * width[is_valid])
        lb[is_valid, 0] = entry_p[is_valid] * np.exp(-self.sl * width[is_valid])

        # Sliding window views of shape (valid_n, k)
        # For prediction t, window covers days t+1 to t+k
        o_win = sliding_window_view(opens[1:], k)[:valid_n]
        h_win = sliding_window_view(highs[1:], k)[:valid_n]
        l_win = sliding_window_view(lows[1:], k)[:valid_n]
        c_win = sliding_window_view(closes[1:], k)[:valid_n]

        # Touch masks
        o_gap_up = o_win >= ub
        o_gap_down = o_win <= lb
        intra_up = h_win >= ub
        intra_down = l_win <= lb

        has_down_day = o_gap_down | intra_down
        has_up_day = o_gap_up | intra_up

        down_touched = has_down_day.any(axis=1) & is_valid
        up_touched = has_up_day.any(axis=1) & is_valid

        # First touch index (offset 0..k-1 within the forward window)
        first_down = np.where(down_touched, has_down_day.argmax(axis=1), k + 1)
        first_up = np.where(up_touched, has_up_day.argmax(axis=1), k + 1)

        # Disambiguation:
        # Same-day double touch or down earlier -> Stop Loss (down_first)
        down_first = is_valid & (
            (first_down < first_up) |
            ((first_down == first_up) & (first_down < k))
        )
        up_first = is_valid & (first_up < first_down)
        vertical_exit = is_valid & (~down_first) & (~up_first)

        # 1. Down first fills (stop loss)
        down_idx = np.where(down_first)[0]
        if len(down_idx) > 0:
            d_days = first_down[down_idx]
            labels[down_idx] = -1
            opened_down = o_gap_down[down_idx, d_days]
            is_double = intra_up[down_idx, d_days] & intra_down[down_idx, d_days]

            fill_down = np.where(opened_down, o_win[down_idx, d_days], lb[down_idx, 0])
            fill_prices[down_idx] = fill_down
            returns[down_idx] = np.log(fill_down / entry_p[down_idx])

            # Classify touch type
            for r_i, d_day, op_dn, dbl in zip(down_idx, d_days, opened_down, is_double):
                if op_dn:
                    touch_types[r_i] = 'gap_down'
                elif dbl:
                    touch_types[r_i] = 'double_touch'
                else:
                    touch_types[r_i] = 'lower_touch'

            t1_dates.iloc[down_idx] = dt_index[down_idx + 1 + d_days]

        # 2. Up first fills (profit target)
        up_idx = np.where(up_first)[0]
        if len(up_idx) > 0:
            u_days = first_up[up_idx]
            labels[up_idx] = 1
            opened_up = o_gap_up[up_idx, u_days]

            fill_up = np.where(opened_up, o_win[up_idx, u_days], ub[up_idx, 0])
            fill_prices[up_idx] = fill_up
            returns[up_idx] = np.log(fill_up / entry_p[up_idx])

            for r_i, op_up in zip(up_idx, opened_up):
                touch_types[r_i] = 'gap_up' if op_up else 'upper_touch'

            t1_dates.iloc[up_idx] = dt_index[up_idx + 1 + u_days]

        # 3. Vertical barrier fills (time-out at day t+k close)
        vert_idx = np.where(vertical_exit)[0]
        if len(vert_idx) > 0:
            labels[vert_idx] = 0
            fill_vert = closes[vert_idx + k]
            fill_prices[vert_idx] = fill_vert
            returns[vert_idx] = np.log(fill_vert / entry_p[vert_idx])
            touch_types[vert_idx] = 'vertical'
            t1_dates.iloc[vert_idx] = dt_index[vert_idx + k]

        # Record t0 (entry date is day t+1)
        valid_indices = np.where(is_valid)[0]
        t0_dates.iloc[valid_indices] = dt_index[valid_indices + 1]

        res = pd.DataFrame({
            f'label_{k}': labels,
            f'ret_{k}': returns,
            f'fill_{k}': fill_prices,
            f't0_{k}': t0_dates,
            f't1_{k}': t1_dates,
            f'touch_type_{k}': touch_types
        }, index=df.index)

        return res

    def get_events_brute_force(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Brute-force loop reference implementation for equivalence verification.
        Iterates day-by-day and bar-by-bar.
        """
        n = len(df)
        labels = np.full(n, 0, dtype=int)
        returns = np.full(n, np.nan, dtype=float)
        fill_prices = np.full(n, np.nan, dtype=float)
        touch_types = np.array(['none'] * n, dtype=object)

        t0_dates = pd.Series(pd.NaT, index=df.index)
        t1_dates = pd.Series(pd.NaT, index=df.index)

        opens = df['open'].values.astype(float)
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        closes = df['close'].values.astype(float)
        vols = df['volatility'].values.astype(float)
        dt_index = df.index.values

        for t in range(n - k):
            entry_p = opens[t + 1]
            sigma = vols[t]

            if not (np.isfinite(entry_p) and entry_p > 0 and np.isfinite(sigma) and sigma > 0):
                continue

            t0_dates.iloc[t] = dt_index[t + 1]
            width = self.m * sigma * np.sqrt(k)
            ub = entry_p * np.exp(self.pt * width)
            lb = entry_p * np.exp(-self.sl * width)

            touched = False
            for j in range(k):
                day_idx = t + 1 + j
                day_o = opens[day_idx]
                day_h = highs[day_idx]
                day_l = lows[day_idx]

                o_gap_up = day_o >= ub
                o_gap_down = day_o <= lb
                intra_up = day_h >= ub
                intra_down = day_l <= lb

                if o_gap_down:
                    labels[t] = -1
                    fill_prices[t] = day_o
                    returns[t] = np.log(day_o / entry_p)
                    touch_types[t] = 'gap_down'
                    t1_dates.iloc[t] = dt_index[day_idx]
                    touched = True
                    break
                elif o_gap_up:
                    labels[t] = 1
                    fill_prices[t] = day_o
                    returns[t] = np.log(day_o / entry_p)
                    touch_types[t] = 'gap_up'
                    t1_dates.iloc[t] = dt_index[day_idx]
                    touched = True
                    break
                elif intra_up and intra_down:
                    # Same-day double touch: stop first
                    labels[t] = -1
                    fill_prices[t] = lb
                    returns[t] = np.log(lb / entry_p)
                    touch_types[t] = 'double_touch'
                    t1_dates.iloc[t] = dt_index[day_idx]
                    touched = True
                    break
                elif intra_down:
                    labels[t] = -1
                    fill_prices[t] = lb
                    returns[t] = np.log(lb / entry_p)
                    touch_types[t] = 'lower_touch'
                    t1_dates.iloc[t] = dt_index[day_idx]
                    touched = True
                    break
                elif intra_up:
                    labels[t] = 1
                    fill_prices[t] = ub
                    returns[t] = np.log(ub / entry_p)
                    touch_types[t] = 'upper_touch'
                    t1_dates.iloc[t] = dt_index[day_idx]
                    touched = True
                    break

            if not touched:
                # Vertical barrier reached
                day_idx = t + k
                labels[t] = 0
                fill_prices[t] = closes[day_idx]
                returns[t] = np.log(closes[day_idx] / entry_p)
                touch_types[t] = 'vertical'
                t1_dates.iloc[t] = dt_index[day_idx]

        return pd.DataFrame({
            f'label_{k}': labels,
            f'ret_{k}': returns,
            f'fill_{k}': fill_prices,
            f't0_{k}': t0_dates,
            f't1_{k}': t1_dates,
            f'touch_type_{k}': touch_types
        }, index=df.index)

    def compute_meta_labels(
        self,
        events_df: pd.DataFrame,
        primary_sides: pd.Series,
        cost_pct: float = 0.0030,
        k: int = 5
    ) -> pd.Series:
        """
        Meta-Labeling Formulation:
        Given primary side signal S in {-1, +1} (or +1 for cash long-only):
        y_meta = 1 if S * (RealizedReturn - RoundTripCost) > 0 else 0.
        """
        ret_col = f'ret_{k}'
        if ret_col not in events_df.columns:
            raise ValueError(f"Missing {ret_col} in events_df")

        rets = events_df[ret_col]
        sides = primary_sides.reindex(events_df.index).fillna(0.0)

        net_returns = rets - cost_pct
        meta_label = (sides * net_returns > 0).astype(int)
        # Mask rows where realized return is NaN
        meta_label = meta_label.where(rets.notna(), np.nan)
        return meta_label

    def compute_sample_weights(
        self,
        t0_series: pd.Series,
        t1_series: pd.Series,
        calendar: Optional[Union[pd.DatetimeIndex, List[pd.Timestamp]]] = None
    ) -> pd.Series:
        """
        Average-Uniqueness Weights via Prefix Sums:
        c_tau: number of concurrent active positions on day tau.
        Calculated in O(N + T) time using difference arrays and cumulative prefix sums.
        Uses master trading calendar (or continuous business days) between min(t0) and max(t1).
        """
        valid_mask = t0_series.notna() & t1_series.notna()
        t0_clean = t0_series.loc[valid_mask]
        t1_clean = t1_series.loc[valid_mask]

        weights = pd.Series(np.nan, index=t0_series.index)
        if len(t0_clean) == 0:
            return weights

        # Establish continuous calendar
        min_date = t0_clean.min()
        max_date = t1_clean.max()

        if calendar is not None:
            cal_idx = pd.DatetimeIndex(calendar)
            all_dates = cal_idx[(cal_idx >= min_date) & (cal_idx <= max_date)].values
        else:
            try:
                min_dt = pd.to_datetime(min_date)
                max_dt = pd.to_datetime(max_date)
                all_dates = pd.bdate_range(min_dt, max_dt).values
            except Exception:
                all_dates = np.unique(np.concatenate([t0_clean.values, t1_clean.values]))
                all_dates.sort()

        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        T = len(all_dates)

        # Difference array for concurrency count
        delta = np.zeros(T + 1, dtype=int)
        start_indices = [date_to_idx.get(d, 0) for d in t0_clean.values]
        end_indices = [date_to_idx.get(d, T - 1) for d in t1_clean.values]

        for s_idx, e_idx in zip(start_indices, end_indices):
            delta[s_idx] += 1
            delta[e_idx + 1] -= 1

        concurrency = np.cumsum(delta)[:T]
        inv_concurrency = 1.0 / np.maximum(concurrency, 1)

        # Cumulative prefix sums of inverse concurrency
        I = np.concatenate([[0.0], np.cumsum(inv_concurrency)])

        # Compute average uniqueness per sample in O(1)
        uniqueness = np.zeros(len(t0_clean))
        for i, (s_idx, e_idx) in enumerate(zip(start_indices, end_indices)):
            span = e_idx - s_idx + 1
            uniqueness[i] = (I[e_idx + 1] - I[s_idx]) / span

        # Normalize weights so that mean weight is 1.0
        norm_factor = len(uniqueness) / (np.sum(uniqueness) + 1e-12)
        weights.loc[valid_mask] = uniqueness * norm_factor

        return weights


class CrossSectionalTargetBuilder:
    """
    Multi-Horizon Cross-Sectional Target Builder:
    Constructs causal continuous and discrete cross-sectional targets for k in {3, 5, 7, 15, 21}:
      1. Beta-Lagged Excess Return: y_excess = r_{i, t->t+k} - beta_{i, t} * r_{m, t->t+k}
      2. Rank-Gauss Target: Phi^{-1}((Rank_t - 0.5) / N_t)
      3. Volatility-Scaled Target: y_excess / (sigma_{i, t} * sqrt(k))
      4. Quantile Bins: floor(u_t * Q) in {0, ..., Q-1}
    """

    def __init__(self):
        pass

    def compute_excess_returns(
        self,
        df: pd.DataFrame,
        k: int,
        market_series: Optional[pd.Series] = None,
        beta_col: str = 'beta',
        date_col: str = 'date'
    ) -> pd.Series:
        """
        Computes beta-lagged excess return for horizon k:
        Entry at O_{t+1}, Exit at C_{t+k}.
        df must be sorted ascending by date.
        """
        open_t1 = df['open'].shift(-1)
        close_tk = df['close'].shift(-k)
        stock_ret = np.log(close_tk / open_t1)

        if market_series is not None:
            if date_col in df.columns:
                m_vals = df[date_col].map(market_series).values
            else:
                m_vals = pd.Series(df.index).map(market_series).values

            m_series_aligned = pd.Series(m_vals, index=df.index, dtype=float)
            m_t1 = m_series_aligned.shift(-1)
            m_tk = m_series_aligned.shift(-k)
            market_ret = np.log(m_tk / m_t1)
        else:
            market_ret = pd.Series(0.0, index=df.index)

        if beta_col in df.columns:
            beta = df[beta_col]
        else:
            beta = 1.0

        excess_ret = stock_ret - beta * market_ret
        return excess_ret

    def compute_rank_gauss_target(
        self,
        panel_df: pd.DataFrame,
        excess_col: str,
        date_col: str = 'date'
    ) -> pd.Series:
        """
        Performs date-wise cross-sectional Rank-Gauss transform on target excess_col.
        """
        def _rank_gauss_fn(s: pd.Series) -> pd.Series:
            valid = s.dropna()
            N = len(valid)
            if N == 0:
                return pd.Series(np.nan, index=s.index)
            u = (s.rank(method='average') - 0.5) / N
            u_clipped = np.clip(u, 1e-4, 1.0 - 1e-4)
            return pd.Series(norm.ppf(u_clipped), index=s.index)

        if date_col in panel_df.columns:
            return panel_df.groupby(date_col)[excess_col].transform(_rank_gauss_fn)
        elif panel_df.index.name == date_col or isinstance(panel_df.index, pd.DatetimeIndex):
            return panel_df.groupby(level=0)[excess_col].transform(_rank_gauss_fn)
        else:
            return _rank_gauss_fn(panel_df[excess_col])

    def compute_vol_scaled_target(
        self,
        excess_series: pd.Series,
        vol_series: pd.Series,
        k: int
    ) -> pd.Series:
        """
        Excess return scaled by horizon volatility: y_excess / (sigma_t * sqrt(k)).
        """
        denom = vol_series * np.sqrt(k)
        return excess_series / (denom + 1e-6)

    def compute_quantile_bins(
        self,
        panel_df: pd.DataFrame,
        target_col: str,
        date_col: str = 'date',
        num_bins: int = 10
    ) -> pd.Series:
        """
        Per-date discrete quantile bins {0, ..., num_bins-1}.
        """
        def _bins_fn(s: pd.Series) -> pd.Series:
            valid = s.dropna()
            N = len(valid)
            if N == 0:
                return pd.Series(np.nan, index=s.index)
            u = (s.rank(method='average') - 0.5) / N
            bins = np.floor(u * num_bins).astype(float).clip(0, num_bins - 1)
            # Retain NaNs from original series
            bins = bins.where(s.notna(), np.nan)
            return pd.Series(bins, index=s.index)

        if date_col in panel_df.columns:
            return panel_df.groupby(date_col)[target_col].transform(_bins_fn)
        elif panel_df.index.name == date_col or isinstance(panel_df.index, pd.DatetimeIndex):
            return panel_df.groupby(level=0)[target_col].transform(_bins_fn)
        else:
            return _bins_fn(panel_df[target_col])

    def build_all_targets(
        self,
        panel_df: pd.DataFrame,
        horizons: List[int] = [3, 5, 7, 15, 21],
        market_series: Optional[pd.Series] = None,
        beta_col: str = 'beta',
        vol_col: str = 'volatility',
        num_quantiles: int = 10,
        ticker_col: str = 'ticker',
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Builds full target matrix across all horizons k in {3, 5, 7, 15, 21}.
        """
        df_out = panel_df.copy()

        for k in horizons:
            excess_list = []
            for ticker, grp in df_out.groupby(ticker_col):
                grp_sorted = grp.sort_values(date_col)
                exc = self.compute_excess_returns(
                    grp_sorted, k=k, market_series=market_series, beta_col=beta_col, date_col=date_col
                )
                excess_list.append(exc)

            excess_series = pd.concat(excess_list).reindex(df_out.index)
            df_out[f'target_excess_{k}d'] = excess_series

            # Vol-scaled
            if vol_col in df_out.columns:
                df_out[f'target_vol_scaled_{k}d'] = self.compute_vol_scaled_target(excess_series, df_out[vol_col], k=k)

            # Rank-Gauss
            df_out[f'target_rank_gauss_{k}d'] = self.compute_rank_gauss_target(df_out, f'target_excess_{k}d', date_col=date_col)

            # Quantile bins
            df_out[f'target_bin_{k}d'] = self.compute_quantile_bins(df_out, f'target_excess_{k}d', date_col=date_col, num_bins=num_quantiles)

        return df_out
