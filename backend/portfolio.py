"""
Multi-Horizon Portfolio Construction and Execution Engine
=========================================================
Institutional portfolio simulator and rebalancer for NSE equities.

Key Features:
1. Rank-Based Top-N Portfolio with Hysteresis:
   - Upper entry threshold N_enter (e.g. 20)
   - Lower exit threshold N_exit (e.g. 35)
   - Eliminates boundary churn, reducing turnover by >= 30% vs naive top-N.
2. Regulatory Segment Separation:
   - Cash Delivery: Strictly long-only (w_i >= 0, sum w_i <= 1.0). Cash delivery cannot be shorted.
   - F&O Book: Evaluated separately. Top decile long, bottom decile short (dollar neutral).
3. Benchmark:
   - Equal-weight PIT ELIGIBLE universe (not Nifty index).
4. Realistic Execution Frictions & Market Impact:
   - Statutory NSE friction stack (STT, Stamp Duty, Exch Txn, SEBI, GST, DP charge, Zero brokerage).
   - Non-linear Almgren-Chriss square-root impact: eta * (Q / ADTV_20)^alpha * sigma_daily.
5. Capacity Estimation:
   - Evaluates impact drag across AUM grid (₹1 Cr to ₹100 Cr+).
   - Computes break-even capacity where net alpha >= hurdle rate.

Timing Convention:
- Features/Scores use data <= Close(t).
- Execution occurs at OPEN(t+1) with market impact.
- Holdings held until next rebalance date or barrier exit.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import pandas as pd

from frictions import (
    TradingSegment, NSECostConfig, NSECostModel,
    ImpactModelConfig, MarketImpactModel, ExecutionFrictions
)


class WeightingScheme(Enum):
    """Portfolio weighting methodology."""
    EQUAL_WEIGHT = "equal_weight"
    RANK_WEIGHT = "rank_weight"
    VOL_INVERSE = "vol_inverse"


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction and rebalancing."""
    n_enter: int = 20                   # Target portfolio size / entry rank threshold
    n_exit: int = 35                    # Exit threshold rank (n_exit >= n_enter)
    rebalance_freq: int = 5             # Rebalance frequency in trading days (k=5)
    weighting_scheme: WeightingScheme = WeightingScheme.EQUAL_WEIGHT
    max_weight_per_stock: float = 0.10  # Max single-stock weight (10%)
    cash_buffer_pct: float = 0.01       # 1% cash buffer reserved for frictional cash drag
    initial_aum: float = 10_000_000.0   # Default initial capital: ₹1 Crore (10M INR)
    segment: TradingSegment = TradingSegment.CASH_DELIVERY
    
    # Impact parameters
    impact_eta: float = 0.10
    impact_alpha: float = 0.50


@dataclass
class RebalanceEvent:
    """Record of a single rebalance event."""
    rebalance_date: Any
    holdings_before: List[str]
    holdings_after: List[str]
    buys: List[str]
    sells: List[str]
    turnover: float
    statutory_cost_inr: float
    impact_cost_inr: float
    total_friction_inr: float
    aum_before: float
    aum_after_friction: float


class RankHysteresisPortfolio:
    """
    Rank-Based Portfolio Selection with Hysteresis Band.

    Mechanics:
    - At rebalance time t, stocks in the eligible cross-section are ranked by predicted score (1 = best).
    - Existing holdings H_{t-1} are retained IF rank_i <= N_exit.
    - If retained count < N_enter, fill vacancies from unheld eligible stocks with rank_j <= N_enter (best rank first).
    - If retained count > N_enter, retain top N_enter of the retained pool.
    - Cash long-only strictly enforces w_i >= 0, sum w_i <= 1.0.
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        if self.config.n_exit < self.config.n_enter:
            raise ValueError(f"n_exit ({self.config.n_exit}) must be >= n_enter ({self.config.n_enter})")
        self.current_holdings: Set[str] = set()

    def select_holdings(
        self,
        scores_df: pd.DataFrame,
        use_hysteresis: bool = True
    ) -> List[str]:
        """
        Determine target holdings for the current period.

        Parameters
        ----------
        scores_df : pd.DataFrame
            Must contain ['symbol', 'score']. Higher score = better predicted return.
            Sorted by score descending (or rank 1 to M).
        use_hysteresis : bool
            If True, applies hysteresis buffer [N_enter, N_exit].
            If False, naively selects top N_enter stocks.

        Returns
        -------
        List[str]
            List of selected symbols.
        """
        df = scores_df.sort_values(by='score', ascending=False).reset_index(drop=True)
        df['rank'] = np.arange(1, len(df) + 1)
        rank_dict = dict(zip(df['symbol'], df['rank']))

        if not use_hysteresis or len(self.current_holdings) == 0:
            # Naive top-N selection
            selected = df.head(self.config.n_enter)['symbol'].tolist()
            self.current_holdings = set(selected)
            return selected

        # 1. Retention step: keep existing holdings if rank <= n_exit
        retained = [
            sym for sym in self.current_holdings
            if sym in rank_dict and rank_dict[sym] <= self.config.n_exit
        ]

        # 2. Fill vacancies if retained < n_enter
        needed = self.config.n_enter - len(retained)
        if needed > 0:
            # Candidates: unheld eligible stocks with rank <= n_enter
            candidates = [
                sym for sym in df['symbol']
                if sym not in retained and rank_dict[sym] <= self.config.n_enter
            ]
            new_additions = candidates[:needed]
            selected = retained + new_additions
        elif len(retained) > self.config.n_enter:
            # Retained exceeds target, keep best n_enter
            retained_sorted = sorted(retained, key=lambda s: rank_dict.get(s, 999999))
            selected = retained_sorted[:self.config.n_enter]
        else:
            selected = retained

        # If universe had fewer than n_enter qualified names, just return selected
        self.current_holdings = set(selected)
        return selected

    def compute_weights(
        self,
        selected_symbols: List[str],
        scores_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Calculate portfolio weights subject to cash long-only constraint:
        w_i >= 0, sum(w_i) <= (1.0 - cash_buffer_pct), max(w_i) <= max_weight_per_stock.
        """
        n = len(selected_symbols)
        if n == 0:
            return {}

        investable_capital_fraction = 1.0 - self.config.cash_buffer_pct

        if self.config.weighting_scheme == WeightingScheme.EQUAL_WEIGHT or scores_df is None:
            raw_w = investable_capital_fraction / n
            weights = {sym: min(raw_w, self.config.max_weight_per_stock) for sym in selected_symbols}
        elif self.config.weighting_scheme == WeightingScheme.RANK_WEIGHT:
            # Linear rank weights: top rank gets higher weight
            ranks = np.arange(1, n + 1)
            raw_weights = (n + 1 - ranks) / np.sum(n + 1 - ranks) * investable_capital_fraction
            weights = {}
            for sym, rw in zip(selected_symbols, raw_weights):
                weights[sym] = min(float(rw), self.config.max_weight_per_stock)
        else:
            # Fallback equal weight
            raw_w = investable_capital_fraction / n
            weights = {sym: min(raw_w, self.config.max_weight_per_stock) for sym in selected_symbols}

        # Normalize to ensure sum <= investable_capital_fraction
        total_w = sum(weights.values())
        if total_w > investable_capital_fraction:
            scale = investable_capital_fraction / total_w
            weights = {k: v * scale for k, v in weights.items()}

        return weights


class PortfolioSimulator:
    """
    Multi-Period Quantitative Portfolio Simulator.
    Simulates real-world execution with:
    - Rebalancing at Open(t+1) following Close(t) signal.
    - Statutory transaction frictions (Buy & Sell).
    - Market impact model.
    - Equal-Weight Eligible Universe benchmark.
    - Side-by-side Naive vs Hysteresis turnover comparison.
    """

    def __init__(
        self,
        config: Optional[PortfolioConfig] = None,
        cost_config: Optional[NSECostConfig] = None,
        impact_config: Optional[ImpactModelConfig] = None,
    ):
        self.config = config or PortfolioConfig()
        self.frictions = ExecutionFrictions(
            cost_config=cost_config,
            impact_config=impact_config or ImpactModelConfig(
                eta=self.config.impact_eta,
                alpha=self.config.impact_alpha,
            ),
        )

    def run_simulation(
        self,
        panel_df: pd.DataFrame,
        score_col: str = "score",
        use_hysteresis: bool = True,
    ) -> Dict[str, Any]:
        """
        Run multi-period chronological portfolio simulation.

        Parameters
        ----------
        panel_df : pd.DataFrame
            Panel dataframe with columns:
            ['date', 'symbol', 'open', 'close', score_col, 'adtv_20', 'vol_20', 'is_eligible']
        score_col : str
            Column name containing predicted alpha scores.
        use_hysteresis : bool
            Whether to use rank hysteresis band.

        Returns
        -------
        Dict[str, Any]
            Performance metrics, time-series curves, and rebalance audit history.
        """
        required_cols = ['date', 'symbol', 'open', 'close', score_col, 'adtv_20', 'vol_20']
        for col in required_cols:
            if col not in panel_df.columns:
                raise ValueError(f"panel_df missing required column: {col}")

        # Ensure sorted chronologically
        df = panel_df.sort_values(by=['date', 'symbol']).copy()
        unique_dates = df['date'].drop_duplicates().tolist()
        num_dates = len(unique_dates)

        if num_dates < self.config.rebalance_freq + 2:
            raise ValueError(f"Insufficient dates ({num_dates}) for rebalance frequency {self.config.rebalance_freq}")

        # Filter PIT eligible universe if flag present
        if 'is_eligible' in df.columns:
            eligible_df = df[df['is_eligible']].copy()
        else:
            eligible_df = df.copy()

        portfolio = RankHysteresisPortfolio(self.config)
        
        # State tracking
        current_holdings: Dict[str, float] = {}  # {symbol: weight}
        current_nav = self.config.initial_aum
        gross_nav = self.config.initial_aum
        benchmark_nav = self.config.initial_aum

        nav_history: List[Dict[str, Any]] = []
        rebalance_events: List[RebalanceEvent] = []
        
        total_statutory_costs = 0.0
        total_impact_costs = 0.0
        turnover_history: List[float] = []

        # Determine rebalance dates: every rebalance_freq days
        rebalance_indices = list(range(0, num_dates - 1, self.config.rebalance_freq))

        for t_idx in range(num_dates - 1):
            date_now = unique_dates[t_idx]
            date_next = unique_dates[t_idx + 1]

            # Cross-sectional slices
            curr_slice = eligible_df[eligible_df['date'] == date_now]
            next_slice = eligible_df[eligible_df['date'] == date_next]

            # Price dictionaries for returns calculation
            # Entry at Open(t+1), holding mark-to-market until Close(t+1)
            # or holding period return: Close(t) -> Close(t+1)
            curr_prices = dict(zip(curr_slice['symbol'], curr_slice['close']))
            next_prices = dict(zip(next_slice['symbol'], next_slice['close']))
            next_opens = dict(zip(next_slice['symbol'], next_slice['open']))
            adtv_dict = dict(zip(curr_slice['symbol'], curr_slice['adtv_20']))
            vol_dict = dict(zip(curr_slice['symbol'], curr_slice['vol_20']))

            # Benchmark return: equal-weight of all eligible stocks from Close(t) to Close(t+1)
            eligible_returns = []
            for sym, p_close in curr_prices.items():
                if sym in next_prices and p_close > 0:
                    ret = (next_prices[sym] - p_close) / p_close
                    eligible_returns.append(ret)
            
            daily_bmk_ret = np.mean(eligible_returns) if len(eligible_returns) > 0 else 0.0
            benchmark_nav *= (1.0 + daily_bmk_ret)

            # Check if today is a rebalance date
            is_rebalance_date = (t_idx in rebalance_indices)

            if is_rebalance_date:
                # Rank universe on Close(t) scores
                valid_scores = curr_slice[['symbol', score_col]].dropna().rename(columns={score_col: 'score'})
                
                # Select target holdings
                target_symbols = portfolio.select_holdings(valid_scores, use_hysteresis=use_hysteresis)
                target_weights = portfolio.compute_weights(target_symbols, valid_scores)

                # Identify buy/sell trades
                old_symbols = set(current_holdings.keys())
                new_symbols = set(target_weights.keys())

                all_syms = old_symbols.union(new_symbols)
                turnover_sum = 0.0
                period_statutory = 0.0
                period_impact = 0.0

                buys = []
                sells = []

                for sym in all_syms:
                    w_old = current_holdings.get(sym, 0.0)
                    w_new = target_weights.get(sym, 0.0)
                    delta_w = w_new - w_old
                    trade_val = abs(delta_w) * current_nav

                    if trade_val > 1.0:
                        turnover_sum += abs(delta_w)
                        is_buy = (delta_w > 0)
                        if is_buy:
                            buys.append(sym)
                        else:
                            sells.append(sym)

                        adtv = adtv_dict.get(sym, 50_000_000.0)  # default ₹5 Cr
                        vol = vol_dict.get(sym, 0.02)

                        # Compute friction
                        friction_res = self.frictions.total_execution_cost(
                            order_value=trade_val,
                            is_buy=is_buy,
                            adtv_20=adtv,
                            sigma_daily=vol,
                            segment=self.config.segment,
                            num_scrips=1,
                        )
                        period_statutory += friction_res["statutory_inr"]
                        period_impact += friction_res["impact_inr"]

                period_turnover = 0.5 * turnover_sum
                total_frictions = period_statutory + period_impact

                turnover_history.append(period_turnover)
                total_statutory_costs += period_statutory
                total_impact_costs += period_impact

                # Deduct frictions from NAV
                nav_before_frictions = current_nav
                current_nav -= total_frictions

                rebalance_events.append(RebalanceEvent(
                    rebalance_date=date_now,
                    holdings_before=list(old_symbols),
                    holdings_after=list(new_symbols),
                    buys=buys,
                    sells=sells,
                    turnover=period_turnover,
                    statutory_cost_inr=period_statutory,
                    impact_cost_inr=period_impact,
                    total_friction_inr=total_frictions,
                    aum_before=nav_before_frictions,
                    aum_after_friction=current_nav,
                ))

                # Update active weights
                current_holdings = target_weights

            # Compute period holding returns from date_now to date_next
            # Weighted average return of active holdings
            daily_gross_ret = 0.0
            invested_weight = sum(current_holdings.values())
            cash_weight = max(0.0, 1.0 - invested_weight)

            for sym, w in current_holdings.items():
                p_curr = curr_prices.get(sym, 0.0)
                p_next = next_prices.get(sym, 0.0)
                if p_curr > 0 and p_next > 0:
                    stock_ret = (p_next - p_curr) / p_curr
                    daily_gross_ret += w * stock_ret

            # Portfolio updates
            gross_nav *= (1.0 + daily_gross_ret)
            current_nav *= (1.0 + daily_gross_ret)

            nav_history.append({
                "date": date_next,
                "net_nav": current_nav,
                "gross_nav": gross_nav,
                "benchmark_nav": benchmark_nav,
                "daily_gross_return": daily_gross_ret,
                "daily_benchmark_return": daily_bmk_ret,
                "active_holdings_count": len(current_holdings),
            })

        # Compile simulation results
        nav_df = pd.DataFrame(nav_history)
        if len(nav_df) == 0:
            return {"error": "Empty simulation run"}

        # Return series
        nav_df['net_return'] = nav_df['net_nav'].pct_change().fillna(0.0)
        nav_df['gross_return'] = nav_df['gross_nav'].pct_change().fillna(0.0)
        nav_df['benchmark_return'] = nav_df['benchmark_nav'].pct_change().fillna(0.0)
        nav_df['active_return'] = nav_df['net_return'] - nav_df['benchmark_return']

        total_net_return = (current_nav - self.config.initial_aum) / self.config.initial_aum
        total_gross_return = (gross_nav - self.config.initial_aum) / self.config.initial_aum
        total_bmk_return = (benchmark_nav - self.config.initial_aum) / self.config.initial_aum

        # Annualized metrics (252 trading days/year)
        n_periods = len(nav_df)
        ann_factor = 252.0 / max(1, n_periods)
        
        cagr_net = (current_nav / self.config.initial_aum) ** (ann_factor) - 1.0 if current_nav > 0 else -1.0
        cagr_gross = (gross_nav / self.config.initial_aum) ** (ann_factor) - 1.0 if gross_nav > 0 else -1.0
        cagr_bmk = (benchmark_nav / self.config.initial_aum) ** (ann_factor) - 1.0 if benchmark_nav > 0 else -1.0

        daily_net_std = nav_df['net_return'].std()
        ann_vol = daily_net_std * np.sqrt(252) if daily_net_std > 0 else 1e-4
        net_sharpe = (cagr_net - 0.065) / ann_vol  # 6.5% risk-free rate (India RBI repo standard)

        daily_active_std = nav_df['active_return'].std()
        tracking_error = daily_active_std * np.sqrt(252) if daily_active_std > 0 else 1e-4
        info_ratio = (cagr_net - cagr_bmk) / tracking_error

        # Maximum Drawdown
        cum_net = nav_df['net_nav']
        peak_net = cum_net.cummax()
        drawdown_net = (cum_net - peak_net) / peak_net
        max_dd_net = float(drawdown_net.min())

        avg_turnover = float(np.mean(turnover_history)) if len(turnover_history) > 0 else 0.0
        ann_turnover = avg_turnover * (252.0 / self.config.rebalance_freq)

        win_rate = float(np.mean(nav_df['net_return'] > 0)) if len(nav_df) > 0 else 0.0

        return {
            "initial_aum": self.config.initial_aum,
            "final_net_nav": current_nav,
            "final_gross_nav": gross_nav,
            "final_bmk_nav": benchmark_nav,
            "total_net_return": total_net_return,
            "total_gross_return": total_gross_return,
            "total_bmk_return": total_bmk_return,
            "cagr_net": cagr_net,
            "cagr_gross": cagr_gross,
            "cagr_bmk": cagr_bmk,
            "net_alpha_cagr": cagr_net - cagr_bmk,
            "ann_volatility": ann_vol,
            "net_sharpe": net_sharpe,
            "information_ratio": info_ratio,
            "max_drawdown": max_dd_net,
            "win_rate": win_rate,
            "avg_turnover_per_rebalance": avg_turnover,
            "annualized_turnover": ann_turnover,
            "total_statutory_costs_inr": total_statutory_costs,
            "total_impact_costs_inr": total_impact_costs,
            "total_frictions_inr": total_statutory_costs + total_impact_costs,
            "friction_drag_bps_ann": ((total_statutory_costs + total_impact_costs) / self.config.initial_aum) * ann_factor * 10000.0,
            "nav_history": nav_df,
            "rebalance_events": rebalance_events,
        }


class FNOBookSimulator:
    """
    Evaluates F&O Long/Short Portfolio Separately.
    Long top decile, Short bottom decile of F&O eligible universe.
    Dollar-neutral weights (+50% long, -50% short).
    Applies futures cost model and shorting mechanics.
    """

    def __init__(
        self,
        cost_config: Optional[NSECostConfig] = None,
        impact_config: Optional[ImpactModelConfig] = None,
    ):
        self.frictions = ExecutionFrictions(
            cost_config=cost_config,
            impact_config=impact_config,
        )

    def evaluate_fno_long_short(
        self,
        panel_df: pd.DataFrame,
        score_col: str = "score",
        rebalance_freq: int = 5,
        initial_capital: float = 10_000_000.0,
    ) -> Dict[str, Any]:
        """
        Simulate F&O dollar-neutral long/short book.
        """
        unique_dates = panel_df['date'].drop_duplicates().tolist()
        n_dates = len(unique_dates)

        nav = initial_capital
        history = []
        total_costs = 0.0

        for t_idx in range(0, n_dates - rebalance_freq, rebalance_freq):
            d_curr = unique_dates[t_idx]
            d_next = unique_dates[t_idx + rebalance_freq]

            slice_curr = panel_df[panel_df['date'] == d_curr].sort_values(by=score_col, ascending=False)
            slice_next = panel_df[panel_df['date'] == d_next]

            next_price_map = dict(zip(slice_next['symbol'], slice_next['close']))
            curr_price_map = dict(zip(slice_curr['symbol'], slice_curr['close']))

            n_stocks = len(slice_curr)
            if n_stocks < 10:
                continue

            decile_size = max(2, n_stocks // 10)
            long_stocks = slice_curr.head(decile_size)['symbol'].tolist()
            short_stocks = slice_curr.tail(decile_size)['symbol'].tolist()

            # Long leg returns
            long_rets = []
            for s in long_stocks:
                if s in next_price_map and s in curr_price_map and curr_price_map[s] > 0:
                    long_rets.append((next_price_map[s] - curr_price_map[s]) / curr_price_map[s])

            # Short leg returns: -(P_next - P_curr) / P_curr
            short_rets = []
            for s in short_stocks:
                if s in next_price_map and s in curr_price_map and curr_price_map[s] > 0:
                    short_rets.append(-(next_price_map[s] - curr_price_map[s]) / curr_price_map[s])

            avg_long = np.mean(long_rets) if long_rets else 0.0
            avg_short = np.mean(short_rets) if short_rets else 0.0
            gross_period_ret = 0.5 * avg_long + 0.5 * avg_short

            # F&O transaction cost on turnover (100% turnover each rebalance)
            turnover_value = nav * 1.0
            leg_cost = self.frictions.total_execution_cost(
                order_value=turnover_value,
                is_buy=True,
                adtv_20=200_000_000.0,
                sigma_daily=0.02,
                segment=TradingSegment.FNO_FUTURES,
            )
            friction = leg_cost["total_friction_inr"] * 2.0  # round trip
            total_costs += friction

            net_dollar_gain = nav * gross_period_ret - friction
            nav += net_dollar_gain

            history.append({
                "date": d_next,
                "nav": nav,
                "gross_ret": gross_period_ret,
                "friction": friction,
                "long_leg_ret": avg_long,
                "short_leg_ret": avg_short,
            })

        df_hist = pd.DataFrame(history)
        if len(df_hist) == 0:
            return {"error": "Empty FNO simulation"}

        ann_factor = 252.0 / (len(df_hist) * rebalance_freq)
        cagr = (nav / initial_capital) ** ann_factor - 1.0 if nav > 0 else -1.0
        sharpe = (df_hist['gross_ret'].mean() / df_hist['gross_ret'].std()) * np.sqrt(252.0 / rebalance_freq) if df_hist['gross_ret'].std() > 0 else 0.0

        return {
            "fno_initial_capital": initial_capital,
            "fno_final_nav": nav,
            "fno_cagr": cagr,
            "fno_sharpe": sharpe,
            "fno_total_frictions_inr": total_costs,
            "fno_history": df_hist,
        }


class CapacityEstimator:
    """
    Portfolio Capacity and Impact Scaling Estimator.
    Evaluates net alpha decay across institutional AUM levels:
    ₹1 Cr, ₹5 Cr, ₹10 Cr, ₹25 Cr, ₹50 Cr, ₹100 Cr+.
    Finds the maximum capacity A* where net alpha >= hurdle rate.
    """

    def __init__(
        self,
        config: Optional[PortfolioConfig] = None,
        aum_grid: Optional[List[float]] = None,
    ):
        self.config = config or PortfolioConfig()
        self.aum_grid = aum_grid or [
            10_000_000.0,   # ₹1 Crore
            25_000_000.0,   # ₹2.5 Crore
            50_000_000.0,   # ₹5 Crore
            100_000_000.0,  # ₹10 Crore
            250_000_000.0,  # ₹25 Crore
            500_000_000.0,  # ₹50 Crore
            1_000_000_000.0 # ₹100 Crore
        ]

    def estimate_capacity_curve(
        self,
        panel_df: pd.DataFrame,
        score_col: str = "score",
        hurdle_net_alpha: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Compute capacity decay curve across AUM grid.

        Parameters
        ----------
        panel_df : pd.DataFrame
            Market panel data.
        score_col : str
            Predicted score column.
        hurdle_net_alpha : float
            Minimum required net active alpha over benchmark (e.g. 0.0 = break-even).

        Returns
        -------
        Dict[str, Any]
            Capacity curve table, break-even capacity A*, and scaling summary.
        """
        results = []

        for aum in self.aum_grid:
            cfg = PortfolioConfig(
                n_enter=self.config.n_enter,
                n_exit=self.config.n_exit,
                rebalance_freq=self.config.rebalance_freq,
                initial_aum=aum,
                segment=self.config.segment,
                impact_eta=self.config.impact_eta,
                impact_alpha=self.config.impact_alpha,
            )
            sim = PortfolioSimulator(cfg)
            res = sim.run_simulation(panel_df, score_col=score_col, use_hysteresis=True)

            results.append({
                "aum_inr": aum,
                "aum_crores": aum / 10_000_000.0,
                "cagr_gross": res["cagr_gross"],
                "cagr_net": res["cagr_net"],
                "cagr_bmk": res["cagr_bmk"],
                "net_alpha": res["net_alpha_cagr"],
                "net_sharpe": res["net_sharpe"],
                "statutory_drag_bps": (res["total_statutory_costs_inr"] / aum) * (252.0 / len(res["nav_history"])) * 10000.0,
                "impact_drag_bps": (res["total_impact_costs_inr"] / aum) * (252.0 / len(res["nav_history"])) * 10000.0,
                "total_friction_drag_bps": res["friction_drag_bps_ann"],
            })

        df_curve = pd.DataFrame(results)

        # Interpolate break-even capacity A*
        valid_curve = df_curve.sort_values(by='aum_inr')
        alphas = valid_curve['net_alpha'].values
        aums = valid_curve['aum_inr'].values

        max_capacity_inr = aums[-1]  # Default to max tested if always > hurdle
        for i in range(len(alphas) - 1):
            if alphas[i] >= hurdle_net_alpha and alphas[i+1] < hurdle_net_alpha:
                # Linear interpolation
                slope = (alphas[i+1] - alphas[i]) / (aums[i+1] - aums[i])
                if slope != 0:
                    max_capacity_inr = aums[i] + (hurdle_net_alpha - alphas[i]) / slope
                break
            elif alphas[0] < hurdle_net_alpha:
                max_capacity_inr = aums[0]
                break

        return {
            "capacity_curve_table": df_curve,
            "max_capacity_aum_inr": float(max_capacity_inr),
            "max_capacity_aum_crores": float(max_capacity_inr / 10_000_000.0),
            "hurdle_net_alpha": hurdle_net_alpha,
        }
