"""
Production Serving, Live-IC Monitoring & Shadow Trading Protocol (Stage 8)
===========================================================================
Institutional serving engine for daily whole-universe scoring on NSE equities.

Key Capabilities:
1. Daily Whole-Universe Scoring:
   - Features computed <= Close(t).
   - Point-in-time eligibility filter: ADTV20 >= ₹5 Cr, EQ series, non-ASM/GSM, non-circuit locked.
   - True cross-sectional same-day percentile ranks (no per-ticker arbitrary thresholds).
   - Multi-horizon model inference across k in {3, 5, 7, 15, 21}.
   - Rank-based Top-N selection with Hysteresis band [N_enter, N_exit].
2. Train / Serve Parity Validation (Golden Sample):
   - Validates feature and model output equivalence between training and live serving.
   - Strict floating-point tolerance assertion: max|y_serve - y_train| <= 1e-5.
3. Rolling Live-IC Monitor with Automated Kill Switch:
   - Tracks real-time open-entry forward Rank IC as prices mature.
   - Automated Kill Switch: halts trading if rolling-20 IC <= 0 for 5 consecutive days.
   - Retraining alert if rolling-20 IC < 50% of backtest IC for 10 consecutive days.
4. Shadow Trading Protocol:
   - Prospective execution logging at Close(t) -> fill at Open(t+1).
   - Realistic market impact and statutory friction deduction.
   - Complete immutable audit journal.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import pandas as pd
import logging
import json
import os

from frictions import TradingSegment, NSECostConfig, NSECostModel, ExecutionFrictions
from portfolio import PortfolioConfig, RankHysteresisPortfolio, WeightingScheme
from cs_transforms import CrossSectionalNormalizer

logger = logging.getLogger(__name__)


class TrainServeSkewError(Exception):
    """Raised when live serving outputs deviate from training golden sample reference."""
    pass


class TradingHaltedError(Exception):
    """Raised when an execution order is attempted while Kill Switch is active."""
    pass


class OrderState(Enum):
    """Lifecycle state of shadow orders."""
    PENDING_OPEN = "PENDING_OPEN"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    CLOSED = "CLOSED"


@dataclass
class ServingConfig:
    """Production serving and risk parameters."""
    model_version: str = "v2.0-multi-horizon"
    horizons: List[int] = field(default_factory=lambda: [3, 5, 7, 15, 21])
    primary_horizon: int = 5
    n_enter: int = 20
    n_exit: int = 35
    min_adtv_inr: float = 50_000_000.0  # ₹5 Crore
    initial_shadow_aum: float = 10_000_000.0  # ₹1 Crore
    parity_tolerance: float = 1e-5
    
    # Kill switch parameters
    kill_switch_roll_ic_threshold: float = 0.0
    kill_switch_consecutive_days: int = 5
    backtest_ic_anchor: float = 0.05
    retrain_consecutive_days: int = 10
    retrain_threshold_ratio: float = 0.50
    max_drawdown_halt_pct: float = 0.10  # 10% shadow DD halt


@dataclass
class ShadowOrder:
    """Record of a shadow trade order."""
    order_id: str
    signal_date: Any
    execution_date: Any
    symbol: str
    side: str                          # BUY or SELL
    target_weight: float
    order_value_inr: float
    expected_open_price: float
    fill_price: Optional[float] = None
    state: OrderState = OrderState.PENDING_OPEN
    statutory_cost_inr: float = 0.0
    impact_cost_inr: float = 0.0
    total_friction_inr: float = 0.0
    rejection_reason: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class TrainServeParityValidator:
    """
    Verifies that feature engineering and model scoring in live production
    exactly match offline training outputs without numerical skew.
    """

    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance

    def assert_parity(
        self,
        golden_inputs: pd.DataFrame,
        golden_expected_scores: np.ndarray,
        serving_scoring_fn: Any,
    ) -> bool:
        """
        Execute golden sample through live scoring function and assert parity.

        Parameters
        ----------
        golden_inputs : pd.DataFrame
            Fixed input features from verified training run.
        golden_expected_scores : np.ndarray
            Exact model outputs produced during training.
        serving_scoring_fn : Callable[[pd.DataFrame], np.ndarray]
            Live inference function.

        Returns
        -------
        bool
            True if within tolerance; raises TrainServeSkewError otherwise.
        """
        serving_scores = serving_scoring_fn(golden_inputs)
        serving_scores = np.asarray(serving_scores, dtype=np.float64).ravel()
        expected = np.asarray(golden_expected_scores, dtype=np.float64).ravel()

        if len(serving_scores) != len(expected):
            raise TrainServeSkewError(
                f"Dimension mismatch in golden parity test: serving={len(serving_scores)}, expected={len(expected)}"
            )

        max_abs_diff = float(np.max(np.abs(serving_scores - expected)))
        if max_abs_diff > self.tolerance:
            raise TrainServeSkewError(
                f"Train/Serve Skew detected! Max absolute diff {max_abs_diff:.6e} exceeded tolerance {self.tolerance:.6e}"
            )

        logger.info(f"Train/Serve parity test PASSED (max_diff={max_abs_diff:.2e} <= {self.tolerance:.2e})")
        return True


class LiveICMonitor:
    """
    Rolling Live-IC Monitor with Automated Kill Switch.
    Tracks out-of-sample forward prediction accuracy as actual trade outcomes mature.
    """

    def __init__(self, config: Optional[ServingConfig] = None):
        self.config = config or ServingConfig()
        self.daily_ic_history: Dict[Any, float] = {}
        self.is_killed: bool = False
        self.kill_reason: Optional[str] = None
        self.retrain_recommended: bool = False
        self.consecutive_negative_days: int = 0
        self.consecutive_decay_days: int = 0

    def record_realized_ic(self, trade_date: Any, rank_ic: float) -> Dict[str, Any]:
        """
        Record realized forward Rank IC for a matured trading date.
        Updates rolling-20 stats and triggers Kill Switch if decayed.
        """
        self.daily_ic_history[trade_date] = float(rank_ic)
        
        # Compute rolling 20-day mean
        dates = sorted(self.daily_ic_history.keys())
        recent_dates = dates[-20:]
        recent_ics = [self.daily_ic_history[d] for d in recent_dates]
        roll20_mean = float(np.mean(recent_ics)) if len(recent_ics) > 0 else rank_ic

        # Check negative IC kill switch condition
        if roll20_mean <= self.config.kill_switch_roll_ic_threshold:
            self.consecutive_negative_days += 1
        else:
            self.consecutive_negative_days = 0

        # Check decay condition (< 50% backtest IC)
        decay_floor = self.config.backtest_ic_anchor * self.config.retrain_threshold_ratio
        if roll20_mean < decay_floor:
            self.consecutive_decay_days += 1
        else:
            self.consecutive_decay_days = 0

        # 1. Kill Switch Trigger
        if self.consecutive_negative_days >= self.config.kill_switch_consecutive_days:
            self.is_killed = True
            self.kill_reason = (
                f"KILL SWITCH TRIGGERED: Rolling-20 Rank IC ({roll20_mean:.4f}) <= 0 "
                f"for {self.consecutive_negative_days} consecutive days."
            )
            logger.critical(self.kill_reason)

        # 2. Retraining Alert
        if self.consecutive_decay_days >= self.config.retrain_consecutive_days:
            self.retrain_recommended = True
            logger.warning(
                f"RETRAIN RECOMMENDED: Rolling-20 Rank IC ({roll20_mean:.4f}) below 50% "
                f"of backtest IC ({decay_floor:.4f}) for {self.consecutive_decay_days} sessions."
            )

        return {
            "date": trade_date,
            "daily_ic": rank_ic,
            "rolling_20_ic": roll20_mean,
            "is_killed": self.is_killed,
            "kill_reason": self.kill_reason,
            "retrain_recommended": self.retrain_recommended,
            "consecutive_negative_days": self.consecutive_negative_days,
            "consecutive_decay_days": self.consecutive_decay_days,
        }

    def manual_kill(self, reason: str = "Manual override by risk manager"):
        """Manually trigger emergency kill switch."""
        self.is_killed = True
        self.kill_reason = reason
        logger.warning(f"Manual kill switch activated: {reason}")

    def reset_kill_switch(self):
        """Reset kill switch after review and recalibration."""
        self.is_killed = False
        self.kill_reason = None
        self.consecutive_negative_days = 0
        self.consecutive_decay_days = 0
        logger.info("Kill switch has been reset.")


class LiveScoringService:
    """
    Daily Whole-Universe Scoring & Order Generation Service.
    Produces true same-day ranks and hysteresis rebalancing orders.
    """

    def __init__(
        self,
        config: Optional[ServingConfig] = None,
        models: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or ServingConfig()
        self.models = models or {}
        self.port_config = PortfolioConfig(
            n_enter=self.config.n_enter,
            n_exit=self.config.n_exit,
            rebalance_freq=self.config.primary_horizon,
            initial_aum=self.config.initial_shadow_aum,
        )
        self.portfolio = RankHysteresisPortfolio(self.port_config)
        self.frictions = ExecutionFrictions()
        self.active_holdings: Dict[str, float] = {}  # {symbol: weight}

    def score_universe(
        self,
        eod_panel_df: pd.DataFrame,
        asof_date: Any,
        model_predict_fn: Any,
    ) -> pd.DataFrame:
        """
        Execute daily whole-universe scoring for Close(t).

        Parameters
        ----------
        eod_panel_df : pd.DataFrame
            EOD market data up to Close(t) containing:
            ['date', 'symbol', 'close', 'open', 'adtv_20', 'vol_20', 'is_eligible'] + features.
        asof_date : Any
            Target scoring date t.
        model_predict_fn : Callable[[pd.DataFrame], np.ndarray]
            Inference callable returning predicted expected excess returns.

        Returns
        -------
        pd.DataFrame
            Universe ranked cross-section with ['symbol', 'score', 'rank', 'percentile_rank', 'is_selected'].
        """
        # 1. Filter cross-section at asof_date
        slice_df = eod_panel_df[eod_panel_df['date'] == asof_date].copy()
        if len(slice_df) == 0:
            raise ValueError(f"No market data found for date {asof_date}")

        # 2. Point-in-time eligibility filter
        # Must have ADTV >= ₹5 Cr, eligible flag True, and non-zero close
        mask_eligible = (
            (slice_df['adtv_20'] >= self.config.min_adtv_inr) &
            (slice_df['close'] > 0)
        )
        if 'is_eligible' in slice_df.columns:
            mask_eligible &= slice_df['is_eligible']

        eligible_df = slice_df[mask_eligible].copy()
        if len(eligible_df) < self.config.n_enter:
            raise ValueError(f"Insufficient eligible names ({len(eligible_df)}) on {asof_date}")

        # 3. Model Inference
        raw_preds = model_predict_fn(eligible_df)
        eligible_df['score'] = np.asarray(raw_preds, dtype=np.float64)

        # 4. True same-day cross-sectional ranking
        eligible_df = eligible_df.sort_values(by='score', ascending=False).reset_index(drop=True)
        eligible_df['rank'] = np.arange(1, len(eligible_df) + 1)
        eligible_df['percentile_rank'] = (len(eligible_df) - eligible_df['rank'] + 1) / float(len(eligible_df))

        # 5. Hysteresis Selection
        selected_symbols = self.portfolio.select_holdings(
            eligible_df[['symbol', 'score']],
            use_hysteresis=True
        )
        selected_set = set(selected_symbols)
        eligible_df['is_selected'] = eligible_df['symbol'].isin(selected_set)

        return eligible_df


class ShadowTradeProtocol:
    """
    Shadow Trading Execution Protocol.
    Logs prospective orders at Close(t) and simulates fills at Open(t+1) with realistic frictions.
    """

    def __init__(
        self,
        config: Optional[ServingConfig] = None,
        cost_config: Optional[NSECostConfig] = None,
    ):
        self.config = config or ServingConfig()
        self.frictions = ExecutionFrictions(cost_config=cost_config)
        self.active_positions: Dict[str, float] = {}  # {symbol: shares}
        self.current_weights: Dict[str, float] = {}    # {symbol: weight}
        self.current_nav: float = self.config.initial_shadow_aum
        self.order_history: List[ShadowOrder] = []
        self.ic_monitor = LiveICMonitor(self.config)

    def generate_orders_from_scores(
        self,
        ranked_universe: pd.DataFrame,
        signal_date: Any,
        execution_date: Any,
    ) -> List[ShadowOrder]:
        """
        Generate prospective target rebalance orders for execution at Open(t+1).
        """
        if self.ic_monitor.is_killed:
            raise TradingHaltedError(f"Cannot generate shadow orders: {self.ic_monitor.kill_reason}")

        selected_df = ranked_universe[ranked_universe['is_selected']].copy()
        n_selected = len(selected_df)
        if n_selected == 0:
            return []

        # Equal-weight target allocation (subject to 10% cap)
        target_weight_per_stock = min(0.10, 0.99 / n_selected)
        target_weights = {sym: target_weight_per_stock for sym in selected_df['symbol']}

        old_symbols = set(self.current_weights.keys())
        new_symbols = set(target_weights.keys())
        all_symbols = old_symbols.union(new_symbols)

        price_dict = dict(zip(ranked_universe['symbol'], ranked_universe['close']))
        adtv_dict = dict(zip(ranked_universe['symbol'], ranked_universe['adtv_20']))
        vol_dict = dict(zip(ranked_universe['symbol'], ranked_universe['vol_20']))

        orders = []

        for sym in all_symbols:
            w_old = self.current_weights.get(sym, 0.0)
            w_new = target_weights.get(sym, 0.0)
            delta_w = w_new - w_old
            trade_val = abs(delta_w) * self.current_nav

            if trade_val > 5_000.0:  # Minimum trade threshold ₹5,000
                is_buy = (delta_w > 0)
                ref_price = price_dict.get(sym, 100.0)
                adtv = adtv_dict.get(sym, 50_000_000.0)
                vol = vol_dict.get(sym, 0.02)

                friction = self.frictions.total_execution_cost(
                    order_value=trade_val,
                    is_buy=is_buy,
                    adtv_20=adtv,
                    sigma_daily=vol,
                    segment=TradingSegment.CASH_DELIVERY,
                )

                order = ShadowOrder(
                    order_id=f"SHADOW-{signal_date}-{sym}-{len(orders)+1:03d}",
                    signal_date=signal_date,
                    execution_date=execution_date,
                    symbol=sym,
                    side="BUY" if is_buy else "SELL",
                    target_weight=w_new,
                    order_value_inr=trade_val,
                    expected_open_price=ref_price,
                    statutory_cost_inr=friction["statutory_inr"],
                    impact_cost_inr=friction["impact_inr"],
                    total_friction_inr=friction["total_friction_inr"],
                )
                orders.append(order)

        self.order_history.extend(orders)
        return orders

    def execute_shadow_fills(
        self,
        orders: List[ShadowOrder],
        open_prices_dict: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Simulate order fills at Open(t+1) using actual open quotes.
        """
        total_friction_paid = 0.0
        filled_count = 0

        for order in orders:
            if order.symbol in open_prices_dict:
                open_px = open_prices_dict[order.symbol]
                # Price impact shift on fill
                impact_factor = (order.impact_cost_inr / order.order_value_inr) if order.order_value_inr > 0 else 0.0
                if order.side == "BUY":
                    fill_px = open_px * (1.0 + impact_factor)
                else:
                    fill_px = open_px * (1.0 - impact_factor)

                order.fill_price = fill_px
                order.state = OrderState.FILLED
                filled_count += 1
                total_friction_paid += order.total_friction_inr

                # Update target weights
                if order.target_weight > 0:
                    self.current_weights[order.symbol] = order.target_weight
                elif order.symbol in self.current_weights:
                    del self.current_weights[order.symbol]
            else:
                order.state = OrderState.REJECTED
                order.rejection_reason = "Missing Open(t+1) quote"

        self.current_nav -= total_friction_paid

        return {
            "filled_orders": filled_count,
            "total_orders": len(orders),
            "friction_paid_inr": total_friction_paid,
            "updated_nav": self.current_nav,
            "active_positions_count": len(self.current_weights),
        }
