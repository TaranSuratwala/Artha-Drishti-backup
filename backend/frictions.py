"""
NSE Execution Frictions and Statutory Cost Model
================================================
Standardized institutional transaction cost and market impact model for NSE equities.

Friction components configured per statutory circulars:
- Securities Transaction Tax (STT): 0.1% Buy AND 0.1% Sell for Cash Delivery (0.2% round trip).
  Source: Finance Act / SEBI STT schedule (verify against contract note).
- NSE Exchange Transaction Charges: 0.00297% (0.297 bps) of trade value.
  Source: NSE Circular NSE/CM/19/2023 & True-to-label circular Oct 2024 (verify against contract note).
- SEBI Turnover Fee: ₹10 per crore (0.0001% = 0.01 bps) of trade value.
  Source: SEBI (Payment of Fees) Regulations (verify against contract note).
- Stamp Duty: 0.015% (1.5 bps) on Buy side only for delivery.
  Source: Indian Stamp (Collection of Stamp-Duty through Stock Exchanges) Rules 2019 (verify against contract note).
- Goods and Services Tax (GST): 18% on (Brokerage + Exchange Txn Charge + SEBI Fee).
  Source: CGST/SGST Acts (verify against contract note).
- Depository Participant (DP) Charge: ₹13.50 + 18% GST = ₹15.93 per scrip per day on Sell.
  Source: CDSL / Zerodha Tariff Schedule (verify against contract note).
- Brokerage: ₹0 for Equity Delivery (Zerodha discount broker standard).
  Source: Brokerage Tariff Sheet (verify against contract note).

Market Impact Model:
- Almgren-Chriss square-root impact formulation:
  Cost_impact = eta * (Q / ADTV_20)^alpha * sigma_daily
  where Q is order value/shares, ADTV_20 is 20-day median turnover/volume,
  sigma_daily is 20d daily volatility, eta=0.10, alpha=0.50.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


class TradingSegment(Enum):
    """Trading segment on National Stock Exchange of India (NSE)."""
    CASH_DELIVERY = "cash_delivery"
    CASH_INTRADAY = "cash_intraday"
    FNO_FUTURES = "fno_futures"
    FNO_OPTIONS = "fno_options"


@dataclass
class NSECostConfig:
    """
    Statutory NSE transaction fee configuration.
    All parameters must be verified against official contract notes.
    """
    # Securities Transaction Tax (STT)
    stt_delivery_buy: float = 0.0010     # 0.10% on Buy value (verify against contract note)
    stt_delivery_sell: float = 0.0010    # 0.10% on Sell value (verify against contract note)
    stt_intraday_buy: float = 0.0        # 0.0% on Buy (verify against contract note)
    stt_intraday_sell: float = 0.00025   # 0.025% on Sell value (verify against contract note)
    stt_futures_sell: float = 0.00020    # 0.02% on Sell turnover (Finance Act 2024 / verify against contract note)
    stt_options_sell: float = 0.0010     # 0.10% on Sell premium (Finance Act 2024 / verify against contract note)

    # Exchange Transaction Charges (NSE)
    nse_txn_cash: float = 0.0000297      # 0.00297% (0.297 bps) (NSE circular / verify against contract note)
    nse_txn_futures: float = 0.0000173   # 0.00173% (0.173 bps) (verify against contract note)
    nse_txn_options: float = 0.0003503   # 0.03503% (3.503 bps on premium) (verify against contract note)

    # SEBI Turnover Fee
    sebi_fee_rate: float = 0.000001      # ₹10 per crore = 0.0001% (verify against contract note)

    # Stamp Duty (Indian Stamp Act, Buy side only)
    stamp_duty_delivery: float = 0.00015 # 0.015% (1.5 bps) on Buy only (verify against contract note)
    stamp_duty_intraday: float = 0.00003 # 0.003% (0.3 bps) on Buy only (verify against contract note)
    stamp_duty_futures: float = 0.00002  # 0.002% (0.2 bps) on Buy only (verify against contract note)
    stamp_duty_options: float = 0.00003  # 0.003% (0.3 bps) on Buy premium (verify against contract note)

    # Goods and Services Tax (GST)
    gst_rate: float = 0.18               # 18% on (Brokerage + Exchange Charge + SEBI Fee) (verify against contract note)

    # Depository Participant (DP) Charge (CDSL / Zerodha)
    dp_charge_base: float = 13.50        # ₹13.50 per scrip per day on delivery sell (verify against contract note)
    # Total DP = dp_charge_base * (1 + gst_rate) = ₹15.93

    # Brokerage Schedule
    brokerage_delivery: float = 0.0      # ₹0 for cash delivery (Zerodha / discount broker standard)
    brokerage_flat_order: float = 20.0   # ₹20 flat per order for F&O/intraday (verify against contract note)
    brokerage_pct_cap: float = 0.0003    # 0.03% cap for intraday (verify against contract note)


class NSECostModel:
    """
    High-precision statutory NSE fee calculator for institutional and algorithmic trading.
    Computes leg-by-leg and round-trip execution costs down to the paisa.
    """

    def __init__(self, config: Optional[NSECostConfig] = None):
        self.config = config or NSECostConfig()

    def compute_leg(
        self,
        trade_value: float,
        is_buy: bool,
        segment: TradingSegment = TradingSegment.CASH_DELIVERY,
        num_scrips: int = 1,
    ) -> Dict[str, float]:
        """
        Compute transaction cost breakdown for a single trade leg.

        Parameters
        ----------
        trade_value : float
            Total traded turnover in INR (price * quantity).
        is_buy : bool
            True for BUY / LONG order, False for SELL / SHORT order.
        segment : TradingSegment
            CASH_DELIVERY, CASH_INTRADAY, FNO_FUTURES, or FNO_OPTIONS.
        num_scrips : int
            Number of distinct scrips sold (for DP charge calculation).

        Returns
        -------
        Dict[str, float]
            Dictionary containing itemized costs and total in INR.
        """
        if trade_value <= 0:
            return {
                "brokerage": 0.0,
                "stt": 0.0,
                "exchange_txn": 0.0,
                "sebi_fee": 0.0,
                "gst": 0.0,
                "stamp_duty": 0.0,
                "dp_charge": 0.0,
                "total": 0.0,
                "bps": 0.0,
            }

        # 1. Brokerage
        if segment == TradingSegment.CASH_DELIVERY:
            brokerage = self.config.brokerage_delivery * trade_value
        elif segment in (TradingSegment.CASH_INTRADAY, TradingSegment.FNO_FUTURES):
            brokerage = min(self.config.brokerage_flat_order, trade_value * self.config.brokerage_pct_cap)
        else:  # FNO_OPTIONS
            brokerage = self.config.brokerage_flat_order

        # 2. STT
        if segment == TradingSegment.CASH_DELIVERY:
            stt = trade_value * (self.config.stt_delivery_buy if is_buy else self.config.stt_delivery_sell)
        elif segment == TradingSegment.CASH_INTRADAY:
            stt = trade_value * (self.config.stt_intraday_buy if is_buy else self.config.stt_intraday_sell)
        elif segment == TradingSegment.FNO_FUTURES:
            stt = trade_value * self.config.stt_futures_sell if not is_buy else 0.0
        else:  # FNO_OPTIONS
            stt = trade_value * self.config.stt_options_sell if not is_buy else 0.0

        # 3. Exchange Transaction Charges
        if segment in (TradingSegment.CASH_DELIVERY, TradingSegment.CASH_INTRADAY):
            exchange_txn = trade_value * self.config.nse_txn_cash
        elif segment == TradingSegment.FNO_FUTURES:
            exchange_txn = trade_value * self.config.nse_txn_futures
        else:
            exchange_txn = trade_value * self.config.nse_txn_options

        # 4. SEBI Turnover Fee
        sebi_fee = trade_value * self.config.sebi_fee_rate

        # 5. GST (18% on Brokerage + Exchange Txn + SEBI Fee)
        # Note: Stamp duty and STT are government taxes and are not taxable under GST.
        gst = (brokerage + exchange_txn + sebi_fee) * self.config.gst_rate

        # 6. Stamp Duty (Buy side only in India under Indian Stamp Act)
        if is_buy:
            if segment == TradingSegment.CASH_DELIVERY:
                stamp_duty = trade_value * self.config.stamp_duty_delivery
            elif segment == TradingSegment.CASH_INTRADAY:
                stamp_duty = trade_value * self.config.stamp_duty_intraday
            elif segment == TradingSegment.FNO_FUTURES:
                stamp_duty = trade_value * self.config.stamp_duty_futures
            else:
                stamp_duty = trade_value * self.config.stamp_duty_options
        else:
            stamp_duty = 0.0

        # 7. DP Charge (CDSL: ₹13.50 + 18% GST = ₹15.93 per scrip per day on Sell of cash delivery)
        if (not is_buy) and segment == TradingSegment.CASH_DELIVERY:
            dp_charge = self.config.dp_charge_base * (1.0 + self.config.gst_rate) * num_scrips
        else:
            dp_charge = 0.0

        total = brokerage + stt + exchange_txn + sebi_fee + gst + stamp_duty + dp_charge
        bps = (total / trade_value) * 10000.0

        return {
            "brokerage": brokerage,
            "stt": stt,
            "exchange_txn": exchange_txn,
            "sebi_fee": sebi_fee,
            "gst": gst,
            "stamp_duty": stamp_duty,
            "dp_charge": dp_charge,
            "total": total,
            "bps": bps,
        }

    def compute_round_trip(
        self,
        entry_value: float,
        exit_value: float,
        segment: TradingSegment = TradingSegment.CASH_DELIVERY,
        num_scrips: int = 1,
    ) -> Dict[str, Any]:
        """
        Compute total round-trip statutory costs (Buy leg + Sell leg).

        Parameters
        ----------
        entry_value : float
            Gross value of entry leg in INR.
        exit_value : float
            Gross value of exit leg in INR.
        segment : TradingSegment
            Trading segment.
        num_scrips : int
            Number of distinct scrips traded.

        Returns
        -------
        Dict[str, Any]
            Itemized breakdown and total cost in INR and basis points.
        """
        buy_leg = self.compute_leg(entry_value, is_buy=True, segment=segment, num_scrips=num_scrips)
        sell_leg = self.compute_leg(exit_value, is_buy=False, segment=segment, num_scrips=num_scrips)

        total_cost = buy_leg["total"] + sell_leg["total"]
        avg_value = (entry_value + exit_value) / 2.0
        total_bps = (total_cost / avg_value) * 10000.0 if avg_value > 0 else 0.0

        return {
            "buy_leg": buy_leg,
            "sell_leg": sell_leg,
            "total_cost": total_cost,
            "total_bps": total_bps,
            "breakdown": {
                "brokerage": buy_leg["brokerage"] + sell_leg["brokerage"],
                "stt": buy_leg["stt"] + sell_leg["stt"],
                "exchange_txn": buy_leg["exchange_txn"] + sell_leg["exchange_txn"],
                "sebi_fee": buy_leg["sebi_fee"] + sell_leg["sebi_fee"],
                "gst": buy_leg["gst"] + sell_leg["gst"],
                "stamp_duty": buy_leg["stamp_duty"] + sell_leg["stamp_duty"],
                "dp_charge": buy_leg["dp_charge"] + sell_leg["dp_charge"],
            }
        }


@dataclass
class ImpactModelConfig:
    """Configuration for square-root market impact model."""
    eta: float = 0.10          # Impact scaling coefficient (typically 0.05 to 0.20)
    alpha: float = 0.50        # Square-root elasticity exponent
    min_impact_bps: float = 1.0   # Minimum impact floor in bps (0.01%)
    max_impact_bps: float = 300.0 # Maximum impact cap in bps (3.00%)


class MarketImpactModel:
    """
    Square-Root Market Impact Model:
    Cost_impact = eta * (Q / ADTV_20)^alpha * sigma_daily

    Timing & Mechanics:
    - Q: Order size in INR (or shares).
    - ADTV_20: 20-day median daily traded turnover in INR (or shares).
    - sigma_daily: 20-day daily volatility (standard deviation of daily returns).
    - eta: Calibration constant.
    - alpha: Power exponent (default 0.5 for square-root law).
    """

    def __init__(self, config: Optional[ImpactModelConfig] = None):
        self.config = config or ImpactModelConfig()

    def estimate_impact(
        self,
        order_value: float,
        adtv_20: float,
        sigma_daily: float,
    ) -> float:
        """
        Estimate price impact fraction for an execution order.

        Parameters
        ----------
        order_value : float
            Value of order in INR.
        adtv_20 : float
            20-day median daily traded turnover in INR.
        sigma_daily : float
            Daily volatility of the stock (e.g. 0.02 for 2% daily vol).

        Returns
        -------
        float
            Impact cost as a decimal fraction of trade value (e.g. 0.0015 = 15 bps).
        """
        if adtv_20 <= 0 or order_value <= 0:
            return self.config.max_impact_bps / 10000.0

        # Participation rate
        participation = order_value / adtv_20

        # Raw square-root impact
        impact_raw = self.config.eta * (participation ** self.config.alpha) * sigma_daily

        # Clip to configured bounds
        min_frac = self.config.min_impact_bps / 10000.0
        max_frac = self.config.max_impact_bps / 10000.0
        return float(np.clip(impact_raw, min_frac, max_frac))

    def calibrate_from_executions(
        self,
        execution_df: pd.DataFrame,
    ) -> Tuple[float, float, float]:
        """
        Calibration Plan:
        Estimate (eta, alpha) via Ordinary Least Squares (OLS) regression in log-space:
        ln(Slippage / sigma_daily) = ln(eta) + alpha * ln(order_value / adtv_20) + epsilon

        Parameters
        ----------
        execution_df : pd.DataFrame
            Must contain columns: ['realized_slippage', 'order_value', 'adtv_20', 'sigma_daily']
            where realized_slippage is |P_exec - P_arrival| / P_arrival.

        Returns
        -------
        Tuple[float, float, float]
            (calibrated_eta, calibrated_alpha, r_squared)
        """
        required = ['realized_slippage', 'order_value', 'adtv_20', 'sigma_daily']
        for col in required:
            if col not in execution_df.columns:
                raise ValueError(f"Missing required execution column: {col}")

        # Filter clean positive records
        mask = (
            (execution_df['realized_slippage'] > 1e-6) &
            (execution_df['order_value'] > 0) &
            (execution_df['adtv_20'] > 0) &
            (execution_df['sigma_daily'] > 1e-4)
        )
        clean = execution_df[mask].copy()

        if len(clean) < 10:
            # Insufficient samples for robust calibration, return defaults
            return self.config.eta, self.config.alpha, 0.0

        clean['y'] = np.log(clean['realized_slippage'] / clean['sigma_daily'])
        clean['x'] = np.log(clean['order_value'] / clean['adtv_20'])

        # Drop inf / NaN
        valid = clean[np.isfinite(clean['y']) & np.isfinite(clean['x'])]
        if len(valid) < 10:
            return self.config.eta, self.config.alpha, 0.0

        x = valid['x'].values
        y = valid['y'].values

        # OLS: y = beta0 + beta1 * x
        A = np.vstack([np.ones_like(x), x]).T
        beta, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

        ln_eta = beta[0]
        alpha = float(beta[1])
        eta = float(np.exp(ln_eta))

        # Clamp calibrated values to economically sensible ranges
        eta = float(np.clip(eta, 0.01, 1.0))
        alpha = float(np.clip(alpha, 0.2, 0.8))

        # R-squared
        y_pred = A @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        # Update instance parameters
        self.config.eta = eta
        self.config.alpha = alpha

        return eta, alpha, r2


class ExecutionFrictions:
    """
    Combined Execution Frictions Engine:
    Integrates Statutory NSE Cost Model and Market Impact Model.
    """

    def __init__(
        self,
        cost_config: Optional[NSECostConfig] = None,
        impact_config: Optional[ImpactModelConfig] = None,
    ):
        self.cost_model = NSECostModel(cost_config)
        self.impact_model = MarketImpactModel(impact_config)

    def total_execution_cost(
        self,
        order_value: float,
        is_buy: bool,
        adtv_20: float,
        sigma_daily: float,
        segment: TradingSegment = TradingSegment.CASH_DELIVERY,
        num_scrips: int = 1,
    ) -> Dict[str, float]:
        """
        Compute total friction (statutory fees + market impact) for an order.

        Parameters
        ----------
        order_value : float
            Order turnover in INR.
        is_buy : bool
            Trade direction.
        adtv_20 : float
            20-day median daily traded turnover in INR.
        sigma_daily : float
            20-day daily price volatility.
        segment : TradingSegment
            Trading segment.
        num_scrips : int
            Number of distinct scrips.

        Returns
        -------
        Dict[str, float]
            Total friction in INR, statutory component, impact component, and basis points.
        """
        statutory = self.cost_model.compute_leg(
            trade_value=order_value,
            is_buy=is_buy,
            segment=segment,
            num_scrips=num_scrips,
        )
        impact_frac = self.impact_model.estimate_impact(
            order_value=order_value,
            adtv_20=adtv_20,
            sigma_daily=sigma_daily,
        )
        impact_inr = order_value * impact_frac
        total_inr = statutory["total"] + impact_inr
        total_bps = (total_inr / order_value) * 10000.0 if order_value > 0 else 0.0

        return {
            "total_friction_inr": total_inr,
            "statutory_inr": statutory["total"],
            "impact_inr": impact_inr,
            "statutory_bps": statutory["bps"],
            "impact_bps": impact_frac * 10000.0,
            "total_bps": total_bps,
            "breakdown": statutory,
        }
