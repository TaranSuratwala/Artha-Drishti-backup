"""
Accurate NSE transaction cost model.
Reference: SEBI circular + broker charge pages.
Updated to align with statutory rules:
- STT Cash Delivery: 0.1% on Buy AND 0.1% on Sell (total 0.2% round trip).
- Zero brokerage on cash delivery (Zerodha discount broker standard).
- GST 18% on (Brokerage + Exchange Txn Charges + SEBI Fee).
- DP Charge ₹15.93 (₹13.50 + 18% GST) on Cash Delivery Sell only.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional

from frictions import TradingSegment, NSECostConfig, NSECostModel as PreciseCostModel

class Segment(Enum):
    CASH_DELIVERY = "cash_delivery"
    CASH_INTRADAY = "cash_intraday"
    FNO_FUTURES = "fno_futures"
    FNO_OPTIONS = "fno_options"

@dataclass
class NSECostModel:
    """
    Backwards-compatible wrapper delegating to institutional NSECostModel in frictions.py.
    """
    flat_brokerage: float = 20.0        # ₹20 per executed order (Zerodha-style for intraday/F&O)
    brokerage_delivery: float = 0.0     # ₹0 for cash delivery

    # STT rates (verify against latest SEBI circular / contract note)
    stt_delivery_buy: float = 0.001     # 0.1% on buy value
    stt_delivery_sell: float = 0.001    # 0.1% on sell value
    stt_intraday_sell: float = 0.00025  # 0.025% on sell value
    stt_futures_sell: float = 0.00020   # 0.02% on sell value (Finance Act 2024)
    stt_options_sell: float = 0.0010    # 0.10% on sell premium (Finance Act 2024)

    # Exchange transaction charges
    nse_txn_cash: float = 0.0000297     # 0.00297% (NSE True-to-label Oct 2024)
    nse_txn_fno: float = 0.0000173      # 0.00173% for futures

    # SEBI turnover fee
    sebi_fee: float = 0.000001          # ₹10 per crore

    # Stamp duty (state-level, buy side only)
    stamp_duty_buy: float = 0.00015     # 0.015% for cash delivery
    stamp_duty_fno: float = 0.00002     # 0.002% for futures

    # GST
    gst_rate: float = 0.18
    
    # DP (Depository Participant) charge (flat per delivery sell per day)
    dp_charge: float = 15.93            # ₹13.50 + 18% GST = ₹15.93 per scrip

    def __post_init__(self):
        dp_base = self.dp_charge / (1.0 + self.gst_rate)
        config = NSECostConfig(
            stt_delivery_buy=self.stt_delivery_buy,
            stt_delivery_sell=self.stt_delivery_sell,
            stt_intraday_sell=self.stt_intraday_sell,
            stt_futures_sell=self.stt_futures_sell,
            stt_options_sell=self.stt_options_sell,
            nse_txn_cash=self.nse_txn_cash,
            nse_txn_futures=self.nse_txn_fno,
            sebi_fee_rate=self.sebi_fee,
            stamp_duty_delivery=self.stamp_duty_buy,
            stamp_duty_futures=self.stamp_duty_fno,
            gst_rate=self.gst_rate,
            dp_charge_base=dp_base,
            brokerage_delivery=self.brokerage_delivery,
            brokerage_flat_order=self.flat_brokerage,
        )
        self._precise = PreciseCostModel(config)

    def compute(self, side: str, price: float, qty: int,
                segment: Segment = Segment.CASH_DELIVERY) -> dict:
        value = price * qty
        is_buy = (side.upper() == "BUY")
        mapped_seg = TradingSegment(segment.value)

        leg = self._precise.compute_leg(
            trade_value=value,
            is_buy=is_buy,
            segment=mapped_seg,
            num_scrips=1
        )

        return {
            "total": round(leg["total"], 2),
            "breakdown": {
                "brokerage": round(leg["brokerage"], 2),
                "dp_charge": round(leg["dp_charge"], 2),
                "stt": round(leg["stt"], 2),
                "exchange": round(leg["exchange_txn"], 2),
                "sebi": round(leg["sebi_fee"], 2),
                "gst": round(leg["gst"], 2),
                "stamp_duty": round(leg["stamp_duty"], 2),
            }
        }
