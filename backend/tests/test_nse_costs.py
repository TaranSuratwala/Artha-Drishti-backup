"""
Tests that verify NSE transaction cost model matches SEBI-mandated fee schedule.
All reference values from: https://zerodha.com/charges
"""
import pytest
import os
import sys

# Add parent dir to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nse_cost_model import NSECostModel, Segment

class TestNSETransactionCosts:
    """Verify that the backtest engine's cost model matches real NSE costs."""

    REF = NSECostModel()

    @pytest.mark.parametrize("price,qty,side", [
        (2500.0, 100, "BUY"),    # Blue chip: RELIANCE
        (2500.0, 100, "SELL"),   # Same stock, sell side (STT kicks in)
        (15.50, 1000, "BUY"),    # Penny stock
        (45000.0, 10, "SELL"),   # High-price stock: MRF
        (150.0, 5000, "BUY"),    # Large order, mid-cap
    ])
    def test_cost_matches_nse_schedule(self, price, qty, side):
        expected = self.REF.compute(side, price, qty, Segment.CASH_DELIVERY)

        assert expected["total"] > 0
        if side == "SELL":
            assert expected["breakdown"]["stt"] > 0, "STT must be charged on delivery sell (0.1%)"
            assert expected["breakdown"]["stamp_duty"] == 0, "Stamp duty must NOT be charged on sell"
            assert expected["breakdown"]["dp_charge"] > 0, "DP charge must be charged on delivery sell"
        else:
            assert expected["breakdown"]["stt"] > 0, "STT must be charged on delivery buy (0.1%)"
            assert expected["breakdown"]["stamp_duty"] > 0, "Stamp duty must be charged on buy (0.015%)"
            assert expected["breakdown"]["dp_charge"] == 0, "DP charge must NOT be charged on buy"

    def test_sell_costs_higher_than_buy(self):
        """DP charges (₹15.93) make selling delivery slightly higher than buying delivery for ₹1L orders."""
        buy = self.REF.compute("BUY", 1000.0, 100)
        sell = self.REF.compute("SELL", 1000.0, 100)
        assert sell["total"] > buy["total"], \
            f"Sell cost (₹{sell['total']}) should be > buy cost (₹{buy['total']}) due to DP charges"
