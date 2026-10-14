"""
Tests for portfolio P&L accounting correctness.
"""
import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PortfolioManager import PortfolioManager

class TestPortfolioAccounting:

    @pytest.fixture
    def pm(self, tmp_path):
        return PortfolioManager(data_dir=str(tmp_path))

    def test_buy_sell_full_position_zero_holdings(self, pm):
        pm.add_transaction("test", "RELIANCE", "BUY", 100, 2500.0)
        pm.add_transaction("test", "RELIANCE", "SELL", 100, 2600.0)
        summary = pm.get_portfolio_summary("test")
        assert summary["summary"]["portfolio_count"] == 0

    def test_average_price_after_multiple_buys(self, pm):
        pm.add_transaction("test", "INFY", "BUY", 50, 1400.0)
        pm.add_transaction("test", "INFY", "BUY", 50, 1600.0)
        summary = pm.get_portfolio_summary("test")
        holding = summary["holdings"][0]
        expected_avg = (50 * 1400 + 50 * 1600) / 100  # ₹1500
        assert abs(holding["avg_price"] - expected_avg) < 0.01

    def test_partial_sell_preserves_cost_basis(self, pm):
        pm.add_transaction("test", "TCS", "BUY", 100, 3500.0)
        pm.add_transaction("test", "TCS", "SELL", 40, 3600.0)
        summary = pm.get_portfolio_summary("test")
        holding = summary["holdings"][0]
        assert holding["quantity"] == 60
        assert abs(holding["avg_price"] - 3500.0) < 0.01

    def test_sell_more_than_owned_raises(self, pm):
        pm.add_transaction("test", "HDFC", "BUY", 10, 1500.0)
        with pytest.raises(ValueError, match="Insufficient shares"):
            pm.add_transaction("test", "HDFC", "SELL", 20, 1600.0)

    def test_negative_quantity_raises(self, pm):
        with pytest.raises(ValueError, match="positive"):
            pm.add_transaction("test", "SBIN", "BUY", -10, 500.0)

    def test_pnl_calculation_with_current_prices(self, pm):
        pm.add_transaction("test", "ITC", "BUY", 200, 450.0)
        current_prices = {"ITC": 480.0}
        summary = pm.get_portfolio_summary("test", current_prices)
        holding = summary["holdings"][0]
        expected_pnl = (480 - 450) * 200
        assert abs(holding["unrealized_pnl"] - expected_pnl) < 0.01
