"""
Tests for Investor-centric features: FIFO taxation, Corporate Actions, and DP charges.
"""
import pytest
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PortfolioManager import PortfolioManager
from nse_cost_model import NSECostModel, Segment

class TestInvestorFeatures:

    @pytest.fixture
    def pm(self, tmp_path):
        return PortfolioManager(data_dir=str(tmp_path))

    def test_fifo_taxation_logic(self, pm):
        """Verify that selling shares consumes oldest lots first and calculates STCG/LTCG."""
        # Buy 100 shares 2 years ago
        date_ltcg = (datetime.now() - timedelta(days=400)).isoformat()
        pm.add_transaction("test_user", "RELIANCE", "BUY", 100, 2000.0, date=date_ltcg)
        
        # Buy 50 shares 10 days ago
        date_stcg = (datetime.now() - timedelta(days=10)).isoformat()
        pm.add_transaction("test_user", "RELIANCE", "BUY", 50, 2500.0, date=date_stcg)
        
        # Sell 120 shares today
        pm.add_transaction("test_user", "RELIANCE", "SELL", 120, 3000.0)
        
        summary = pm.get_portfolio_summary("test_user")
        tax_events = summary['tax_events']
        
        assert len(tax_events) == 2, "Should create two tax events from two consumed lots"
        
        # First lot (100 shares) should be LTCG
        ltcg_event = tax_events[0]
        assert ltcg_event['quantity'] == 100
        assert ltcg_event['buy_price'] == 2000.0
        assert ltcg_event['gain'] == (3000.0 - 2000.0) * 100
        assert ltcg_event['tax_type'] == 'LTCG'
        
        # Second lot (20 shares) should be STCG
        stcg_event = tax_events[1]
        assert stcg_event['quantity'] == 20
        assert stcg_event['buy_price'] == 2500.0
        assert stcg_event['gain'] == (3000.0 - 2500.0) * 20
        assert stcg_event['tax_type'] == 'STCG'
        
        # Remaining holdings should be 30 shares
        holding = summary['holdings'][0]
        assert holding['quantity'] == 30
        assert holding['avg_price'] == 2500.0  # From the second lot

    def test_corporate_action_dividend(self, pm):
        """Verify dividends are added to cash balance."""
        pm.add_transaction("test", "TCS", "BUY", 100, 3500.0)
        pm.process_corporate_action("test", "TCS", "DIVIDEND", dps=50.0)
        
        summary = pm.get_portfolio_summary("test")
        assert summary['summary']['cash_balance'] == 5000.0  # 100 * 50
        
        # Ensure cost basis is unchanged
        holding = summary['holdings'][0]
        assert holding['avg_price'] == 3500.0

    def test_corporate_action_split(self, pm):
        """Verify stock splits adjust quantities and average prices."""
        pm.add_transaction("test", "INFY", "BUY", 100, 1500.0)
        # 2-for-1 split
        pm.process_corporate_action("test", "INFY", "SPLIT", ratio=2.0)
        
        summary = pm.get_portfolio_summary("test")
        holding = summary['holdings'][0]
        
        assert holding['quantity'] == 200.0
        assert holding['avg_price'] == 750.0  # 1500 / 2
        assert holding['total_cost'] == 150000.0  # Unchanged total cost

    def test_dp_charge_on_delivery_sell(self):
        """Verify DP charge is only applied on delivery sell."""
        model = NSECostModel(dp_charge=15.34)
        
        # Delivery Buy - NO DP charge
        buy_res = model.compute("BUY", 1000.0, 100, Segment.CASH_DELIVERY)
        assert buy_res['breakdown']['dp_charge'] == 0.0
        
        # Intraday Sell - NO DP charge
        intra_res = model.compute("SELL", 1000.0, 100, Segment.CASH_INTRADAY)
        assert intra_res['breakdown']['dp_charge'] == 0.0
        
        # Delivery Sell - YES DP charge
        sell_res = model.compute("SELL", 1000.0, 100, Segment.CASH_DELIVERY)
        assert sell_res['breakdown']['dp_charge'] == 15.34
