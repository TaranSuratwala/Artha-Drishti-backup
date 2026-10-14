"""
Tests that detect look-ahead bias in backtesting engine.
The key principle: a signal generated on day T must not use any data from day T+1 or later.
"""
import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BacktestEngine import BacktestEngine
from backtest_models import BacktestConfig

class TestLookAheadBias:

    @pytest.fixture
    def engine(self):
        config = BacktestConfig(
            initial_capital=100000,
            use_next_bar_open=True,
            transaction_cost_pct=0.0,
            base_slippage_pct=0.0,
        )
        return BacktestEngine(config=config)

    @pytest.fixture
    def synthetic_data(self):
        """100-day price series with known properties."""
        np.random.seed(42)
        dates = pd.bdate_range("2024-01-01", periods=100)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        prices = np.maximum(prices, 10)  # floor
        return pd.DataFrame({
            "date": dates,
            "open": prices + np.random.randn(100) * 0.1,
            "high": prices + abs(np.random.randn(100)) * 1.0,
            "low": prices - abs(np.random.randn(100)) * 1.0,
            "close": prices,
            "volume": np.random.randint(10000, 100000, 100),
        })

    def test_future_signal_cannot_improve_returns(self, engine, synthetic_data):
        data = synthetic_data.copy()
        returns = data["close"].pct_change().shift(-1)  # FUTURE returns = look-ahead

        # Perfect foresight signals
        signals = pd.DataFrame({
            "date": data["date"],
            "signal": np.where(returns > 0, 1, -1)
        })

        result = engine.backtest_strategy(signals, data, "PerfectForesight")

        # Even with perfect foresight, with next-bar-open fills, win rate should not be 100%
        # because the return is based on close, but the fill is at open.
        # It's an anti-leakage check.
        assert result.win_rate_pct < 100, \
            "Perfect foresight should not yield 100% wins with next-bar-open fills"

    def test_random_signals_produce_near_zero_returns(self, engine, synthetic_data):
        """Random signals should produce near-zero return after enough trades."""
        np.random.seed(123)
        signals = pd.DataFrame({
            "date": synthetic_data["date"],
            "signal": np.random.choice([1, -1, 0], size=100, p=[0.3, 0.3, 0.4])
        })
        result = engine.backtest_strategy(signals, synthetic_data, "Random")

        # With zero costs/slippage, random signals should give near 0% return
        assert -20 < result.total_return_pct < 20, \
            f"Random strategy returned {result.total_return_pct}% — possible bias"
