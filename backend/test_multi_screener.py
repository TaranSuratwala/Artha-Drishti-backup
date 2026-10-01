"""Unit tests for MultiStrategyScreener (Feature #1).

Run with:  python -m pytest backend/test_multi_screener.py -v
"""

import time

import pytest

from MultiStrategyScreener import (
    ScreenerJobManager,
    combine_screen_results,
    run_multi_screen,
)


def _entry(ticker, score, price=100.0, indicators=None):
    return {
        "ticker": ticker,
        "score": score,
        "current_price": price,
        "conditions_passed": 2,
        "total_conditions": 3,
        "confidence": 66.7,
        "key_indicators": indicators or {"rsi_14": 55.0},
        "patterns": [],
    }


class TestCombineScreenResults:
    def test_and_logic_keeps_only_intersection(self):
        results = {
            "momentum": [_entry("RELIANCE", 80), _entry("TCS", 70)],
            "breakout": [_entry("RELIANCE", 60), _entry("INFY", 90)],
        }
        combined = combine_screen_results(results, logic="AND")
        tickers = [r["ticker"] for r in combined]
        assert tickers == ["RELIANCE"]
        assert combined[0]["matched_strategies"] == ["breakout", "momentum"]
        assert combined[0]["combined_score"] == pytest.approx(70.0)

    def test_or_logic_keeps_union(self):
        results = {
            "momentum": [_entry("RELIANCE", 80), _entry("TCS", 70)],
            "breakout": [_entry("RELIANCE", 60), _entry("INFY", 90)],
        }
        combined = combine_screen_results(results, logic="OR")
        tickers = {r["ticker"] for r in combined}
        assert tickers == {"RELIANCE", "TCS", "INFY"}
        # Multi-strategy matches rank above single-strategy matches
        assert combined[0]["ticker"] == "RELIANCE"
        assert combined[0]["matched_count"] == 2

    def test_or_logic_averages_over_all_selected_strategies(self):
        results = {
            "a": [_entry("TCS", 80)],
            "b": [],
        }
        combined = combine_screen_results(results, logic="OR")
        assert combined[0]["combined_score"] == pytest.approx(40.0)

    def test_max_results_respected(self):
        results = {"a": [_entry(f"T{i}", 50 + i) for i in range(20)]}
        combined = combine_screen_results(results, logic="OR", max_results=5)
        assert len(combined) == 5

    def test_invalid_logic_raises(self):
        with pytest.raises(ValueError):
            combine_screen_results({"a": []}, logic="XOR")

    def test_empty_input(self):
        assert combine_screen_results({}, logic="AND") == []

    def test_key_indicators_merged(self):
        results = {
            "a": [_entry("TCS", 80, indicators={"rsi_14": 60})],
            "b": [_entry("TCS", 70, indicators={"macd_hist": 1.5})],
        }
        combined = combine_screen_results(results, logic="AND")
        ki = combined[0]["key_indicators"]
        assert ki["rsi_14"] == 60
        assert ki["macd_hist"] == 1.5


class FakeScreener:
    """Stub matching InteractiveStockScreener.run_screening's contract."""

    def __init__(self, data, fail_on=None):
        self.data = data
        self.fail_on = fail_on or set()
        self.calls = []

    def run_screening(self, strategy_name, max_results=50, overrides=None):
        self.calls.append((strategy_name, max_results, overrides))
        if strategy_name in self.fail_on:
            return {"status": "error", "message": "boom"}
        return {
            "status": "success",
            "strategy_name": strategy_name,
            "results": self.data.get(strategy_name, []),
        }


class TestRunMultiScreen:
    def test_combines_and_reports_counts(self):
        screener = FakeScreener({
            "momentum": [_entry("RELIANCE", 80)],
            "breakout": [_entry("RELIANCE", 60), _entry("INFY", 90)],
        })
        out = run_multi_screen(screener, ["momentum", "breakout"], logic="AND")
        assert out["status"] == "success"
        assert out["count"] == 1
        assert out["per_strategy_counts"] == {"momentum": 1, "breakout": 2}
        assert out["errors"] is None

    def test_strategy_error_is_isolated(self):
        screener = FakeScreener({"momentum": [_entry("TCS", 70)]}, fail_on={"broken"})
        out = run_multi_screen(screener, ["momentum", "broken"], logic="OR")
        assert out["status"] == "success"
        assert "broken" in out["errors"]
        assert {r["ticker"] for r in out["results"]} == {"TCS"}

    def test_progress_callback_invoked(self):
        screener = FakeScreener({"momentum": []})
        seen = []
        run_multi_screen(
            screener, ["momentum"], logic="OR",
            progress_cb=lambda d, t, c: seen.append((d, t, c)),
        )
        assert (1, 1, "momentum") in seen

    def test_per_strategy_pool_expands_candidates(self):
        screener = FakeScreener({"momentum": []})
        run_multi_screen(screener, ["momentum"], max_results=50)
        _, pool, _ = screener.calls[0]
        assert pool >= 500


class TestScreenerJobManager:
    def test_job_lifecycle_success(self):
        mgr = ScreenerJobManager(max_workers=1)
        job_id = mgr.submit(lambda progress: {"ok": True})
        deadline = time.time() + 5
        while time.time() < deadline:
            job = mgr.get(job_id)
            if job["state"] == "completed":
                break
            time.sleep(0.05)
        job = mgr.get(job_id)
        assert job["state"] == "completed"
        assert job["result"] == {"ok": True}
        assert job["error"] is None

    def test_job_lifecycle_failure(self):
        mgr = ScreenerJobManager(max_workers=1)

        def boom(progress):
            raise RuntimeError("explode")

        job_id = mgr.submit(boom)
        deadline = time.time() + 5
        while time.time() < deadline:
            job = mgr.get(job_id)
            if job["state"] == "failed":
                break
            time.sleep(0.05)
        job = mgr.get(job_id)
        assert job["state"] == "failed"
        assert "explode" in job["error"]

    def test_unknown_job_returns_none(self):
        mgr = ScreenerJobManager(max_workers=1)
        assert mgr.get("does-not-exist") is None
