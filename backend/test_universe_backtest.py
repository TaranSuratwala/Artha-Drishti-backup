"""Unit tests for universe backtesting (Feature #3).

Run with:  python -m pytest backend/test_universe_backtest.py -v
"""

import pytest

from index_universe import INDEX_SOURCES, IndexUniverseProvider, normalize_index_name
from universe_backtest import (
    UniverseBacktestRunner,
    aggregate_universe_results,
)


class TestNormalizeIndexName:
    @pytest.mark.parametrize("raw,expected", [
        ("nifty 50", "NIFTY50"),
        ("NIFTY-BANK", "NIFTYBANK"),
        ("Nifty500", "NIFTY500"),
        ("nifty midcap 150", "NIFTYMIDCAP150"),
    ])
    def test_normalization(self, raw, expected):
        assert normalize_index_name(raw) == expected

    def test_all_sources_are_normalized_keys(self):
        for name in INDEX_SOURCES:
            assert normalize_index_name(name) == name


class TestIndexUniverseProvider:
    def test_unsupported_index_raises(self, tmp_path):
        provider = IndexUniverseProvider(cache_dir=str(tmp_path))
        with pytest.raises(ValueError):
            provider.get_constituents("NIFTY9999")

    def test_parse_csv_extracts_symbols(self):
        text = "Company Name,Industry,Symbol,Series,ISIN Code\n" \
               "Reliance Industries Ltd.,Oil & Gas,RELIANCE,EQ,INE002A01018\n" \
               "Tata Consultancy Services,IT,TCS,EQ,INE467B01029\n"
        assert IndexUniverseProvider._parse_csv(text) == ["RELIANCE", "TCS"]

    def test_parse_csv_without_symbol_column(self):
        assert IndexUniverseProvider._parse_csv("A,B\n1,2\n") == []


def _summary(ret, trades=10, sharpe=1.0, win=55.0):
    return {
        "total_return_pct": ret,
        "cagr_pct": ret / 2,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sharpe * 1.2,
        "max_drawdown_pct": 12.0,
        "total_trades": trades,
        "win_rate_pct": win,
        "profit_factor": 1.5,
    }


class TestAggregation:
    def test_aggregates_means_and_extremes(self):
        per_ticker = {
            "RELIANCE": _summary(20.0),
            "TCS": _summary(-5.0),
            "INFY": _summary(10.0),
        }
        agg = aggregate_universe_results(per_ticker)
        assert agg["n_tickers"] == 3
        assert agg["total_trades"] == 30
        assert agg["profitable_tickers"] == 2
        assert agg["mean_total_return_pct"] == pytest.approx(25.0 / 3, abs=0.01)
        assert agg["best_ticker"]["ticker"] == "RELIANCE"
        assert agg["worst_ticker"]["ticker"] == "TCS"

    def test_handles_non_finite_values(self):
        per_ticker = {"A": _summary(float("nan")), "B": _summary(10.0)}
        agg = aggregate_universe_results(per_ticker)
        assert agg["mean_total_return_pct"] == pytest.approx(10.0)
        assert agg["profitable_pct"] == 100.0

    def test_empty_input(self):
        assert aggregate_universe_results({})["n_tickers"] == 0


class TestUniverseBacktestRunner:
    def test_runs_all_tickers_and_ranks(self):
        def run_single(ticker):
            return _summary({"A": 5.0, "B": 15.0, "C": -2.0}[ticker])

        runner = UniverseBacktestRunner(max_workers=2)
        out = runner.run(["A", "B", "C"], run_single)
        assert out["n_completed"] == 3
        assert out["n_failed"] == 0
        assert [r["ticker"] for r in out["per_ticker"]] == ["B", "A", "C"]

    def test_failures_are_isolated(self):
        def run_single(ticker):
            if ticker == "BAD":
                raise ValueError("no data")
            return _summary(8.0)

        runner = UniverseBacktestRunner(max_workers=2)
        out = runner.run(["GOOD", "BAD"], run_single)
        assert out["n_completed"] == 1
        assert out["n_failed"] == 1
        assert "BAD" in out["errors"]
        assert out["aggregate"]["n_tickers"] == 1

    def test_progress_callback_reaches_total(self):
        seen = []
        runner = UniverseBacktestRunner(max_workers=1)
        runner.run(["A", "B"], lambda t: _summary(1.0),
                   progress_cb=lambda d, t, c: seen.append((d, t)))
        assert (2, 2) in seen
