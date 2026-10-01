"""Unit tests for the walk-forward harness and drift monitor (Feature #2).

Run with:  python -m pytest backend/test_walkforward.py -v
"""

import random
from datetime import datetime, timedelta

import pytest

from drift_monitor import FeatureDriftMonitor, classify_psi, population_stability_index
from walkforward import (
    PredictionRecord,
    PurgedWalkForwardSplitter,
    WalkForwardRunner,
    aggregate_fold_reports,
    direction_metrics,
    regression_metrics,
)


class TestPurgedWalkForwardSplitter:
    def test_embargo_gap_enforced(self):
        splitter = PurgedWalkForwardSplitter(train_days=100, test_days=20, embargo_days=10)
        windows = splitter.split(datetime(2023, 1, 1), datetime(2024, 6, 1))
        assert windows, "expected at least one fold"
        for w in windows:
            gap = (w.test_start - w.train_end).days
            assert gap == 10, "embargo must separate train end from test start"

    def test_no_test_window_overlaps_training_of_same_fold(self):
        splitter = PurgedWalkForwardSplitter(train_days=100, test_days=20, embargo_days=5)
        for w in splitter.split(datetime(2023, 1, 1), datetime(2024, 6, 1)):
            assert w.test_start > w.train_end

    def test_folds_advance_by_step(self):
        splitter = PurgedWalkForwardSplitter(train_days=100, test_days=20, embargo_days=5, step_days=20)
        windows = splitter.split(datetime(2023, 1, 1), datetime(2024, 6, 1))
        assert len(windows) >= 2
        assert (windows[1].train_start - windows[0].train_start).days == 20

    def test_short_range_yields_no_folds(self):
        splitter = PurgedWalkForwardSplitter(train_days=500, test_days=60, embargo_days=10)
        assert splitter.split(datetime(2024, 1, 1), datetime(2024, 3, 1)) == []

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            PurgedWalkForwardSplitter(train_days=0)
        with pytest.raises(ValueError):
            PurgedWalkForwardSplitter(embargo_days=-1)


def _rec(prob, pred_ret, act_ret, ticker="TCS"):
    return PredictionRecord(ticker=ticker, prob_bullish=prob,
                            predicted_return=pred_ret, actual_return=act_ret)


class TestMetrics:
    def test_direction_metrics_asymmetric_thresholds(self):
        records = [
            _rec(0.80, 0.03, 0.02),   # buy, hit
            _rec(0.70, 0.02, -0.01),  # buy, miss
            _rec(0.20, -0.02, -0.03), # sell, hit
            _rec(0.50, 0.00, 0.01),   # not actionable
        ]
        m = direction_metrics(records)
        assert m["n_actionable"] == 3
        assert m["buy_signals"] == 2 and m["buy_precision"] == 0.5
        assert m["sell_signals"] == 1 and m["sell_precision"] == 1.0
        assert m["hit_rate"] == pytest.approx(2 / 3, abs=1e-3)

    def test_regression_metrics(self):
        records = [_rec(0.6, 0.02, 0.01), _rec(0.6, -0.01, 0.01)]
        m = regression_metrics(records)
        assert m["n"] == 2
        assert m["mae"] == pytest.approx(0.015, abs=1e-6)
        assert m["sign_agreement"] == 0.5

    def test_regression_metrics_filters_non_finite(self):
        records = [_rec(0.6, float("nan"), 0.01), _rec(0.6, 0.02, 0.01)]
        assert regression_metrics(records)["n"] == 1

    def test_aggregate_reports_worst_fold(self):
        folds = [
            {"direction": {"hit_rate": 0.6}, "regression": {"mae": 0.02}},
            {"direction": {"hit_rate": 0.4}, "regression": {"mae": 0.05}},
        ]
        agg = aggregate_fold_reports(folds)
        assert agg["mean_hit_rate"] == 0.5
        assert agg["worst_fold_hit_rate"] == 0.4
        assert agg["worst_fold_mae"] == 0.05


class TestWalkForwardRunner:
    def test_runner_trains_only_on_past_data(self, tmp_path):
        """The runner must never pass test dates into train_fn."""
        seen = []

        def train_fn(train_start, train_end):
            seen.append((train_start, train_end))
            return "model"

        def predict_fn(model, test_start, test_end):
            # every training window seen so far must end before this test starts
            assert all(te < test_start for _, te in seen[-1:])
            return [_rec(0.8, 0.02, 0.01)]

        runner = WalkForwardRunner(
            train_fn, predict_fn,
            splitter=PurgedWalkForwardSplitter(train_days=100, test_days=20, embargo_days=10),
            report_path=tmp_path / "report.json",
        )
        report = runner.run(datetime(2023, 1, 1), datetime(2024, 6, 1))
        assert report["summary"]["n_folds"] == len(seen)
        assert (tmp_path / "report.json").exists()

    def test_runner_raises_on_short_range(self, tmp_path):
        runner = WalkForwardRunner(
            lambda a, b: None, lambda m, a, b: [],
            splitter=PurgedWalkForwardSplitter(train_days=500, test_days=60, embargo_days=10),
            report_path=tmp_path / "r.json",
        )
        with pytest.raises(ValueError):
            runner.run(datetime(2024, 1, 1), datetime(2024, 2, 1))


class TestDriftMonitor:
    def test_identical_distributions_are_stable(self):
        rng = random.Random(42)
        sample = [rng.gauss(0, 1) for _ in range(1000)]
        psi = population_stability_index(sample, list(sample))
        assert psi < 0.01
        assert classify_psi(psi) == "stable"

    def test_shifted_distribution_flags_severe(self):
        rng = random.Random(42)
        baseline = [rng.gauss(0, 1) for _ in range(1000)]
        shifted = [rng.gauss(2.0, 1) for _ in range(1000)]
        psi = population_stability_index(baseline, shifted)
        assert psi >= 0.25
        assert classify_psi(psi) == "severe"

    def test_monitor_report_structure(self):
        rng = random.Random(7)
        baseline = {"rsi_14": [rng.gauss(50, 10) for _ in range(500)],
                    "volume_ratio": [rng.gauss(1, 0.2) for _ in range(500)]}
        live = {"rsi_14": [rng.gauss(50, 10) for _ in range(500)],
                "volume_ratio": [rng.gauss(3, 0.2) for _ in range(500)]}
        report = FeatureDriftMonitor().check(baseline, live)
        assert report["overall_status"] == "severe"
        assert "volume_ratio" in report["severe_features"]
        assert report["trigger_recalibration"] is True

    def test_insufficient_data_handled(self):
        report = FeatureDriftMonitor().check({"x": [1.0]}, {"x": [1.0]})
        assert report["per_feature"]["x"]["status"] == "insufficient_data"
