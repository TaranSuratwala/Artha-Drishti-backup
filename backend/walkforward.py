"""
Purged Walk-Forward Validation Harness (Feature #2)
===================================================
Model-agnostic rolling-origin evaluation for the prediction stack.

Why purged walk-forward?
- Random K-fold on time series leaks future information into training.
- Targets computed over an N-day horizon overlap fold boundaries, so an
  *embargo* gap between train end and test start is mandatory.
- Rolling windows measure how the model would actually have performed
  if retrained periodically, which matches the production retraining
  pipeline in MLPredictor.py.

The runner is decoupled from any specific model: you provide
`train_fn(train_start, train_end) -> model` and
`predict_fn(model, test_start, test_end) -> List[PredictionRecord]`.
This lets the same harness compare the unified PyTorch model against
baselines (e.g., gradient boosting) on identical splits.
"""

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "unified_metrics" / "walkforward_report.json"


# ──────────────────────────────────────────────────────────────────────
# Splitting
# ──────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardWindow:
    """One train/test fold with a purge embargo between them."""
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("train_start", "train_end", "test_start", "test_end"):
            d[k] = d[k].strftime("%Y-%m-%d")
        return d


class PurgedWalkForwardSplitter:
    """Generate rolling train/test windows with an embargo gap.

    embargo_days must be >= the prediction horizon so that no label in
    the training set is computed from prices inside the test window.
    """

    def __init__(
        self,
        train_days: int = 504,      # ~2 trading years (calendar approximation)
        test_days: int = 63,        # ~1 quarter
        embargo_days: int = 10,     # >= prediction horizon (e.g., 5d) + buffer
        step_days: Optional[int] = None,
    ):
        if train_days <= 0 or test_days <= 0:
            raise ValueError("train_days and test_days must be positive")
        if embargo_days < 0:
            raise ValueError("embargo_days cannot be negative")
        self.train_days = train_days
        self.test_days = test_days
        self.embargo_days = embargo_days
        self.step_days = step_days or test_days

    def split(self, start: datetime, end: datetime) -> List[WalkForwardWindow]:
        """Return all complete folds between start and end."""
        if end <= start:
            raise ValueError("end must be after start")
        windows: List[WalkForwardWindow] = []
        fold = 0
        cursor = start
        while True:
            train_start = cursor
            train_end = train_start + timedelta(days=self.train_days)
            test_start = train_end + timedelta(days=self.embargo_days)
            test_end = test_start + timedelta(days=self.test_days)
            if test_end > end:
                break
            windows.append(WalkForwardWindow(fold, train_start, train_end, test_start, test_end))
            fold += 1
            cursor = cursor + timedelta(days=self.step_days)
        return windows


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PredictionRecord:
    """One out-of-sample prediction with its realized outcome."""
    ticker: str
    prob_bullish: float            # calibrated P(direction up)
    predicted_return: float        # predicted horizon return (decimal)
    actual_return: float           # realized horizon return (decimal)
    metadata: Dict[str, Any] = field(default_factory=dict)


def direction_metrics(
    records: Sequence[PredictionRecord],
    buy_threshold: float = 0.65,
    sell_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Asymmetric-threshold signal metrics matching production gating."""
    buys = [r for r in records if r.prob_bullish > buy_threshold]
    sells = [r for r in records if r.prob_bullish < sell_threshold]
    actionable = len(buys) + len(sells)

    buy_hits = sum(1 for r in buys if r.actual_return > 0)
    sell_hits = sum(1 for r in sells if r.actual_return < 0)

    def _rate(hits: int, total: int) -> Optional[float]:
        return round(hits / total, 4) if total else None

    def _wilson_lower(hits: int, total: int) -> Optional[float]:
        # 95% confidence interval lower bound (z = 1.96)
        if total == 0: return None
        p = hits / total
        z = 1.96
        denominator = 1 + z**2 / total
        center = p + z**2 / (2 * total)
        margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
        return round((center - margin) / denominator, 4)

    overall_hits = buy_hits + sell_hits
    return {
        "n_predictions": len(records),
        "n_actionable": actionable,
        "coverage": _rate(actionable, len(records)) if records else None,
        "buy_signals": len(buys),
        "buy_precision": _rate(buy_hits, len(buys)),
        "buy_wilson_lower": _wilson_lower(buy_hits, len(buys)),
        "sell_signals": len(sells),
        "sell_precision": _rate(sell_hits, len(sells)),
        "sell_wilson_lower": _wilson_lower(sell_hits, len(sells)),
        "hit_rate": _rate(overall_hits, actionable),
        "hit_rate_wilson_lower": _wilson_lower(overall_hits, actionable),
    }


def regression_metrics(records: Sequence[PredictionRecord]) -> Dict[str, Any]:
    """Error metrics on the predicted horizon return."""
    pairs = [
        (r.predicted_return, r.actual_return)
        for r in records
        if _finite(r.predicted_return) and _finite(r.actual_return)
    ]
    if not pairs:
        return {"n": 0, "mae": None, "rmse": None, "sign_agreement": None}

    abs_errs = [abs(p - a) for p, a in pairs]
    sq_errs = [(p - a) ** 2 for p, a in pairs]
    sign_hits = sum(1 for p, a in pairs if (p > 0) == (a > 0))
    n = len(pairs)
    return {
        "n": n,
        "mae": round(sum(abs_errs) / n, 6),
        "rmse": round(math.sqrt(sum(sq_errs) / n), 6),
        "sign_agreement": round(sign_hits / n, 4),
    }


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def aggregate_fold_reports(fold_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-fold metrics into a stability-aware summary.

    Reports mean and worst-fold values: a model whose hit rate collapses
    in some folds is overfit even if the average looks good.
    """
    def _collect(path_a: str, path_b: str) -> List[float]:
        vals = []
        for fr in fold_reports:
            v = (fr.get(path_a) or {}).get(path_b)
            if v is not None:
                vals.append(float(v))
        return vals

    hit_rates = _collect("direction", "hit_rate")
    wilson_lowers = _collect("direction", "hit_rate_wilson_lower")
    maes = _collect("regression", "mae")

    def _mean(vals: List[float]) -> Optional[float]:
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "n_folds": len(fold_reports),
        "mean_hit_rate": _mean(hit_rates),
        "worst_fold_hit_rate": round(min(hit_rates), 4) if hit_rates else None,
        "worst_fold_wilson_lower": round(min(wilson_lowers), 4) if wilson_lowers else None,
        "hit_rate_std": _std(hit_rates),
        "mean_mae": _mean(maes),
        "worst_fold_mae": round(max(maes), 6) if maes else None,
    }


def _std(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return round(math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)), 4)


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────

class WalkForwardRunner:
    """Execute a purged walk-forward evaluation for any model.

    train_fn(train_start, train_end) -> model
    predict_fn(model, test_start, test_end) -> List[PredictionRecord]
    """

    def __init__(
        self,
        train_fn: Callable[[datetime, datetime], Any],
        predict_fn: Callable[[Any, datetime, datetime], List[PredictionRecord]],
        splitter: Optional[PurgedWalkForwardSplitter] = None,
        report_path: Optional[Path] = None,
        model_name: str = "unified_model",
    ):
        self.train_fn = train_fn
        self.predict_fn = predict_fn
        self.splitter = splitter or PurgedWalkForwardSplitter()
        self.report_path = Path(report_path) if report_path else DEFAULT_REPORT_PATH
        self.model_name = model_name

    def run(
        self,
        start: datetime,
        end: datetime,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        windows = self.splitter.split(start, end)
        if not windows:
            raise ValueError(
                "Date range too short for the configured train/test/embargo windows"
            )

        fold_reports: List[Dict[str, Any]] = []
        for w in windows:
            if progress_cb:
                progress_cb(w.fold, len(windows))
            logger.info(
                f"Walk-forward fold {w.fold}: train {w.train_start:%Y-%m-%d}→{w.train_end:%Y-%m-%d} "
                f"| embargo {self.splitter.embargo_days}d | test {w.test_start:%Y-%m-%d}→{w.test_end:%Y-%m-%d}"
            )
            model = self.train_fn(w.train_start, w.train_end)
            records = self.predict_fn(model, w.test_start, w.test_end)
            fold_reports.append({
                "window": w.to_dict(),
                "direction": direction_metrics(records),
                "regression": regression_metrics(records),
            })

        if progress_cb:
            progress_cb(len(windows), len(windows))

        report = {
            "model_name": self.model_name,
            "generated_at": datetime.now().isoformat(),
            "protocol": {
                "type": "purged_walk_forward",
                "train_days": self.splitter.train_days,
                "test_days": self.splitter.test_days,
                "embargo_days": self.splitter.embargo_days,
                "step_days": self.splitter.step_days,
            },
            "summary": aggregate_fold_reports(fold_reports),
            "folds": fold_reports,
        }

        if persist:
            self._persist(report)
        return report

    def _persist(self, report: Dict[str, Any]):
        try:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Walk-forward report written to {self.report_path}")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not persist walk-forward report: {e}")


def load_latest_report(report_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load the most recently persisted walk-forward report, if any."""
    path = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not load walk-forward report: {e}")
    return None
