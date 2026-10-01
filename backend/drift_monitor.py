"""
Feature Drift Monitor (Feature #2)
==================================
Detects distribution shift between the feature data a model was trained
on (baseline) and the data it currently scores (live), using the
Population Stability Index (PSI).

Interpretation (industry convention):
    PSI < 0.10        — no significant shift
    0.10 ≤ PSI < 0.25 — moderate shift, monitor closely
    PSI ≥ 0.25        — severe shift, retraining recommended

Drift on key features is one of the strongest leading indicators of
prediction degradation, so this feeds the auto-retraining decision in
the scheduler.
"""

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

PSI_MODERATE = 0.10
PSI_SEVERE = 0.25
_EPS = 1e-6


def population_stability_index(
    expected: Sequence[float],
    actual: Sequence[float],
    bins: int = 10,
) -> float:
    """Compute PSI between a baseline (expected) and live (actual) sample.

    Bin edges are derived from the baseline's quantiles so each baseline
    bin holds ~equal mass, which makes PSI robust to outliers.
    """
    exp = [float(x) for x in expected if _finite(x)]
    act = [float(x) for x in actual if _finite(x)]
    if len(exp) < bins or len(act) < 1:
        raise ValueError("Not enough finite samples to compute PSI")

    exp_sorted = sorted(exp)
    edges = [exp_sorted[int(len(exp_sorted) * i / bins)] for i in range(1, bins)]

    def _bucketise(values: List[float]) -> List[int]:
        counts = [0] * bins
        for v in values:
            idx = 0
            for e in edges:
                if v > e:
                    idx += 1
                else:
                    break
            counts[idx] += 1
        return counts

    exp_counts = _bucketise(exp)
    act_counts = _bucketise(act)

    psi = 0.0
    for ec, ac in zip(exp_counts, act_counts):
        e_pct = max(ec / len(exp), _EPS)
        a_pct = max(ac / len(act), _EPS)
        psi += (a_pct - e_pct) * math.log(a_pct / e_pct)
    return round(psi, 6)


def classify_psi(psi: float) -> str:
    if psi >= PSI_SEVERE:
        return "severe"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


class FeatureDriftMonitor:
    """Compare per-feature distributions between baseline and live data."""

    def __init__(self, bins: int = 10):
        self.bins = bins

    def check(
        self,
        baseline: Dict[str, Sequence[float]],
        live: Dict[str, Sequence[float]],
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return a drift report for the intersection of feature columns."""
        cols = features or sorted(set(baseline.keys()) & set(live.keys()))
        per_feature: Dict[str, Dict[str, Any]] = {}
        severe, moderate = [], []

        for col in cols:
            if col not in baseline or col not in live:
                continue
            try:
                psi = population_stability_index(baseline[col], live[col], bins=self.bins)
            except ValueError as e:
                per_feature[col] = {"psi": None, "status": "insufficient_data", "detail": str(e)}
                continue
            status = classify_psi(psi)
            per_feature[col] = {"psi": psi, "status": status}
            if status == "severe":
                severe.append(col)
            elif status == "moderate":
                moderate.append(col)

        overall = "severe" if severe else ("moderate" if moderate else "stable")
        
        # Per-feature drift attribution
        if overall in ["severe", "moderate"]:
            top_drifted = sorted(
                [(col, data["psi"]) for col, data in per_feature.items() if data["psi"] is not None],
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            logger.info(f"Top-5 drifted features: {[(f, f'{psi:.4f}') for f, psi in top_drifted]}")

        return {
            "generated_at": datetime.now().isoformat(),
            "overall_status": overall,
            "trigger_recalibration": bool(severe),
            "severe_features": severe,
            "moderate_features": moderate,
            "features_checked": len(per_feature),
            "per_feature": per_feature,
            "thresholds": {"moderate": PSI_MODERATE, "severe": PSI_SEVERE},
        }
