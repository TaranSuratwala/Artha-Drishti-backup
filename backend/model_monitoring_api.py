"""
Model Monitoring API (Feature #2)
=================================
Exposes walk-forward evaluation reports, model health and on-demand
feature drift checks.

Endpoints:
    GET  /api/model/walkforward/report — latest persisted evaluation report
    GET  /api/model/health             — model artifact / predictor status
    POST /api/model/drift/check        — PSI drift check on supplied samples
"""

import logging
from typing import Any, Callable, Optional

from flask import Blueprint, jsonify, request

from drift_monitor import FeatureDriftMonitor
from walkforward import load_latest_report

logger = logging.getLogger(__name__)


def create_model_monitoring_blueprint(
    get_predictor: Optional[Callable[[], Any]] = None,
) -> Blueprint:
    bp = Blueprint("model_monitoring", __name__)

    @bp.route("/api/model/walkforward/report", methods=["GET"])
    def walkforward_report():
        """Latest purged walk-forward evaluation report."""
        report = load_latest_report()
        if report is None:
            return jsonify({
                "status": "error",
                "message": "No walk-forward report available. Run an evaluation first.",
            }), 404
        return jsonify({"status": "success", "data": report})

    @bp.route("/api/model/health", methods=["GET"])
    def model_health():
        """Predictor/model artifact status for monitoring dashboards."""
        info = {"predictor_available": False, "model_loaded": False}
        if get_predictor is not None:
            try:
                predictor = get_predictor()
                info["predictor_available"] = predictor is not None
                info["model_loaded"] = getattr(predictor, "model", None) is not None
                device = getattr(predictor, "device", None)
                if device is not None:
                    info["device"] = str(device)
            except Exception as e:
                info["error"] = str(e)
        report = load_latest_report()
        if report:
            info["last_walkforward"] = {
                "generated_at": report.get("generated_at"),
                "summary": report.get("summary"),
            }
        return jsonify({"status": "success", "data": info})

    @bp.route("/api/model/drift/check", methods=["POST"])
    def drift_check():
        """PSI drift check.

        Body: {"baseline": {feature: [..]}, "live": {feature: [..]},
               "features": [optional subset], "bins": 10}
        """
        data = request.get_json(silent=True) or {}
        baseline = data.get("baseline")
        live = data.get("live")
        if not isinstance(baseline, dict) or not isinstance(live, dict):
            return jsonify({
                "status": "error",
                "message": "'baseline' and 'live' must be objects mapping feature -> list of values",
            }), 400
        try:
            bins = int(data.get("bins", 10))
            monitor = FeatureDriftMonitor(bins=max(4, min(bins, 50)))
            report = monitor.check(baseline, live, features=data.get("features"))
            return jsonify({"status": "success", "data": report})
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    return bp
