"""
Universe Backtesting (Feature #3)
=================================
Runs a strategy backtest across an entire index universe (NIFTY50,
NIFTY500, NIFTYBANK, ...) or a custom ticker list, aggregating
per-ticker results into a portfolio-level report.

Heavy runs execute as background jobs with progress polling so the API
stays responsive. The per-ticker backtest is injected as a callable, so
this module stays decoupled from BacktestEngine internals and is fully
unit-testable.

Endpoints (via create_universe_backtest_blueprint):
    GET  /api/backtest/universes                 — supported index universes
    GET  /api/backtest/universes/<name>          — constituents of an index
    POST /api/backtest/universe/jobs             — submit a universe backtest
    GET  /api/backtest/universe/jobs/<job_id>    — poll job status / result
"""

import logging
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

MAX_UNIVERSE_TICKERS = int(os.getenv("BACKTEST_MAX_UNIVERSE_TICKERS", "600"))
BACKTEST_WORKERS = int(os.getenv("BACKTEST_UNIVERSE_WORKERS", "6"))
JOB_TTL_SECONDS = int(os.getenv("BACKTEST_JOB_TTL_SECONDS", "7200"))


# ──────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────

_METRIC_KEYS = (
    "total_return_pct", "cagr_pct", "sharpe_ratio", "sortino_ratio",
    "max_drawdown_pct", "win_rate_pct", "profit_factor",
)


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def aggregate_universe_results(per_ticker: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-ticker backtest summaries into a universe report."""
    valid = {t: r for t, r in per_ticker.items() if isinstance(r, dict) and r}
    if not valid:
        return {"n_tickers": 0, "message": "No successful backtests"}

    means: Dict[str, Optional[float]] = {}
    for key in _METRIC_KEYS:
        vals = [v for v in (_finite(r.get(key)) for r in valid.values()) if v is not None]
        means[f"mean_{key}"] = round(sum(vals) / len(vals), 4) if vals else None

    total_trades = sum(int(r.get("total_trades") or 0) for r in valid.values())

    returns = {
        t: _finite(r.get("total_return_pct"))
        for t, r in valid.items()
        if _finite(r.get("total_return_pct")) is not None
    }
    profitable = [t for t, v in returns.items() if v > 0]
    best = max(returns, key=returns.get) if returns else None
    worst = min(returns, key=returns.get) if returns else None

    return {
        "n_tickers": len(valid),
        "total_trades": total_trades,
        "profitable_tickers": len(profitable),
        "profitable_pct": round(len(profitable) / len(returns) * 100, 1) if returns else None,
        **means,
        "best_ticker": {"ticker": best, "total_return_pct": returns[best]} if best else None,
        "worst_ticker": {"ticker": worst, "total_return_pct": returns[worst]} if worst else None,
    }


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────

class UniverseBacktestRunner:
    """Run per-ticker backtests in parallel and aggregate the results."""

    def __init__(self, max_workers: int = BACKTEST_WORKERS):
        self.max_workers = max(1, max_workers)

    def run(
        self,
        tickers: List[str],
        run_single_fn: Callable[[str], Dict[str, Any]],
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        start = time.time()
        per_ticker: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}
        done = 0
        total = len(tickers)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(run_single_fn, t): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                done += 1
                try:
                    per_ticker[ticker] = future.result(timeout=300)
                except Exception as e:
                    errors[ticker] = str(e)
                if progress_cb:
                    progress_cb(done, total, ticker)

        # Rank per-ticker results by total return for display
        ranked = sorted(
            ({"ticker": t, **r} for t, r in per_ticker.items()),
            key=lambda x: _finite(x.get("total_return_pct")) or float("-inf"),
            reverse=True,
        )

        return {
            "status": "success",
            "n_requested": total,
            "n_completed": len(per_ticker),
            "n_failed": len(errors),
            "aggregate": aggregate_universe_results(per_ticker),
            "per_ticker": ranked,
            "errors": errors or None,
            "execution_time": round(time.time() - start, 2),
            "timestamp": datetime.now().isoformat(),
        }


# ──────────────────────────────────────────────────────────────────────
# Job manager (in-process; queue-agnostic contract)
# ──────────────────────────────────────────────────────────────────────

class BacktestJobManager:
    def __init__(self, max_workers: int = 2, job_ttl: int = JOB_TTL_SECONDS):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="btjob")
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._job_ttl = job_ttl

    def submit(self, fn: Callable[[Callable[[int, int, str], None]], Dict[str, Any]],
               meta: Optional[Dict[str, Any]] = None) -> str:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._cleanup_locked()
            self._jobs[job_id] = {
                "id": job_id, "state": "queued",
                "progress": {"done": 0, "total": 0, "current": None},
                "meta": meta or {}, "created_at": now, "updated_at": now,
                "result": None, "error": None,
            }

        def _progress(done: int, total: int, current: str):
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job["progress"] = {"done": done, "total": total, "current": current}
                    job["updated_at"] = time.time()

        def _run():
            with self._lock:
                job = self._jobs.get(job_id)
                if not job:
                    return
                job["state"] = "running"
                job["updated_at"] = time.time()
            try:
                result = fn(_progress)
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job:
                        job["state"] = "completed"
                        job["result"] = result
                        job["updated_at"] = time.time()
            except Exception as e:
                logger.exception(f"Universe backtest job {job_id} failed")
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job:
                        job["state"] = "failed"
                        job["error"] = str(e)
                        job["updated_at"] = time.time()

        self._executor.submit(_run)
        return job_id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._cleanup_locked()
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def _cleanup_locked(self):
        cutoff = time.time() - self._job_ttl
        stale = [jid for jid, j in self._jobs.items()
                 if j["updated_at"] < cutoff and j["state"] in {"completed", "failed"}]
        for jid in stale:
            self._jobs.pop(jid, None)


# ──────────────────────────────────────────────────────────────────────
# Blueprint
# ──────────────────────────────────────────────────────────────────────

def create_universe_backtest_blueprint(
    run_single_factory: Callable[[Dict[str, Any]], Callable[[str], Dict[str, Any]]],
    job_manager: Optional[BacktestJobManager] = None,
) -> Blueprint:
    """Build the universe backtest blueprint.

    run_single_factory(payload) returns a callable(ticker) -> summary dict
    that runs the configured single-ticker backtest. Injected by
    application.py so this module stays engine-agnostic.
    """
    bp = Blueprint("universe_backtest", __name__)
    jobs = job_manager or BacktestJobManager()

    @bp.route("/api/backtest/universes", methods=["GET"])
    def list_universes():
        from index_universe import get_index_universe
        return jsonify({"status": "success", "data": get_index_universe().list_supported()})

    @bp.route("/api/backtest/universes/<name>", methods=["GET"])
    def universe_constituents(name):
        from index_universe import get_index_universe
        try:
            info = get_index_universe().get_info(name)
            return jsonify({"status": "success", "data": info})
        except ValueError as ve:
            return jsonify({"status": "error", "message": str(ve)}), 400
        except Exception as e:
            logger.error(f"Universe constituents error for {name}: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @bp.route("/api/backtest/universe/jobs", methods=["POST"])
    def submit_universe_backtest():
        """Submit a universe backtest.

        Body: {"universe": "NIFTY50" | "tickers": [...], "strategy": "momentum",
               "initial_capital": 100000, "start_date": ..., "end_date": ...}
        """
        from index_universe import get_index_universe
        data = request.get_json(silent=True) or {}

        tickers = data.get("tickers")
        universe = data.get("universe")
        if not tickers and not universe:
            return jsonify({
                "status": "error",
                "message": "Provide either 'universe' (e.g. NIFTY50) or 'tickers' (list)",
            }), 400

        try:
            if not tickers:
                tickers = get_index_universe().get_constituents(universe)
        except ValueError as ve:
            return jsonify({"status": "error", "message": str(ve)}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

        tickers = [str(t).strip().upper() for t in tickers if str(t).strip()][:MAX_UNIVERSE_TICKERS]
        if not tickers:
            return jsonify({"status": "error", "message": "Universe resolved to zero tickers"}), 400

        try:
            run_single_fn = run_single_factory(data)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid configuration: {e}"}), 400

        runner = UniverseBacktestRunner()

        def _job(progress_cb):
            result = runner.run(tickers, run_single_fn, progress_cb=progress_cb)
            result["universe"] = universe or "custom"
            result["strategy"] = data.get("strategy", "momentum")
            return result

        job_id = jobs.submit(_job, meta={
            "universe": universe or "custom",
            "strategy": data.get("strategy", "momentum"),
            "n_tickers": len(tickers),
        })
        return jsonify({
            "status": "accepted",
            "job_id": job_id,
            "n_tickers": len(tickers),
            "poll_url": f"/api/backtest/universe/jobs/{job_id}",
        }), 202

    @bp.route("/api/backtest/universe/jobs/<job_id>", methods=["GET"])
    def universe_job_status(job_id):
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"status": "error", "message": "Job not found or expired"}), 404
        payload = {
            "status": "success",
            "job": {
                "id": job["id"], "state": job["state"],
                "progress": job["progress"], "meta": job["meta"],
                "error": job["error"],
            },
        }
        if job["state"] == "completed":
            payload["job"]["result"] = job["result"]
        return jsonify(payload)

    return bp
