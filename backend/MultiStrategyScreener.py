"""
Multi-Strategy Screener (Feature #1)
====================================
Runs one or more screening strategies (predefined or custom) over the
full NSE universe and combines results with AND / OR logic.

Design goals:
- **Low latency API**: full-universe scans run as background jobs with
  status polling (POST /api/screener/multi/jobs → GET .../jobs/<id>).
- **Reuse**: delegates per-strategy screening to the existing
  InteractiveStockScreener (StockScreener.py) so condition evaluation,
  bulk DB feature fetching and caching are shared with single-strategy
  screening.
- **Replaceable executor**: the in-process ThreadPoolExecutor job
  manager can later be swapped for Celery/RQ without changing the API
  contract.

Endpoints (registered via create_multi_screener_blueprint):
    GET  /api/screener/universe                — NSE universe metadata
    POST /api/screener/multi/run               — synchronous combined screen
    POST /api/screener/multi/jobs              — submit async combined screen
    GET  /api/screener/multi/jobs/<job_id>     — poll job status / result
"""

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

VALID_LOGIC = {"AND", "OR"}
MAX_STRATEGIES_PER_RUN = int(os.getenv("SCREENER_MAX_STRATEGIES", "8"))
JOB_TTL_SECONDS = int(os.getenv("SCREENER_JOB_TTL_SECONDS", "3600"))
JOB_WORKERS = int(os.getenv("SCREENER_JOB_WORKERS", "2"))


# ──────────────────────────────────────────────────────────────────────
# Result combination
# ──────────────────────────────────────────────────────────────────────

def combine_screen_results(
    results_by_strategy: Dict[str, List[Dict[str, Any]]],
    logic: str = "AND",
    max_results: int = 50,
) -> List[Dict[str, Any]]:
    """Combine per-strategy screen results by ticker using AND / OR logic.

    AND → ticker must satisfy every selected strategy.
    OR  → ticker must satisfy at least one selected strategy.

    Returns entries sorted by combined score (average of per-strategy
    scores; for OR, strategies that did not match contribute 0).
    """
    logic = (logic or "AND").upper()
    if logic not in VALID_LOGIC:
        raise ValueError(f"logic must be one of {sorted(VALID_LOGIC)}")

    strategy_names = list(results_by_strategy.keys())
    if not strategy_names:
        return []

    by_ticker: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for strat, results in results_by_strategy.items():
        for entry in results or []:
            ticker = entry.get("ticker")
            if not ticker:
                continue
            by_ticker.setdefault(ticker, {})[strat] = entry

    combined: List[Dict[str, Any]] = []
    n_strategies = len(strategy_names)

    for ticker, hits in by_ticker.items():
        if logic == "AND" and len(hits) < n_strategies:
            continue

        matched = sorted(hits.keys())
        score_sum = sum(float(h.get("score") or 0) for h in hits.values())
        # AND: average across all (all matched). OR: average across all
        # selected strategies so broader agreement ranks higher.
        combined_score = score_sum / n_strategies

        key_indicators: Dict[str, Any] = {}
        current_price = None
        per_strategy: Dict[str, Dict[str, Any]] = {}
        for strat in matched:
            h = hits[strat]
            if current_price is None:
                current_price = h.get("current_price")
            for k, v in (h.get("key_indicators") or {}).items():
                key_indicators.setdefault(k, v)
            per_strategy[strat] = {
                "score": h.get("score"),
                "conditions_passed": h.get("conditions_passed"),
                "total_conditions": h.get("total_conditions"),
                "confidence": h.get("confidence"),
            }

        combined.append({
            "ticker": ticker,
            "combined_score": round(combined_score, 2),
            "current_price": current_price,
            "matched_strategies": matched,
            "matched_count": len(matched),
            "total_strategies": n_strategies,
            "strategies": per_strategy,
            "key_indicators": key_indicators,
        })

    combined.sort(key=lambda x: (x["matched_count"], x["combined_score"]), reverse=True)
    return combined[:max_results]


def run_multi_screen(
    screener,
    strategy_names: List[str],
    logic: str = "AND",
    max_results: int = 50,
    overrides_by_strategy: Optional[Dict[str, Dict[str, Any]]] = None,
    per_strategy_pool: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Run each strategy through the existing screener and combine results.

    per_strategy_pool controls how many candidates each strategy keeps
    before combination (large enough that AND-intersections survive).
    """
    start = time.time()
    overrides_by_strategy = overrides_by_strategy or {}
    pool = per_strategy_pool or max(max_results * 10, 500)

    results_by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    errors: Dict[str, str] = {}

    total = len(strategy_names)
    for idx, name in enumerate(strategy_names, 1):
        if progress_cb:
            progress_cb(idx - 1, total, name)
        try:
            res = screener.run_screening(
                name, max_results=pool, overrides=overrides_by_strategy.get(name)
            )
            if res.get("status") == "success":
                results_by_strategy[name] = res.get("results", [])
            else:
                errors[name] = res.get("message", "unknown screening error")
                results_by_strategy[name] = []
        except Exception as e:
            logger.error(f"Multi-screen: strategy '{name}' failed: {e}")
            errors[name] = str(e)
            results_by_strategy[name] = []
        if progress_cb:
            progress_cb(idx, total, name)

    combined = combine_screen_results(results_by_strategy, logic=logic, max_results=max_results)

    return {
        "status": "success",
        "logic": logic.upper(),
        "strategies": strategy_names,
        "count": len(combined),
        "results": combined,
        "per_strategy_counts": {k: len(v) for k, v in results_by_strategy.items()},
        "errors": errors or None,
        "execution_time": round(time.time() - start, 2),
        "timestamp": datetime.now().isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────
# Background job manager
# ──────────────────────────────────────────────────────────────────────

class ScreenerJobManager:
    """Minimal in-process async job manager with status polling.

    Interface is intentionally queue-agnostic so it can be replaced by
    Celery/RQ later without changing the HTTP contract.
    """

    def __init__(self, max_workers: int = JOB_WORKERS, job_ttl: int = JOB_TTL_SECONDS):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="screenjob")
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._job_ttl = job_ttl

    def submit(self, fn: Callable[[Callable[[int, int, str], None]], Dict[str, Any]],
               meta: Optional[Dict[str, Any]] = None) -> str:
        """Submit a callable(progress_cb) and return a job id."""
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._cleanup_locked()
            self._jobs[job_id] = {
                "id": job_id,
                "state": "queued",
                "progress": {"done": 0, "total": 0, "current": None},
                "meta": meta or {},
                "created_at": now,
                "updated_at": now,
                "result": None,
                "error": None,
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
                logger.exception(f"Screener job {job_id} failed")
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
# Flask blueprint
# ──────────────────────────────────────────────────────────────────────

def _parse_multi_payload(data: Dict[str, Any]):
    strategies = data.get("strategies") or data.get("strategy_names")
    if isinstance(strategies, str):
        strategies = [strategies]
    if not strategies or not isinstance(strategies, list):
        raise ValueError("'strategies' must be a non-empty list of strategy names")
    strategies = [str(s).strip() for s in strategies if str(s).strip()]
    if not strategies:
        raise ValueError("'strategies' must contain at least one strategy name")
    if len(strategies) > MAX_STRATEGIES_PER_RUN:
        raise ValueError(f"At most {MAX_STRATEGIES_PER_RUN} strategies per run")

    logic = str(data.get("logic", "AND")).upper()
    if logic not in VALID_LOGIC:
        raise ValueError("'logic' must be 'AND' or 'OR'")

    max_results = int(data.get("max_results", 50))
    max_results = max(1, min(max_results, 500))

    overrides = data.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("'overrides' must be an object keyed by strategy name")

    return strategies, logic, max_results, overrides


def create_multi_screener_blueprint(
    get_screener: Callable[[], Any],
    get_pipeline: Optional[Callable[[], Any]] = None,
    job_manager: Optional[ScreenerJobManager] = None,
) -> Blueprint:
    """Build the multi-strategy screener blueprint.

    get_screener / get_pipeline are zero-arg callables so the blueprint
    can be registered before the heavy singletons finish initialising.
    """
    bp = Blueprint("multi_screener", __name__)
    jobs = job_manager or ScreenerJobManager()

    def _fallback_tickers() -> List[str]:
        if get_pipeline is None:
            return []
        try:
            pipeline = get_pipeline()
            rows = pipeline.get_latest_data(limit=None) or []
            return sorted({r.get("ticker") for r in rows if r.get("ticker")})
        except Exception as e:
            logger.warning(f"Universe fallback via pipeline failed: {e}")
            return []

    @bp.route("/api/screener/universe", methods=["GET"])
    def universe_info():
        """NSE universe metadata. ?refresh=true forces re-download;
        ?symbols=true includes the full symbol list."""
        from nse_universe import get_nse_universe
        try:
            provider = get_nse_universe(fallback_tickers_fn=_fallback_tickers)
            if request.args.get("refresh", "").lower() in {"1", "true", "yes"}:
                info = provider.refresh()
            else:
                info = provider.get_info()
            payload = {"status": "success", "data": info}
            if request.args.get("symbols", "").lower() in {"1", "true", "yes"}:
                payload["data"]["symbols"] = provider.get_symbols()
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Universe endpoint error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @bp.route("/api/screener/multi/run", methods=["POST"])
    def multi_run_sync():
        """Synchronous combined screen (suitable for small strategy sets)."""
        try:
            data = request.get_json(silent=True) or {}
            strategies, logic, max_results, overrides = _parse_multi_payload(data)
        except ValueError as ve:
            return jsonify({"status": "error", "message": str(ve)}), 400
        try:
            result = run_multi_screen(
                get_screener(), strategies, logic=logic,
                max_results=max_results, overrides_by_strategy=overrides,
            )
            return jsonify(result)
        except Exception as e:
            logger.error(f"Multi-screen sync error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @bp.route("/api/screener/multi/jobs", methods=["POST"])
    def multi_run_async():
        """Submit a combined screen as a background job (202 + job id)."""
        try:
            data = request.get_json(silent=True) or {}
            strategies, logic, max_results, overrides = _parse_multi_payload(data)
        except ValueError as ve:
            return jsonify({"status": "error", "message": str(ve)}), 400

        screener = get_screener()

        def _job(progress_cb):
            return run_multi_screen(
                screener, strategies, logic=logic,
                max_results=max_results, overrides_by_strategy=overrides,
                progress_cb=progress_cb,
            )

        job_id = jobs.submit(_job, meta={
            "strategies": strategies, "logic": logic, "max_results": max_results,
        })
        return jsonify({
            "status": "accepted",
            "job_id": job_id,
            "poll_url": f"/api/screener/multi/jobs/{job_id}",
        }), 202

    @bp.route("/api/screener/multi/jobs/<job_id>", methods=["GET"])
    def multi_job_status(job_id):
        """Poll a background screening job."""
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"status": "error", "message": "Job not found or expired"}), 404
        payload = {
            "status": "success",
            "job": {
                "id": job["id"],
                "state": job["state"],
                "progress": job["progress"],
                "meta": job["meta"],
                "error": job["error"],
            },
        }
        if job["state"] == "completed":
            payload["job"]["result"] = job["result"]
        return jsonify(payload)

    return bp
