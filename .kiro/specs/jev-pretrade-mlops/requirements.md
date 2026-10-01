# Requirements Document

## Introduction

This feature covers three interconnected areas of the **Artha Drishti** AI-driven financial platform (Python/FastAPI/Flask backend):

1. **JEV Pre-Trade Gate API Testing** — a comprehensive pytest-based test harness that exercises the `/api/predict/{ticker}` endpoint and its embedded `JevPreTradeGate` logic, covering both the TypeSafe SDK path and the heuristic fallback chain.
2. **MLOps Pipeline Implementation** — a production-grade model lifecycle system layered on top of the existing `MLOpsTracker` (MLflow), `MultiSeedCertifier`, `FeatureDriftMonitor`, `tasks.py` (Celery), and `model_monitoring_api.py`, adding structured training pipelines, model versioning, experiment comparison, automated drift-triggered retraining, model promotion gates, and alerting.
3. **MLOps Usage and Tracking Guide** — living documentation (Markdown) that explains how to operate, monitor, and extend the MLOps system, including runbooks for common failure scenarios.

---

## Glossary

- **JEV_Gate**: The `JevPreTradeGate` class in `jev_pretrade_gate.py` that evaluates market microstructure conditions before order placement.
- **MarketSnapshot**: Structured dataclass passed to `JEV_Gate.evaluate()` containing ticker, ML probability, ensemble std, bid/ask spread, volume ratio, PSI drift score, regime state, ATR%, signal type, position size, and days since last trade.
- **JevDecision**: Typed response from JEV_Gate with action (`EXECUTE`/`WAIT`/`SKIP`), action confidence, execution quality score (0–10), liquidity judgment, and latency.
- **Heuristic_Fallback**: The deterministic `_heuristic_fallback()` method inside `JEV_Gate` that is used when the TypeSafe SDK is unavailable.
- **Test_Harness**: The pytest-based API test suite in `backend/tests/` that exercises the JEV gate via Flask test client.
- **MLOps_Pipeline**: The end-to-end system managing training, versioning, drift monitoring, promotion, serving, and alerting for the `UnifiedStockPredictor` and `SeedEnsemblePredictor` models.
- **MLflow**: Open-source ML experiment tracking library, already integrated in `mlops_tracker.py`.
- **MLOps_Tracker**: The `MLOpsTracker` singleton (`mlops_tracker.py`) wrapping MLflow for experiment tracking.
- **Certification**: The multi-seed go/no-go evaluation process run by `MultiSeedCertifier` producing a `CertificationReport`.
- **PSI**: Population Stability Index — a measure of feature distribution shift computed by `FeatureDriftMonitor` in `drift_monitor.py`.
- **Celery_Worker**: The async task executor defined in `celery_app.py` and `tasks.py` that runs training, drift checks, and certification tasks.
- **Model_Registry**: MLflow's model registry component that stores versioned, named model artifacts with lifecycle stages (Staging, Production, Archived).
- **Drift_Alert**: An automated notification triggered when PSI drift classification exceeds a defined threshold.
- **Promotion_Gate**: The automated check that a model must pass (certification + drift baseline) before it transitions from `Staging` to `Production` in the Model_Registry.
- **Serving_Endpoint**: The existing `/api/predict/{ticker}` endpoint in `application.py` that loads the active Production model and generates signals via `UnifiedStockPredictor`.
- **Usage_Guide**: The Markdown documentation file at `docs/mlops_guide.md` describing the full MLOps workflow for operators and contributors.

---

## Requirements

### Requirement 1: JEV Pre-Trade Gate Unit Tests

**User Story:** As a backend engineer, I want a pytest test suite for the `JevPreTradeGate` class, so that I can verify gate logic in isolation without a live server or TypeSafe SDK.

#### Acceptance Criteria

1. THE `Test_Harness` SHALL contain at least one test per `ExecutionDecision` outcome (`EXECUTE`, `WAIT`, `SKIP`) using the `Heuristic_Fallback` path.
2. WHEN a `MarketSnapshot` has `psi_drift_score > 1.0`, THE `JEV_Gate` SHALL return a `JevDecision` with `action == "SKIP"` via `Heuristic_Fallback`.
3. WHEN a `MarketSnapshot` has `psi_drift_score > 0.5` and `psi_drift_score <= 1.0`, THE `JEV_Gate` SHALL return a `JevDecision` with `action == "WAIT"` via `Heuristic_Fallback`.
4. WHEN a `MarketSnapshot` has `volume_ratio_20d < 0.3`, THE `JEV_Gate` SHALL return a `JevDecision` with `is_liquid == False` via `Heuristic_Fallback`.
5. WHEN a `MarketSnapshot` has `bid_ask_spread_bps > 50`, THE `JEV_Gate` SHALL return a `JevDecision` with `execution_quality_score` reduced by at least 2.0 compared to a clean snapshot via `Heuristic_Fallback`.
6. WHEN a `MarketSnapshot` has all conditions nominal, THE `JEV_Gate` SHALL return a `JevDecision` with `action == "EXECUTE"` and `fallback_used == True` via `Heuristic_Fallback`.
7. THE `Test_Harness` SHALL verify that `JEV_Gate.should_execute()` returns `False` for any `JevDecision` with `action != "EXECUTE"`.
8. WHEN `action_confidence < min_execute_confidence`, THE `JEV_Gate.should_execute()` SHALL return `False`.
9. WHEN `is_liquid == False` and `is_liquid_confidence > min_liquidity_confidence`, THE `JEV_Gate.should_execute()` SHALL return `False`.
10. THE `Test_Harness` SHALL cover the `_build_state_text()` method, verifying that the returned string contains the ticker symbol, ML probability, and regime state from the input `MarketSnapshot`.
11. FOR ALL valid `MarketSnapshot` inputs with all-nominal conditions, constructing a `JevDecision` via `Heuristic_Fallback` and passing it to `should_execute()` SHALL produce consistent `True` or `False` results (idempotence property).

---

### Requirement 2: JEV Pre-Trade Gate API Integration Tests

**User Story:** As a QA engineer, I want end-to-end API tests for the `/api/predict/{ticker}` endpoint that confirm the JEV gate response is embedded in the prediction payload, so that I can verify the gate is wired correctly into the serving pipeline.

#### Acceptance Criteria

1. WHEN `POST /api/predict/{ticker}` is called with a valid ticker and JWT token, THE `Serving_Endpoint` SHALL return HTTP 200 with a JSON body containing a `jev_gate` key.
2. WHEN `POST /api/predict/{ticker}` is called with a valid ticker and JWT token, THE `Serving_Endpoint` SHALL return a `jev_gate` object containing at minimum the keys `action`, `action_confidence`, `execution_quality_score`, `is_liquid`, and `fallback_used`.
3. WHEN `POST /api/predict/{ticker}` returns HTTP 200, THE `Serving_Endpoint` SHALL include a top-level `signal` key with value in `{"BUY", "SELL", "HOLD"}`.
4. WHEN the `JEV_Gate` blocks a BUY signal, THE `Serving_Endpoint` SHALL return `signal == "HOLD"` and include `signal_before_gate` and `signal_override_reason` in the response body.
5. WHEN `POST /api/predict/{ticker}` is called without a JWT token, THE `Serving_Endpoint` SHALL return HTTP 401.
6. WHEN `POST /api/predict/{ticker}` is called with a ticker symbol containing special characters, THE `Serving_Endpoint` SHALL return HTTP 400 with an `error` key.
7. THE `Test_Harness` SHALL use Flask's built-in test client (`app.test_client()`) and `create_access_token` from `flask_jwt_extended` rather than making live HTTP requests.
8. THE `Test_Harness` SHALL include a parametrized test covering at least 3 different valid NSE tickers to verify consistent response schema across tickers.

---

### Requirement 3: JEV Gate Boundary and Property Tests

**User Story:** As a senior engineer, I want property-based and boundary tests for the JEV gate heuristics, so that edge cases at PSI thresholds, confidence bounds, and spread limits are systematically validated.

#### Acceptance Criteria

1. THE `Test_Harness` SHALL include a property-based test using `hypothesis` that generates random `MarketSnapshot` inputs and verifies that `JEV_Gate._heuristic_fallback()` always returns a `JevDecision` without raising an exception.
2. FOR ALL `MarketSnapshot` inputs generated by the property test, THE `JevDecision` returned by `Heuristic_Fallback` SHALL have `execution_quality_score` in the range `[0.0, 10.0]`.
3. FOR ALL `MarketSnapshot` inputs, THE `JevDecision.action` returned by `Heuristic_Fallback` SHALL be one of `{"EXECUTE", "WAIT", "SKIP"}`.
4. WHEN `psi_drift_score` is exactly at the boundary value `0.5`, THE `JEV_Gate` SHALL classify it as `"WAIT"` (boundary is inclusive at the lower threshold per `PSI_SEVERE = 0.25` convention in `drift_monitor.py` — gate uses `> 0.5` threshold).
5. WHEN `psi_drift_score` is exactly at the boundary value `1.0`, THE `JEV_Gate` SHALL classify it as `"WAIT"` (boundary is not `SKIP` since the condition is `> 1.0`).
6. THE `Test_Harness` SHALL verify the round-trip property: serializing a `MarketSnapshot` to the state text via `_build_state_text()` and back via field extraction preserves the ticker symbol without modification.

---

### Requirement 4: Training Pipeline with Experiment Tracking

**User Story:** As an ML engineer, I want every model training run to be tracked in MLflow with full hyperparameters, metrics, and artifact logging, so that I can compare experiments and reproduce any model.

#### Acceptance Criteria

1. WHEN `predictor.train()` is called, THE `MLOps_Pipeline` SHALL call `MLOps_Tracker.start_run()` at the start and `MLOps_Tracker.end_run()` at completion, whether training succeeds or fails.
2. WHEN a training run completes, THE `MLOps_Tracker` SHALL log all hyperparameters from the `CONFIG` dictionary in `MLPredictor.py` as MLflow parameters.
3. WHEN a training run completes, THE `MLOps_Tracker` SHALL log evaluation metrics including `direction_accuracy`, `sharpe_ratio`, `win_rate_pct`, `max_drawdown_pct`, and `rank_ic_mean` as MLflow metrics.
4. WHEN a training run completes, THE `MLOps_Tracker` SHALL log the trained PyTorch model artifact, the feature scaler (`.pkl`), and the feature column list (`.pkl`) to the MLflow run.
5. WHEN a training run fails with an exception, THE `MLOps_Tracker` SHALL log the exception message as an MLflow tag `training_failure` before ending the run.
6. THE `MLOps_Pipeline` SHALL assign a human-readable run name following the pattern `{model_type}-{timestamp}` (e.g., `attention_lstm-20260115_143022`) to each training run.
7. THE `MLOps_Tracker` SHALL log the number of training tickers, training date range start, training date range end, and random seed as MLflow parameters.

---

### Requirement 5: Model Versioning and Registry

**User Story:** As an ML engineer, I want every trained model registered in MLflow's Model Registry with lifecycle stages, so that I can promote a model from Staging to Production with a single API call.

#### Acceptance Criteria

1. WHEN a training run completes and metrics exceed minimum quality thresholds, THE `MLOps_Pipeline` SHALL register the model in the MLflow Model Registry under the name `ArthaDrishti-StockPredictor`.
2. WHEN a model is registered after training, THE `MLOps_Pipeline` SHALL set the initial registry stage to `Staging`.
3. WHEN a `CertificationReport` with `deployment_ready == True` is produced for a Staging model, THE `MLOps_Pipeline` SHALL automatically promote that model version to `Production` stage in the registry.
4. WHEN a model is promoted to `Production`, THE `MLOps_Pipeline` SHALL archive the previously active `Production` model version.
5. WHEN a `Production` model version exists in the registry, THE `Serving_Endpoint` SHALL load the `Production` version artifact path at startup rather than using a hardcoded file path.
6. THE `MLOps_Pipeline` SHALL tag each registered model version with `cache_hash`, `n_seeds`, `median_sharpe`, and `deployment_timestamp`.

---

### Requirement 6: Automated Drift Monitoring and Retraining Trigger

**User Story:** As an MLOps operator, I want the system to automatically detect feature drift and trigger a retraining Celery task when drift reaches the severe threshold, so that the model never operates significantly out-of-distribution.

#### Acceptance Criteria

1. THE `Celery_Worker` SHALL run a drift check task on a configurable schedule (default: every 6 hours) using `FeatureDriftMonitor.check()` comparing a stored baseline feature snapshot to recent live inference features.
2. WHEN the drift check returns `overall_status == "severe"`, THE `Celery_Worker` SHALL enqueue a `train_unified_model_task` within 5 minutes of the drift check completing.
3. WHEN the drift check returns `overall_status == "moderate"`, THE `MLOps_Pipeline` SHALL log a warning via Python `logging` and increment a `drift_moderate_count` metric in MLflow.
4. WHEN a new baseline feature snapshot is required, THE `MLOps_Pipeline` SHALL compute it from the most recent 30 days of live inference feature data stored in the database.
5. WHEN a drift-triggered retraining task completes, THE `MLOps_Pipeline` SHALL update the stored baseline snapshot to the post-training feature distribution.
6. THE `MLOps_Pipeline` SHALL expose a `POST /api/model/drift/baseline/update` endpoint that allows operators to manually refresh the baseline snapshot from the last N days of data (default: 30 days).
7. WHEN the drift check is run, THE `MLOps_Pipeline` SHALL persist the full `FeatureDriftMonitor` report (JSON) to a file at `logs/drift_reports/drift_{timestamp}.json` for audit purposes.

---

### Requirement 7: Model Serving and Hot-Reload

**User Story:** As a platform engineer, I want the prediction serving layer to support hot-reloading of a new Production model without restarting the Flask server, so that model promotions cause zero downtime.

#### Acceptance Criteria

1. WHEN a new model version is promoted to `Production` in the registry, THE `Serving_Endpoint` SHALL reload the model artifact within 60 seconds without restarting the Flask process.
2. THE `Serving_Endpoint` SHALL maintain a read lock during inference and a write lock during model reload, preventing concurrent prediction requests from reading a partially loaded model.
3. WHEN a model reload fails, THE `Serving_Endpoint` SHALL retain the previously loaded model and log the reload failure at ERROR level.
4. THE `MLOps_Pipeline` SHALL expose a `POST /api/model/reload` endpoint (JWT-protected) that triggers an immediate hot-reload of the current Production model.
5. WHEN `GET /api/model/health` is called, THE `Serving_Endpoint` SHALL include `active_model_version`, `last_reloaded_at`, and `registry_stage` in the response JSON.

---

### Requirement 8: Monitoring, Alerting, and Metrics Dashboard

**User Story:** As an MLOps operator, I want real-time model performance metrics and configurable alerts, so that I can detect prediction degradation before it affects trading signals.

#### Acceptance Criteria

1. THE `MLOps_Pipeline` SHALL expose a `GET /api/model/metrics/live` endpoint that returns prediction statistics for the last 24 hours including `total_predictions`, `buy_signal_pct`, `sell_signal_pct`, `hold_signal_pct`, `avg_direction_probability`, and `avg_ensemble_std`.
2. WHEN `avg_ensemble_std` over the last 100 predictions exceeds `0.08`, THE `MLOps_Pipeline` SHALL log a WARNING with the message `"High ensemble disagreement: σ={value:.3f}"`.
3. THE `MLOps_Pipeline` SHALL write a Prometheus-compatible metrics text file at `logs/metrics/model_metrics.prom` on every prediction, readable by a Prometheus scrape endpoint.
4. WHEN configured with `ADMIN_EMAIL` in `config.py`, THE `MLOps_Pipeline` SHALL send an email alert via the configured SMTP server when drift status transitions from `stable` to `severe`.
5. THE `MLOps_Pipeline` SHALL store a rolling prediction log in PostgreSQL with columns: `prediction_id`, `ticker`, `timestamp`, `direction_probability`, `ensemble_std`, `signal`, `jev_action`, `jev_quality_score`, `model_version`.
6. WHEN the prediction log table contains at least 100 rows for a ticker, THE `MLOps_Pipeline` SHALL compute and expose a `calibration_error` metric via `GET /api/model/metrics/live`.

---

### Requirement 9: MLOps Usage and Tracking Guide

**User Story:** As a new engineer or operator joining the Artha Drishti project, I want a comprehensive usage guide for the MLOps system, so that I can operate, monitor, and extend the system without needing to read all source code.

#### Acceptance Criteria

1. THE `Usage_Guide` SHALL be written in Markdown and stored at `docs/mlops_guide.md` in the repository root.
2. THE `Usage_Guide` SHALL include a section on **Quick Start** that describes the minimum commands required to start the Celery worker, beat scheduler, and Flask server, with exact commands for both development and production environments.
3. THE `Usage_Guide` SHALL include a section on **Training a Model** that describes how to trigger training manually (via Celery task, via API endpoint `POST /api/model/train`), monitor progress in the MLflow UI, and interpret training run metrics.
4. THE `Usage_Guide` SHALL include a section on **Model Promotion** that explains the promotion workflow: Staging → Certification → Production, and the commands/API calls required for each step.
5. THE `Usage_Guide` SHALL include a section on **Drift Monitoring** that explains PSI thresholds (`stable < 0.10`, `0.10 ≤ moderate < 0.25`, `severe ≥ 0.25`), how to read drift reports from `logs/drift_reports/`, and how to manually trigger a drift check via `POST /api/model/drift/check`.
6. THE `Usage_Guide` SHALL include a section on **Experiment Tracking** that describes how to launch the MLflow UI locally, navigate experiments, compare runs, and download artifacts.
7. THE `Usage_Guide` SHALL include a **Runbook** section with step-by-step resolution procedures for at least the following scenarios: (a) drift-triggered retraining fails, (b) model promotion is blocked, (c) JEV gate is blocking all signals.
8. THE `Usage_Guide` SHALL include a **Configuration Reference** table listing all relevant environment variables (`MLFLOW_TRACKING_URI`, `TYPESAFE_API_KEY`, `DRIFT_CHECK_INTERVAL_HOURS`, `MIN_EXECUTE_CONFIDENCE`, `MIN_LIQUIDITY_CONFIDENCE`) with their default values and descriptions.
9. THE `Usage_Guide` SHALL include a **Glossary** section defining all domain terms: JEV_Gate, MLflow, PSI, Certification, Promotion_Gate, Drift_Alert, Model_Registry.
10. WHEN a new model version is promoted to Production, THE `Usage_Guide` SHALL describe the expected change in `/api/model/health` response so operators can verify the promotion succeeded.

---

### Requirement 10: End-to-End MLOps Pipeline Celery Tasks

**User Story:** As an MLOps engineer, I want all long-running MLOps operations (drift checks, training, certification, baseline updates) to run as Celery tasks so they do not block API request handling.

#### Acceptance Criteria

1. THE `Celery_Worker` SHALL include a `run_drift_check_task` task that calls `FeatureDriftMonitor.check()`, persists the report, and conditionally enqueues retraining.
2. THE `Celery_Worker` SHALL include a `update_drift_baseline_task` task that computes a new baseline from the last N days of inference feature data and stores it to disk.
3. WHEN `certify_model_task` produces a `CertificationReport` with `deployment_ready == True`, THE `Celery_Worker` SHALL automatically call the promotion logic to transition the model to Production in the registry.
4. WHEN any MLOps Celery task fails after all retries are exhausted, THE `Celery_Worker` SHALL log the failure at CRITICAL level with the task name, exception message, and task ID.
5. THE `Celery_Worker` SHALL configure Celery Beat with a periodic schedule entry for `run_drift_check_task` with a default interval of 6 hours, configurable via the `DRIFT_CHECK_INTERVAL_HOURS` environment variable.
6. WHEN `train_unified_model_task` completes successfully, THE `Celery_Worker` SHALL automatically enqueue `certify_model_task` to run the K-seed certification on the newly trained model.
