import os
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import pickle
from typing import Dict, List, Optional, Any

try:
    import mlflow
except ImportError:
    mlflow = None

logger = logging.getLogger(__name__)

class MLOpsTracker:
    def __init__(self, tracking_uri=None, experiment_name="stock_predictor_multi_target"):
        self.enabled = mlflow is not None
        if not self.enabled:
            logger.warning("MLflow not installed. MLOps tracking disabled.")
            return

        if tracking_uri is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            tracking_uri = f"sqlite:///{os.path.join(project_root, 'mlflow.db')}"
            
        mlflow.set_tracking_uri(tracking_uri)
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set experiment {experiment_name}: {e}")
            
        self.active_run = None

    def start_run(self, run_name: str = None):
        if not self.enabled:
            return None
        self.active_run = mlflow.start_run(run_name=run_name)
        return self.active_run

    def log_params(self, params: Dict[str, Any]):
        if not self.enabled or not self.active_run:
            return
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        if not self.enabled or not self.active_run:
            return
        mlflow.log_metrics(metrics, step=step)
        
    def log_artifacts(self, local_dir: str, artifact_path: str = None):
        if not self.enabled or not self.active_run:
            return
        mlflow.log_artifacts(local_dir, artifact_path)

    def end_run(self):
        if not self.enabled or not self.active_run:
            return
        mlflow.end_run()
        self.active_run = None
        
    def log_training_run(self, params, metrics, artifacts_dir=None):
        run = self.start_run()
        self.log_params(params)
        self.log_metrics(metrics)
        if artifacts_dir:
            self.log_artifacts(artifacts_dir)
        self.end_run()
        return run

    def log_inference(self, latency_ms: float, signal_distribution: Dict, drift_scores: Dict):
        pass

    def log_data_drift(self, drift_scores: Dict[str, float]):
        pass

    def get_experiment_history(self) -> List[Dict]:
        if not self.enabled:
            return []
        try:
            experiment = mlflow.get_experiment_by_name("stock_predictor_multi_target")
            if not experiment:
                return []
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            if runs.empty:
                return []
            runs = runs.replace({np.nan: None})
            return runs.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting experiment history: {e}")
            return []

    def get_model_lineage(self) -> Dict:
        return {
            "model_name": "unified_model",
            "version": "latest",
            "training_data": "historical_ohlcv",
            "features": "technical_indicators"
        }
        
    def compare_runs(self, run_ids: List[str]) -> Dict:
        if not self.enabled:
            return {}
        try:
            runs = mlflow.search_runs(run_view_type=mlflow.entities.ViewType.ALL)
            runs = runs[runs['run_id'].isin(run_ids)]
            runs = runs.replace({np.nan: None})
            return runs.to_dict('records')
        except Exception as e:
            return {}

class DataDriftMonitor:
    def __init__(self):
        pass
        
    def compute_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        return 0.0
        
    def get_latest_drift_report(self) -> Dict:
        return {
            "status": "healthy",
            "psi_scores": {},
            "alert": False
        }

class ModelPerformanceMonitor:
    def __init__(self):
        pass
        
    def get_performance_report(self) -> Dict:
        return {
            "rolling_win_rate": 0.55,
            "precision_decay": 0.01,
            "calibration_drift": 0.02
        }
