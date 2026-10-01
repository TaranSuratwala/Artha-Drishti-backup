"""
Seed-ensemble inference: loads K independently-trained models,
runs forward passes in parallel, and averages calibrated probabilities.

DEPRECATION WARNING: This module averages probabilities, which produces scale
mismatch under isotonic calibration. Use `score_universe` in `nse_tasks.py` 
which uses rank-averaging across seeds.
"""
import os
import glob
import torch
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SeedEnsemblePredictor:
    """
    Wraps K independently-trained UnifiedStockPredictor instances.
    At inference, runs all K models and returns the averaged
    calibrated probability.
    """
    
    def __init__(
        self,
        model_dirs: List[str],
        device: str = 'cpu',  # CPU is fine for <5 small models
    ):
        self.device = device
        self.predictors = []
        
        # Lazy import to avoid circular dependency
        from MLPredictor import UnifiedStockPredictor
        
        for model_dir in model_dirs:
            if not os.path.exists(model_dir):
                logger.warning(f"Model dir not found: {model_dir}, skipping")
                continue
            predictor = UnifiedStockPredictor()
            # Override MODEL_DIR for this instance to load the right checkpoint
            predictor._load_from_dir(model_dir)
            if predictor.model is not None:
                predictor.model.eval()
            self.predictors.append(predictor)
        
        if not self.predictors:
            raise RuntimeError("No valid models loaded for ensemble")
        
        logger.info(f"SeedEnsemble loaded {len(self.predictors)} models from {len(model_dirs)} dirs")
    
    def predict(self, ticker: str, capital: float = 100000, risk_pct: float = 2.0) -> Dict:
        """
        Run all K models, average calibrated probabilities,
        then apply signal generation on the averaged probability.
        """
        all_probs = []
        base_result = None
        
        for i, predictor in enumerate(self.predictors):
            try:
                result = predictor.predict(ticker, capital, risk_pct)
                if 'error' in result:
                    logger.warning(f"Seed {i} failed for {ticker}: {result['error']}")
                    continue
                
                prob = result.get('direction_probability', 0.5)
                all_probs.append(prob)
                
                if base_result is None:
                    base_result = result  # Use first successful result as template
                    
            except Exception as e:
                logger.warning(f"Seed {i} exception for {ticker}: {e}")
                continue
        
        if not all_probs or base_result is None:
            return {"error": f"All {len(self.predictors)} ensemble members failed for {ticker}"}
        
        # Average calibrated probabilities
        ensemble_prob = float(np.mean(all_probs))
        ensemble_std = float(np.std(all_probs))
        
        # Override the base result with ensemble probability
        base_result['direction_probability'] = ensemble_prob
        base_result['ensemble_std'] = ensemble_std
        base_result['ensemble_n_models'] = len(all_probs)
        base_result['ensemble_individual_probs'] = all_probs
        
        # Re-derive signal from ensemble probability
        # (uses the same threshold logic as single-model predict)
        base_result['signal'] = self._derive_signal(
            ensemble_prob,
            base_result.get('dynamic_buy_threshold', 0.55),
            base_result.get('dynamic_sell_threshold', -1.0),
        )
        
        # Confidence discount if seeds disagree
        if ensemble_std > 0.05:
            base_result['ensemble_warning'] = (
                f"High seed disagreement (σ={ensemble_std:.3f}). "
                f"Consider reducing position size."
            )
        
        return base_result
    
    def _derive_signal(self, prob: float, buy_thr: float, sell_thr: float) -> str:
        if prob > buy_thr:
            return "BUY"
        elif sell_thr > 0 and prob < sell_thr:
            return "SELL"
        return "HOLD"
