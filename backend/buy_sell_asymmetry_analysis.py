"""
Diagnostic script for BUY/SELL asymmetry.
Three hypotheses tested:
  H1: Label asymmetry (bullish moves harder to separate from market beta)
  H2: Focal loss asymmetry (gamma_bull=1.0 vs gamma_bear=2.5 at L1670)
  H3: NSE microstructure (gap_down_rally and gap_up_selloff dominance)
"""
import numpy as np
import pandas as pd
from scipy import stats
import joblib
import logging

logger = logging.getLogger(__name__)


def analyze_label_asymmetry(test_metrics_path: str) -> dict:
    """H1: Are bullish excess returns inherently noisier than bearish?"""
    data = joblib.load(test_metrics_path)
    probs = data.get('test_probs', np.array([]))
    directions = data.get('test_directions', np.array([]))
    returns = data.get('test_returns', np.array([]))
    
    bull_mask = directions == 1
    bear_mask = directions == 0
    
    bull_returns = returns[bull_mask]
    bear_returns = returns[bear_mask]
    
    result = {
        'bull_count': int(bull_mask.sum()),
        'bear_count': int(bear_mask.sum()),
        'bull_return_mean': float(np.mean(bull_returns)),
        'bear_return_mean': float(np.mean(bear_returns)),
        'bull_return_std': float(np.std(bull_returns)),
        'bear_return_std': float(np.std(bear_returns)),
        'bull_return_skew': float(stats.skew(bull_returns)),
        'bear_return_skew': float(stats.skew(bear_returns)),
        # Key metric: is the bull signal-to-noise ratio lower?
        'bull_snr': float(abs(np.mean(bull_returns)) / (np.std(bull_returns) + 1e-8)),
        'bear_snr': float(abs(np.mean(bear_returns)) / (np.std(bear_returns) + 1e-8)),
    }
    
    # Levene's test: are bull/bear return variances significantly different?
    lev_stat, lev_p = stats.levene(bull_returns, bear_returns)
    result['levene_stat'] = float(lev_stat)
    result['levene_p'] = float(lev_p)
    
    logger.info(f"H1 Label asymmetry:")
    logger.info(f"  Bull SNR: {result['bull_snr']:.4f} vs Bear SNR: {result['bear_snr']:.4f}")
    logger.info(f"  Bull std: {result['bull_return_std']:.4f} vs Bear std: {result['bear_return_std']:.4f}")
    logger.info(f"  Levene p: {result['levene_p']:.4f}")
    return result


def analyze_focal_loss_gamma_sweep(
    frozen_cache_path: str,
    gamma_pairs: list = None,
) -> dict:
    """
    H2: Is gamma_bull=1.0 / gamma_bear=2.5 the right ratio?
    Train 3 quick models (10 epochs) with different gamma ratios:
      - Symmetric: gamma_bull=2.0, gamma_bear=2.0
      - Current:   gamma_bull=1.0, gamma_bear=2.5  (from L1670)
      - Inverted:  gamma_bull=2.5, gamma_bear=1.0
    Compare BUY precision and significance across each.
    """
    if gamma_pairs is None:
        gamma_pairs = [
            (2.0, 2.0, "symmetric"),
            (1.0, 2.5, "current"),
            (2.5, 1.0, "inverted"),
            (1.5, 2.0, "mild_bull_boost"),
        ]
    
    results = {}
    # Implementation: modify CONFIG['focal_gamma_bull'] / ['focal_gamma_bear']
    # per run and collect BUY-significance from each short training.
    # See FocalLoss.__init__ at L1670 for where these are consumed.
    # For full analysis, integrate this sweep script into training pipeline.
    
    return results


def analyze_microstructure_asymmetry(feature_importances: dict) -> dict:
    """
    H3: Do gap features create systematic directional bias?
    Check if gap_down_rally (top GBDT feature) correlates with
    SELL predictions more than BUY, creating asymmetric signal strength.
    """
    # gap_down_rally: stock gaps down but rallies intraday -> bullish reversal
    # gap_up_selloff: stock gaps up but sells off -> bearish reversal
    # If these are the top features and bearish reversals are sharper/more
    # reliable than bullish ones, the model will naturally have higher
    # SELL precision.
    
    top_features = sorted(
        feature_importances.items(), key=lambda x: x[1], reverse=True
    )[:20]
    
    # Classify features by directional tendency
    bearish_leaning = ['gap_up_selloff', 'amihud', 'volume_shock']
    bullish_leaning = ['gap_down_rally', 'rsi_9_oversold']
    
    result = {
        'top_20_features': top_features,
        'bearish_feature_total_importance': sum(
            v for k, v in top_features if any(b in k for b in bearish_leaning)
        ),
        'bullish_feature_total_importance': sum(
            v for k, v in top_features if any(b in k for b in bullish_leaning)
        ),
    }
    return result
