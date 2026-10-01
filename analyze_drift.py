#!/usr/bin/env python3
"""Compute per-feature PSI for given tickers vs training quantiles."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from MLPredictor import UnifiedStockPredictor, CONFIG
import joblib
import numpy as np

TICKERS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS', 'SBIN.NS']

p = UnifiedStockPredictor()
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'unified_models')
quantiles_path = os.path.join(os.path.dirname(__file__), 'unified_models', 'training_quantiles.pkl')

qbins = None
qbins = getattr(p, '_training_quantile_bins', None)
if qbins is None:
    if os.path.exists(quantiles_path):
        qbins = joblib.load(quantiles_path)
    else:
        alt = os.path.join(os.path.dirname(__file__), 'backend', 'unified_models', 'training_quantiles.pkl')
        if os.path.exists(alt):
            qbins = joblib.load(alt)
        else:
            # attempt locating in parent workspaces (common repo root)
            found = False
            cur = os.path.dirname(__file__)
            for _ in range(4):
                cur = os.path.dirname(cur)
                candidate = os.path.join(cur, 'unified_models', 'training_quantiles.pkl')
                if os.path.exists(candidate):
                    qbins = joblib.load(candidate)
                    found = True
                    break
            if not found:
                print('No training quantiles found; looked at:', quantiles_path, 'and', alt)
                sys.exit(1)

feature_cols = getattr(p, 'feature_cols', None)
if not feature_cols:
    # attempt to load saved feature_cols from artifacts
    found = False
    cur = os.path.dirname(__file__)
    for _ in range(4):
        cur = os.path.dirname(cur)
        candidate = os.path.join(cur, 'unified_models', 'feature_cols.pkl')
        if os.path.exists(candidate):
            feature_cols = joblib.load(candidate)
            found = True
            break
    if not found:
        print('Predictor has no feature_cols and none found on disk; abort')
        sys.exit(1)

n_bins = qbins.shape[0] - 1

print('Analyzing drift for tickers:', TICKERS)
for t in TICKERS:
    try:
        df = p._fetch_recent_ticker_data(t, 800)
        if df.empty:
            print(f'{t}: no data')
            continue
        from AdvancedFeatureEngine import AdvancedFeatureEngine
        df_eng = AdvancedFeatureEngine.engineer(df)
        avail = [c for c in feature_cols if c in df_eng.columns]
        if len(avail) < len(feature_cols) * 0.7:
            print(f'{t}: Feature mismatch {len(avail)}/{len(feature_cols)}')
            continue
        seq_len = CONFIG.get('seq_len', 60)
        feat_arr = df_eng[feature_cols].ffill().fillna(0).values[-seq_len:].astype(np.float32)
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
        psi_per = []
        for fi in range(feat_arr.shape[1]):
            edges = qbins[:, fi]
            train_pct = np.ones(n_bins) / n_bins
            hist, _ = np.histogram(feat_arr[:, fi], bins=edges)
            inf_pct = hist / max(feat_arr.shape[0], 1) + 1e-6
            inf_pct = inf_pct / inf_pct.sum()
            train_pct = train_pct + 1e-6
            train_pct = train_pct / train_pct.sum()
            psi = float(np.sum((inf_pct - train_pct) * np.log(inf_pct / train_pct)))
            psi_per.append(psi)
        mean_psi = float(np.mean(psi_per))
        top_idx = sorted(range(len(psi_per)), key=lambda i: psi_per[i], reverse=True)[:10]
        top = [(feature_cols[i], round(psi_per[i],4)) for i in top_idx]
        print(f'{t}: mean_PSI={mean_psi:.3f} | top drift features: {top}')
    except Exception as e:
        print(f'{t}: error {e}')

print('Done')
