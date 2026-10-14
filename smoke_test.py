import numpy as np
import pandas as pd
from backend.fracdiff import FractionalDifferentiation
from backend.nse_features import MicrostructureFeatures
from backend.cs_transforms import CrossSectionalNormalizer, BetaNeutralizer
from backend.labeling import TripleBarrierLabeler
from backend.purged_cv import CombinatorialPurgedCV
from backend.frictions import ExecutionFrictions
from backend.multi_horizon_models import LightGBMBaseline
import time

def run_smoke_test():
    print("--- Artha-Drishti Pipeline End-to-End Smoke Test ---")
    
    # 1. Generate synthetic dataset
    print("\nGenerating synthetic panel data (5 tickers, 300 days)...")
    np.random.seed(42)
    tickers = [f'T{i}' for i in range(5)]
    dates = pd.date_range('2020-01-01', periods=300, freq='B')
    
    data = []
    for t in tickers:
        close = np.exp(np.random.randn(300) * 0.02).cumprod() * 100
        vol = np.random.randint(1000, 10000, 300)
        df = pd.DataFrame({
            'date': dates,
            'ticker': t,
            'open': close * np.exp(np.random.randn(300) * 0.005),
            'close': close,
            'high': close * 1.02,
            'low': close * 0.98,
            'volume': vol,
            'delivery_percentage': np.random.uniform(0.1, 0.8, 300),
            'delivery_qty': vol * np.random.uniform(0.1, 0.8, 300),
            'sector': np.random.choice(['IT', 'Bank', 'Auto']),
            'beta': np.random.normal(1.0, 0.2, 300),
            'mcap': np.random.uniform(1e8, 1e10, 300),
            'returns': np.random.normal(0, 0.02, 300),
            'volatility': np.random.uniform(0.01, 0.03, 300)
        })
        data.append(df)
        
    panel = pd.concat(data).set_index('date')
    
    # STAGE 1a
    print("Running FracDiff (Stage 1a)...")
    fd = FractionalDifferentiation()
    fd.fit(panel, 'volume', 'ticker')
    panel['volume_fd'] = fd.transform(panel, 'volume', 'ticker')
    
    # STAGE 1b
    print("Running Microstructure (Stage 1b)...")
    mf = MicrostructureFeatures()
    deliv_feats = mf.compute_delivery_features(panel)
    panel = pd.concat([panel, deliv_feats], axis=1)
    
    # STAGE 1c
    print("Running Cross-Sectional Normalisation (Stage 1c)...")
    # CS normalisation needs to group by date
    panel = panel.reset_index()
    cs_results = []
    csn = CrossSectionalNormalizer(min_sector_size=1)
    for dt, group in panel.groupby('date'):
        cs_res = csn.transform(group, ['volume_fd', 'delivery_pct_zscore'])
        cs_res['date'] = group['date']
        cs_res['ticker'] = group['ticker']
        cs_results.append(cs_res)
    cs_df = pd.concat(cs_results)
    
    panel = pd.merge(panel, cs_df[['date', 'ticker', 'volume_fd_cs', 'delivery_pct_zscore_cs']], on=['date', 'ticker'])
    panel = panel.set_index('date')
    
    # STAGE 2
    print("Running Labeling (Stage 2)...")
    labeler = TripleBarrierLabeler(k_horizons=[5])
    # Compute per ticker
    events_list = []
    weights_list = []
    for t, group in panel.groupby('ticker'):
        events = labeler.get_events(group, k=5)
        events['ticker'] = t
        events_list.append(events)
        
        weights = labeler.compute_sample_weights(
            pd.Series(group.index, index=group.index),
            events['t1_5']
        )
        weights.name = 'weight_5'
        weights_list.append(weights)
        
    labels_df = pd.concat(events_list)
    weights_df = pd.concat(weights_list)
    
    panel = pd.concat([panel, labels_df[['ret_5', 'label_5', 't1_5']], weights_df], axis=1)
    
    # Drop Nans for modeling
    model_df = panel.dropna(subset=['volume_fd_cs', 'ret_5'])
    model_df = model_df.reset_index()
    
    # STAGE 3
    print("Running Purged CV (Stage 3)...")
    cpcv = CombinatorialPurgedCV(n_groups=3, k_test_groups=1, embargo_tdays=5)
    
    # rename date to t0
    model_df = model_df.rename(columns={'date': 't0', 't1_5': 't1'})
    splits = list(cpcv.split(model_df, 't0', 't1'))
    print(f"Generated {len(splits)} CV splits.")
    
    # STAGE 4
    print("Running Baseline Model (Stage 4)...")
    X = model_df[['volume_fd_cs', 'delivery_pct_zscore']]
    y_dict = {5: model_df['ret_5']}
    w_dict = {5: model_df['weight_5']}
    
    model = LightGBMBaseline(horizons=[5])
    model.fit(X, y_dict, w_dict)
    
    preds = model.predict(X, 5)
    model_df['pred_5'] = preds
    ic = model_df['pred_5'].corr(model_df['ret_5'])
    print(f"In-sample IC: {ic:.4f}")
    
    # STAGE 5
    print("Running Frictions (Stage 5)...")
    fric = ExecutionFrictions()
    cost = fric.estimate_costs(100000, True)
    print(f"Sample execution cost (1L Buy): {cost:.2f} INR")
    
    print("\n--- Smoke Test Completed Successfully! ---")

if __name__ == "__main__":
    run_smoke_test()
