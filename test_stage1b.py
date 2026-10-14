import numpy as np
import pandas as pd
from backend.nse_features import MicrostructureFeatures, VIXRegimeModel

def test_stage1b():
    print("Running Stage 1b Acceptance Tests...")
    
    mf = MicrostructureFeatures()
    
    # Create synthetic data
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, n),
        'high': np.random.uniform(110, 120, n),
        'low': np.random.uniform(90, 100, n),
        'close': np.random.uniform(100, 110, n),
        'volume': np.random.randint(1000, 10000, n),
        'delivery_percentage': np.random.uniform(0.1, 0.8, n),
        'delivery_qty': np.random.randint(500, 5000, n),
        'turnover_lacs': np.random.uniform(10, 100, n),
        'no_of_trades': np.random.randint(100, 1000, n),
        'returns': np.random.normal(0, 0.02, n),
        'open_interest': np.random.randint(10000, 50000, n)
    })
    
    # Delivery features
    deliv = mf.compute_delivery_features(df)
    assert 'delivery_pct_zscore' in deliv.columns
    assert 'deliv_qty_surge' in deliv.columns
    assert 'avg_trade_size' in deliv.columns
    assert 'signed_delivery_flow' in deliv.columns
    
    # Gap features
    gap = mf.compute_gap_features(df)
    assert 'gap_zscore' in gap.columns
    
    # Liquidity features
    liq = mf.compute_liquidity_features(df)
    assert 'amihud' in liq.columns
    assert 'corwin_schultz' in liq.columns
    
    # Volatility features
    vol = mf.compute_volatility_features(df)
    assert 'parkinson_20' in vol.columns
    assert 'garman_klass_20' in vol.columns
    assert 'vol_term_structure' in vol.columns
    assert 'vol_of_vol' in vol.columns
    
    # VIX Regime
    vix = pd.Series(np.concatenate([np.random.normal(15, 2, 50), np.random.normal(25, 4, 50)]))
    model = VIXRegimeModel()
    model.fit(vix[:75]) # fit on train
    probas = model.predict_regime_proba(vix)
    assert len(probas) == len(vix)
    assert not probas.isna().all()
    
    print("OK: Stage 1b tests passed. NSE microstructure features and causal VIX regime generated successfully.")

if __name__ == "__main__":
    test_stage1b()
