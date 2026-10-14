import numpy as np
import pandas as pd
from backend.cs_transforms import CrossSectionalNormalizer, BetaNeutralizer

def test_stage1c():
    print("Running Stage 1c Acceptance Tests...")
    
    # Generate mock cross-section for a single date
    n = 100
    np.random.seed(42)
    df = pd.DataFrame({
        'ticker': [f'T{i}' for i in range(n)],
        'feature_raw': np.random.normal(0, 10, n),
        'sector': np.random.choice(['IT', 'Bank', 'Pharma', 'Auto', 'SmallSector'], n, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
        'beta': np.random.normal(1.0, 0.2, n),
        'mcap': np.random.uniform(1e8, 1e10, n)
    })
    # Add some outliers
    df.loc[0, 'feature_raw'] = 1000
    
    # 1. Normalisation
    csn = CrossSectionalNormalizer(min_sector_size=8, gaussianize=True)
    res_cs = csn.transform(df, ['feature_raw'], sector_col='sector')
    
    df['feature_cs'] = res_cs['feature_raw_cs']
    
    # The outlier should be winsorized and rank-scaled
    assert not np.isnan(df.loc[0, 'feature_cs'])
    assert df['feature_cs'].max() < 4.0 # Gaussian range
    print("OK: Cross-sectional normalisation applied (MAD winsorization + sector rank-Gauss)")
    
    # 2. Beta Neutralisation
    bn = BetaNeutralizer()
    df['feature_neu'] = bn.neutralise(df, 'feature_cs', 'beta', 'mcap', 'sector')
    
    # Verify X' * resid = 0
    df_clean = df.dropna(subset=['feature_neu'])
    y = df_clean['feature_neu'].values
    X_beta = df_clean['beta'].values
    X_mcap = np.log(df_clean['mcap'].values)
    
    dot_beta = np.dot(X_beta, y)
    dot_mcap = np.dot(X_mcap, y)
    
    print(f"X' * resid for beta: {dot_beta}")
    print(f"X' * resid for log_mcap: {dot_mcap}")
    
    assert np.abs(dot_beta) < 1e-10, "Beta feature not orthogonalized"
    assert np.abs(dot_mcap) < 1e-10, "Mcap feature not orthogonalized"
    print("OK: Features are strictly orthogonalized (X' * resid ~= 0).")
    
if __name__ == "__main__":
    test_stage1c()
