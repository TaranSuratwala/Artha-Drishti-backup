import numpy as np
import pandas as pd
from backend.orthogonalization import CollinearityController

def test_stage1d():
    print("Running Stage 1d Acceptance Tests...")
    
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    # x2 is highly correlated with x1
    x2 = x1 + np.random.normal(0, 0.1, n)
    x3 = np.random.normal(0, 1, n)
    
    df = pd.DataFrame({'f1': x1, 'f2': x2, 'f3': x3})
    
    cc = CollinearityController(method='clustering', threshold=0.4) # rho ~ 0.68
    
    # Calculate initial VIF
    vif_before = cc._compute_vif(df)
    print("VIF before:")
    print(vif_before)
    
    cc.fit(df, priority_order=['f1', 'f2', 'f3'])
    df_transformed = cc.transform(df)
    
    vif_after = cc._compute_vif(df_transformed)
    print("\nSelected features:", cc.selected_features)
    print("VIF after:")
    print(vif_after)
    
    assert 'f2' not in df_transformed.columns, "Highly correlated feature not dropped"
    print("OK: Collinearity controlled via Hierarchical Clustering.")
    
    # Gram-Schmidt
    cc_gs = CollinearityController(method='gram-schmidt')
    cc_gs.fit(df, priority_order=['f1', 'f2', 'f3'])
    df_gs = cc_gs.transform(df)
    
    vif_gs = cc_gs._compute_vif(df_gs)
    print("\nVIF after Gram-Schmidt:")
    print(vif_gs)
    assert vif_gs.max() < 1.1, "Features not orthogonalized"
    print("OK: Features orthogonalized via Sequential Gram-Schmidt.")

if __name__ == "__main__":
    test_stage1d()
