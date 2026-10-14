import numpy as np
import pandas as pd
from backend.fracdiff import FractionalDifferentiation

def test_stage1a():
    print("Running Stage 1a Acceptance Tests...")
    
    # 1. FFD with d=1 reproduces first differences exactly
    fd = FractionalDifferentiation(tau=1e-4)
    # Generate some random price data
    np.random.seed(42)
    prices = pd.Series(np.exp(np.random.randn(100).cumsum()))
    
    # We apply log first in the transform usually, but let's test frac_diff_ffd directly
    diff_d1 = fd.frac_diff_ffd(prices, d=1.0)
    actual_diff = prices.diff()
    
    # Align and compare (d=1 requires window width 2 since tau=1e-4 will truncate at k=2 where w_2=0)
    w_d1 = fd.get_weights_ffd(d=1.0, tau=1e-4)
    print(f"Weights for d=1.0, tau=1e-4: {w_d1.flatten()}")
    
    diff_d1 = diff_d1.dropna()
    actual_diff = actual_diff.dropna()
    
    # Compare
    max_error = np.abs(diff_d1 - actual_diff.loc[diff_d1.index]).max()
    print(f"Max error between d=1 FFD and pd.Series.diff(): {max_error}")
    assert max_error < 1e-10, "FFD d=1 failed to reproduce first difference"
    print("OK: FFD with d=1 reproduces first differences exactly.")
    
    # 2. Changing future data leaves past outputs unchanged (causality)
    prices_future = prices.copy()
    prices_future.iloc[-10:] += 100  # Modify future
    
    diff_past = fd.frac_diff_ffd(prices, d=0.5, tau=1e-2)
    diff_future = fd.frac_diff_ffd(prices_future, d=0.5, tau=1e-2)
    
    past_comparison = np.abs(diff_past.iloc[:-10] - diff_future.iloc[:-10]).max()
    print(f"Max difference in past values when future changes: {past_comparison}")
    assert past_comparison < 1e-10, "Causality violated"
    print("OK: Causality confirmed.")
    
    # 3. Report tau vs window-width trade-off
    print("\nTau vs Window-Width Trade-off for d=0.5:")
    taus = [1e-2, 1e-3, 1e-4, 1e-5]
    for t in taus:
        w = fd.get_weights_ffd(0.5, t)
        print(f"tau={t}: window width = {len(w)}")
        
if __name__ == "__main__":
    test_stage1a()
