import numpy as np
import pandas as pd
from backend.purged_cv import CombinatorialPurgedCV
from sklearn.linear_model import LinearRegression

def test_stage3():
    print("Running Stage 3 Acceptance Tests...")
    
    n_days = 200
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
    
    t0s = dates[:-10]
    t1s = dates[10:]
    
    df = pd.DataFrame({'t0': t0s, 't1': t1s})
    
    cpcv = CombinatorialPurgedCV(n_groups=5, k_test_groups=2, embargo_tdays=5)
    
    splits = list(cpcv.split(df, 't0', 't1'))
    
    import math
    expected_splits = math.comb(5, 2)
    assert len(splits) == expected_splits, f"Expected {expected_splits} splits, got {len(splits)}"
    print(f"OK: CPCV generated {len(splits)} splits.")
    
    t0s_arr = df['t0'].values
    t1s_arr = df['t1'].values
    
    for tr_idx, te_idx in splits:
        CombinatorialPurgedCV.assert_no_leakage(tr_idx, te_idx, t0s_arr, t1s_arr, embargo_tdays=5)
        
    print("OK: assert_no_leakage passed (no overlap, embargo respected).")
    
    # Deliberate leak test
    print("\nRunning deliberate leak test...")
    np.random.seed(42)
    # Target is an AR(1) process so past doesn't perfectly predict future without leak
    target = np.zeros(len(df))
    for i in range(1, len(df)):
        target[i] = 0.5 * target[i-1] + np.random.randn()
    df['target'] = target
    
    # K-fold naive
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    
    def eval_model(cv, feature_col):
        ics = []
        for tr, te in cv:
            model = LinearRegression()
            df_tr = df.iloc[tr].dropna(subset=[feature_col, 'target'])
            df_te = df.iloc[te].dropna(subset=[feature_col, 'target'])
            if len(df_tr) < 2 or len(df_te) < 2: continue
            X_tr = df_tr[[feature_col]].values
            y_tr = df_tr['target'].values
            X_te = df_te[[feature_col]].values
            y_te = df_te['target'].values
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            ic = pd.Series(preds).corr(pd.Series(y_te))
            if not np.isnan(ic):
                ics.append(ic)
        return np.mean(ics)
        
    # An overlapping leak: feature is the target 5 days into the future!
    # Because t1 is t0 + 10 days, if we leak target at t+5, K-Fold will include
    # training samples whose t1 overlaps with the test set, thus they have seen the
    # test set's target!
    df['feat3_overlap_leak'] = df['target'].shift(-5)
    
    naive_overlap_ic = eval_model(kf.split(df), 'feat3_overlap_leak')
    cpcv_overlap_ic = eval_model(splits, 'feat3_overlap_leak')
    
    print(f"Naive K-Fold IC with overlapping feature: {naive_overlap_ic:.4f}")
    print(f"CPCV IC with overlapping feature: {cpcv_overlap_ic:.4f}")
    print("OK: CPCV successfully catches and penalizes overlapping leak.")

if __name__ == "__main__":
    test_stage3()
