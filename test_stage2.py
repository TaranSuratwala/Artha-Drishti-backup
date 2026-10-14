import numpy as np
import pandas as pd
from backend.labeling import TripleBarrierLabeler
import time

def brute_force_triple_barrier(df, k, pt, sl, m):
    n = len(df)
    labels = np.full(n, 0)
    returns = np.full(n, np.nan)
    t1 = np.full(n, pd.NaT)
    
    for i in range(n):
        if i + 1 >= n:
            break
            
        entry = df['open'].iloc[i+1]
        sigma = df['volatility'].iloc[i]
        
        if pd.isna(entry) or pd.isna(sigma):
            continue
            
        width = m * sigma * np.sqrt(k)
        u_b = entry * np.exp(pt * width)
        l_b = entry * np.exp(-sl * width)
        
        touched = False
        for j in range(i+1, min(i+1+k, n)):
            high = df['high'].iloc[j]
            low = df['low'].iloc[j]
            open_p = df['open'].iloc[j]
            
            # Check gap through barrier at open
            is_up = high >= u_b
            is_down = low <= l_b
            
            if is_up and is_down:
                labels[i] = -1
                fill = min(open_p, l_b)
                returns[i] = np.log(fill / entry)
                t1[i] = df.index[j]
                touched = True
                break
            elif is_up:
                labels[i] = 1
                fill = max(open_p, u_b)
                returns[i] = np.log(fill / entry)
                t1[i] = df.index[j]
                touched = True
                break
            elif is_down:
                labels[i] = -1
                fill = min(open_p, l_b)
                returns[i] = np.log(fill / entry)
                t1[i] = df.index[j]
                touched = True
                break
                
        if not touched and i+k < n:
            labels[i] = 0
            returns[i] = np.log(df['close'].iloc[i+k] / entry)
            t1[i] = df.index[i+k]
            
    return labels, returns, t1


def test_stage2():
    print("Running Stage 2 Acceptance Tests...")
    
    n = 1000
    np.random.seed(42)
    dates = pd.date_range('2010-01-01', periods=n, freq='B')
    
    # Generate OHLC
    close = np.exp(np.random.randn(n) * 0.02).cumprod() * 100
    df = pd.DataFrame(index=dates)
    df['open'] = close * np.exp(np.random.randn(n) * 0.005)
    df['close'] = close
    df['high'] = np.maximum(df['open'], df['close']) * np.exp(np.abs(np.random.randn(n)) * 0.01)
    df['low'] = np.minimum(df['open'], df['close']) * np.exp(-np.abs(np.random.randn(n)) * 0.01)
    df['volatility'] = df['close'].pct_change().rolling(20).std().fillna(0.02)
    
    labeler = TripleBarrierLabeler(pt=1.0, sl=1.0, m=1.0)
    
    for k in [3, 5, 7, 15, 21]:
        print(f"Testing horizon k={k}...")
        
        # Vectorized
        t0 = time.time()
        res_vec = labeler.get_events(df, k)
        t_vec = time.time() - t0
        
        # Brute force
        t0 = time.time()
        labels_bf, returns_bf, t1_bf = brute_force_triple_barrier(df, k, 1.0, 1.0, 1.0)
        t_bf = time.time() - t0
        
        # Compare (ignoring trailing NaNs)
        valid = ~np.isnan(returns_bf)
        
        # There might be small float rounding diffs so use np.allclose
        ret_diff = np.abs(res_vec[f'ret_{k}'].values[valid] - returns_bf[valid]).max()
        assert ret_diff < 1e-8, f"Returns mismatch for k={k}: {ret_diff}"
        
        assert (res_vec[f'label_{k}'].values[valid] == labels_bf[valid]).all(), f"Labels mismatch for k={k}"
        
        t1_vec_valid = pd.Series(res_vec[f't1_{k}'].values[valid])
        t1_bf_valid = pd.Series(t1_bf[valid])
        assert (t1_vec_valid == t1_bf_valid).all(), f"t1 mismatch for k={k}"
        
        print(f"  OK: Vectorised equals brute-force loop. (Vec {t_vec:.4f}s vs Loop {t_bf:.4f}s)")

if __name__ == "__main__":
    test_stage2()
