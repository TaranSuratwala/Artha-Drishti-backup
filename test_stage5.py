import numpy as np
import pandas as pd
from backend.frictions import ExecutionFrictions

def test_stage5():
    print("Running Stage 5 Acceptance Tests...")
    
    frictions = ExecutionFrictions()
    
    trade_val = 100000.0 # 1 lakh
    
    buy_cost = frictions.estimate_costs(trade_val, is_buy=True)
    sell_cost = frictions.estimate_costs(trade_val, is_buy=False)
    
    print(f"Buy Cost for 1L: {buy_cost:.2f}")
    print(f"Sell Cost for 1L: {sell_cost:.2f}")
    
    # Verify against Zerodha calculator approx for 1L delivery:
    # Buy: STT 100, Txn 3.22, GST 0.58, SEBI 0.1, Stamp 15 => ~118.9
    # Sell: STT 100, Txn 3.22, GST 0.58, SEBI 0.1, DP 15.93 => ~119.83
    
    assert np.abs(buy_cost - 118.9) < 2.0
    assert np.abs(sell_cost - 119.83) < 2.0
    
    print("OK: Cost model reproduces hand-computed INR amount for a sample round trip.")

if __name__ == "__main__":
    test_stage5()
