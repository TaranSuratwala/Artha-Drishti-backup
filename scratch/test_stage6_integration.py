"""
Integration Test: Stage 5 Multi-Horizon Models -> Stage 6 Portfolio Execution
=============================================================================
Verifies end-to-end integration:
1. Features & Targets (Stage 2 & 3)
2. LightGBM / MultiTaskNN model scores (Stage 5)
3. Rank-based Top-N Hysteresis Portfolio Selection & Simulation (Stage 6)
4. Net-of-frictions performance vs Equal-Weight Eligible Universe benchmark.
"""

import sys
import os
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from frictions import TradingSegment, NSECostConfig, ImpactModelConfig
from portfolio import PortfolioConfig, PortfolioSimulator, CapacityEstimator
from multi_horizon_models import LightGBMMultiHorizon


def test_stage5_stage6_pipeline():
    print("\n" + "="*80)
    print("INTEGRATION TEST: Stage 5 Multi-Horizon Model -> Stage 6 Execution")
    print("="*80)

    np.random.seed(42)
    n_days = 90
    n_stocks = 60
    symbols = [f"NSE_EQ_{i:02d}" for i in range(n_stocks)]
    dates = pd.date_range("2026-01-01", periods=n_days, freq="B")

    # Features and price dynamics
    feature_names = [f"feat_{j}" for j in range(8)]
    records = []

    # Latent true factor loadings
    true_weights = np.array([0.4, -0.3, 0.5, -0.2, 0.0, 0.1, -0.1, 0.25])
    stock_betas = np.linspace(0.8, 1.2, n_stocks)

    for s_idx, sym in enumerate(symbols):
        p_close = 200.0 + s_idx * 5.0
        for d in dates:
            feats = np.random.normal(0, 1.0, len(feature_names))
            latent_alpha = np.dot(feats, true_weights) * 0.005  # daily alpha component
            market_ret = np.random.normal(0.0003, 0.010)
            idio_ret = np.random.normal(0, 0.012)
            daily_ret = stock_betas[s_idx] * market_ret + latent_alpha + idio_ret

            p_open = p_close * (1.0 + np.random.normal(0, 0.002))
            p_close = p_open * (1.0 + daily_ret)

            row = {
                "date": d,
                "symbol": sym,
                "open": p_open,
                "close": p_close,
                "adtv_20": 80_000_000.0,  # ₹8 Crore
                "vol_20": 0.021,
                "is_eligible": True,
            }
            for fn, fv in zip(feature_names, feats):
                row[fn] = fv
            # Target 5d return
            row["target_5d"] = latent_alpha * 5.0 + np.random.normal(0, 0.02)
            records.append(row)

    panel_df = pd.DataFrame(records).sort_values(by=["date", "symbol"]).reset_index(drop=True)

    # Train / Test split
    split_date = dates[45]
    train_mask = panel_df["date"] < split_date
    test_mask = panel_df["date"] >= split_date

    train_df = panel_df[train_mask].copy()
    test_df = panel_df[test_mask].copy()

    # Train LightGBM model for k=5
    model = LightGBMMultiHorizon(horizons=[5], mode="regression", learning_rate=0.05, num_boost_round=60)

    targets_dict = {5: train_df["target_5d"]}
    model.fit(train_df[feature_names], targets_dict, dates_train=train_df["date"])

    # Predict scores out-of-sample on test_df
    preds_5 = model.predict(test_df[feature_names], k=5)
    test_df["predicted_score"] = preds_5

    # Portfolio simulation on test window
    port_config = PortfolioConfig(
        n_enter=15,
        n_exit=25,
        rebalance_freq=5,
        initial_aum=10_000_000.0,
    )
    simulator = PortfolioSimulator(port_config)
    res = simulator.run_simulation(test_df, score_col="predicted_score", use_hysteresis=True)

    print(f"Test Window:                   {test_df['date'].min().strftime('%Y-%m-%d')} -> {test_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Gross Portfolio CAGR:          {res['cagr_gross']*100:.2f}%")
    print(f"Net Portfolio CAGR:            {res['cagr_net']*100:.2f}%")
    print(f"Equal-Weight Benchmark CAGR:   {res['cagr_bmk']*100:.2f}%")
    print(f"Net Active Alpha:              {res['net_alpha_cagr']*100:.2f}%")
    print(f"Net Sharpe Ratio:              {res['net_sharpe']:.2f}")
    print(f"Information Ratio:             {res['information_ratio']:.2f}")
    print(f"Max Drawdown (Net):            {res['max_drawdown']*100:.2f}%")
    print(f"Statutory Costs Paid:          ₹{res['total_statutory_costs_inr']:,.2f}")
    print(f"Impact Costs Paid:             ₹{res['total_impact_costs_inr']:,.2f}")

    assert res["cagr_gross"] > res["cagr_bmk"], "Model did not beat benchmark grossly"
    print(">>> Stage 5 -> Stage 6 Pipeline Integration Test Passed Successfully.")


if __name__ == "__main__":
    test_stage5_stage6_pipeline()
