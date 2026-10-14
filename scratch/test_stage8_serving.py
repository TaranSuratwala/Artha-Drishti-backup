"""
Acceptance Test Suite for Stage 8: Production Serving, Parity & Shadow Trading
==============================================================================
Verifies:
1. Train/Serve Parity Test (Golden Sample) with floating-point tolerance <= 1e-5.
2. Deliberate skew detection: catches corrupted/shifted features at serving time.
3. Daily Whole-Universe Scoring with true same-day ranks and hysteresis selection.
4. Shadow Trade Protocol: Close(t) order generation -> Open(t+1) execution with frictions.
5. Rolling Live-IC Monitor: tracks forward accuracy and alerts on decay.
6. Automated Kill Switch: halts trading after 5 consecutive negative rolling-20 IC sessions.
7. Trading halt enforcement: blocks new orders during kill switch activation.
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

from serving import (
    ServingConfig, TrainServeParityValidator, TrainServeSkewError,
    LiveICMonitor, LiveScoringService, ShadowTradeProtocol,
    TradingHaltedError, OrderState
)
from multi_horizon_models import LightGBMMultiHorizon


def test_train_serve_parity_and_planted_skew():
    """Test 1: Train/Serve Parity Validation on Golden Sample & Planted Skew Detection."""
    print("\n" + "="*80)
    print("TEST 1: Train / Serve Parity Validation & Planted Skew Detection")
    print("="*80)

    validator = TrainServeParityValidator(tolerance=1e-5)

    # 1. Train a reference model on offline data
    np.random.seed(42)
    n_samples = 300
    n_feats = 6
    feat_cols = [f"f_{j}" for j in range(n_feats)]

    X_offline = pd.DataFrame(np.random.normal(0, 1.0, (n_samples, n_feats)), columns=feat_cols)
    y_offline = X_offline["f_0"] * 0.4 - X_offline["f_1"] * 0.3 + np.random.normal(0, 0.1, n_samples)

    model = LightGBMMultiHorizon(horizons=[5], mode="regression", learning_rate=0.05, num_boost_round=40)
    model.fit(X_offline, {5: y_offline})

    # 2. Golden Sample (inputs and verified training predictions)
    golden_inputs = X_offline.iloc[:50].copy()
    golden_expected = model.predict(golden_inputs, k=5)

    # Define live serving scoring function
    def serving_scoring_fn(df_inputs):
        return model.predict(df_inputs[feat_cols], k=5)

    # Parity check on pristine golden sample
    parity_passed = validator.assert_parity(golden_inputs, golden_expected, serving_scoring_fn)
    assert parity_passed, "Pristine golden sample failed parity check!"
    print(">>> Pristine Train/Serve Golden Sample passed with parity tolerance <= 1e-5.")

    # 3. Deliberate Skew: corrupt feature column 'f_0' by adding small offset (silent serving skew)
    corrupted_inputs = golden_inputs.copy()
    corrupted_inputs["f_0"] += 0.05  # 5 bps feature shift

    skew_caught = False
    try:
        validator.assert_parity(corrupted_inputs, golden_expected, serving_scoring_fn)
    except TrainServeSkewError as e:
        skew_caught = True
        print(f"Planted Feature Skew Caught Successfully: {e}")

    assert skew_caught, "Train/Serve validator failed to catch planted feature skew!"
    print(">>> Planted Train/Serve Skew flagged and rejected.")


def test_daily_whole_universe_scoring():
    """Test 2: Daily Whole-Universe Scoring with True Same-Day Ranks."""
    print("\n" + "="*80)
    print("TEST 2: Daily Whole-Universe Scoring & True Same-Day Ranks")
    print("="*80)

    config = ServingConfig(n_enter=15, n_exit=25, min_adtv_inr=50_000_000.0)
    service = LiveScoringService(config)

    # Create synthetic whole-universe snapshot on date t
    np.random.seed(101)
    n_stocks = 80
    symbols = [f"NSE_{i:03d}" for i in range(n_stocks)]
    asof_date = pd.Timestamp("2026-10-08").date()

    # Mixed universe: some liquid, some illiquid (< ₹5 Cr), some circuit locked
    records = []
    for idx, sym in enumerate(symbols):
        is_illiquid = (idx < 10)  # first 10 names fail ADTV
        is_circuit = (idx == 15 or idx == 16) # circuit locked
        adtv = 20_000_000.0 if is_illiquid else 75_000_000.0
        is_eligible = not (is_illiquid or is_circuit)

        records.append({
            "date": asof_date,
            "symbol": sym,
            "close": 150.0 + idx * 5.0,
            "open": 149.0 + idx * 5.0,
            "adtv_20": adtv,
            "vol_20": 0.02,
            "is_eligible": is_eligible,
            "alpha_signal": np.random.normal(0, 1.0) + (0.5 if idx > 60 else -0.5),
        })

    snapshot_df = pd.DataFrame(records)

    # Scoring function returning predicted score
    def mock_model_predict(df):
        return df["alpha_signal"].values

    ranked_df = service.score_universe(snapshot_df, asof_date, mock_model_predict)

    print(f"Total Universe Names:          {len(snapshot_df)}")
    print(f"Eligible Screened Names:       {len(ranked_df)}")
    print(f"Selected Target Names:         {ranked_df['is_selected'].sum()} (Target: {config.n_enter})")
    print(f"Best Ranked Stock:             {ranked_df.iloc[0]['symbol']} (Score={ranked_df.iloc[0]['score']:.4f}, Rank={ranked_df.iloc[0]['rank']})")
    print(f"Worst Ranked Stock:            {ranked_df.iloc[-1]['symbol']} (Score={ranked_df.iloc[-1]['score']:.4f}, Rank={ranked_df.iloc[-1]['rank']})")

    # Verify ranking monotonicity and continuity
    assert len(ranked_df) == 68, f"Screened names {len(ranked_df)} mismatch"
    assert ranked_df['rank'].min() == 1
    assert ranked_df['rank'].max() == len(ranked_df)
    assert ranked_df['is_selected'].sum() == config.n_enter
    assert (ranked_df.head(config.n_enter)['is_selected']).all(), "Top names should be selected on inception"

    print(">>> Daily whole-universe scoring produced true same-day ranks and valid target selection.")
    return ranked_df


def test_shadow_trading_execution(ranked_df: pd.DataFrame):
    """Test 3: Shadow Trading Execution Protocol."""
    print("\n" + "="*80)
    print("TEST 3: Shadow Trading Protocol (Close(t) -> Open(t+1) Execution)")
    print("="*80)

    config = ServingConfig(n_enter=15, n_exit=25, initial_shadow_aum=10_000_000.0)
    protocol = ShadowTradeProtocol(config)

    signal_date = pd.Timestamp("2026-10-08").date()
    exec_date = pd.Timestamp("2026-10-09").date()

    # 1. Generate prospective orders at Close(t)
    orders = protocol.generate_orders_from_scores(ranked_df, signal_date, exec_date)
    print(f"Generated Prospective Orders:  {len(orders)} at Close({signal_date})")
    assert len(orders) == config.n_enter, f"Expected {config.n_enter} entry orders, got {len(orders)}"

    for o in orders[:3]:
        print(f"  Order {o.order_id}: {o.side} {o.symbol} | Val=₹{o.order_value_inr:,.2f} | Est Friction=₹{o.total_friction_inr:.2f} ({o.state.value})")

    # 2. Simulate execution at Open(t+1) with market quotes
    np.random.seed(888)
    open_prices = {}
    for sym in ranked_df["symbol"]:
        ref_px = ranked_df.loc[ranked_df["symbol"] == sym, "close"].values[0]
        # Open price with small gap
        open_prices[sym] = ref_px * (1.0 + np.random.normal(0, 0.003))

    fill_result = protocol.execute_shadow_fills(orders, open_prices)
    print(f"\nExecution Fills Summary at Open({exec_date}):")
    print(f"  Filled Orders:               {fill_result['filled_orders']} / {fill_result['total_orders']}")
    print(f"  Total Frictions Paid:        ₹{fill_result['friction_paid_inr']:,.2f}")
    print(f"  Updated Shadow NAV:          ₹{fill_result['updated_nav']:,.2f}")
    print(f"  Active Positions:            {fill_result['active_positions_count']}")

    assert fill_result["filled_orders"] == config.n_enter
    assert fill_result["updated_nav"] < config.initial_shadow_aum  # Frictions deducted
    print(">>> Shadow execution completed with fill prices, impact, and statutory fees logged.")
    return protocol


def test_live_ic_monitor_and_kill_switch(protocol: ShadowTradeProtocol, ranked_df: pd.DataFrame):
    """Test 4 & 5: Live-IC Monitor, Decay Alert & Automated Kill Switch."""
    print("\n" + "="*80)
    print("TEST 4 & 5: Live-IC Monitor, Retrain Alert & Automated Kill Switch")
    print("="*80)

    monitor = protocol.ic_monitor

    # 1. Normal healthy trading sessions
    print("Simulating 15 normal sessions with positive Rank IC...")
    for d_idx in range(15):
        t_date = pd.Timestamp("2026-09-01") + pd.Timedelta(days=d_idx)
        res = monitor.record_realized_ic(t_date.date(), rank_ic=0.065 + np.random.normal(0, 0.02))

    print(f"  Session 15 Rolling-20 IC:     {res['rolling_20_ic']:.4f}")
    print(f"  Kill Switch Active:          {monitor.is_killed}")
    assert not monitor.is_killed, "Kill switch should not be active during healthy IC"

    # 2. Simulate performance decay (< 50% of backtest IC 0.05 -> < 0.025) for 35 sessions
    print("\nSimulating 35 sessions with decayed Rank IC (+0.010)...")
    for d_idx in range(15, 50):
        t_date = pd.Timestamp("2026-09-01") + pd.Timedelta(days=d_idx)
        res = monitor.record_realized_ic(t_date.date(), rank_ic=0.010)

    print(f"  Session 50 Rolling-20 IC:     {res['rolling_20_ic']:.4f}")
    print(f"  Retrain Recommended Flag:    {monitor.retrain_recommended}")
    assert monitor.retrain_recommended, "Retrain alert failed to fire on persistent decay"
    print(">>> Retraining alert triggered after 10 sessions of decayed IC.")

    # 3. Simulate severe failure (negative rolling IC <= 0 for 5 consecutive sessions)
    print("\nSimulating 30 severe negative sessions (-0.120)...")
    for d_idx in range(50, 80):
        t_date = pd.Timestamp("2026-09-01") + pd.Timedelta(days=d_idx)
        res = monitor.record_realized_ic(t_date.date(), rank_ic=-0.120)

    print(f"  Session 80 Rolling-20 IC:     {res['rolling_20_ic']:.4f}")
    print(f"  Kill Switch Active:          {monitor.is_killed}")
    print(f"  Kill Switch Reason:          {monitor.kill_reason}")
    assert monitor.is_killed, "Kill switch failed to trigger on negative IC!"
    print(">>> Automated Kill Switch successfully triggered.")

    # 4. Enforce Trading Halt: attempt to generate orders while killed must raise TradingHaltedError
    order_blocked = False
    try:
        protocol.generate_orders_from_scores(ranked_df, "2026-10-10", "2026-10-11")
    except TradingHaltedError as e:
        order_blocked = True
        print(f"\nOrder Generation Blocked by Safety Gate: {e}")

    assert order_blocked, "Safety gate failed to block orders during kill switch activation!"
    print(">>> Trading halt strictly enforced; zero orders generated while killed.")

    # 5. Reset kill switch
    monitor.reset_kill_switch()
    assert not monitor.is_killed
    print(">>> Kill switch reset verified.")


if __name__ == "__main__":
    test_train_serve_parity_and_planted_skew()
    df_ranked = test_daily_whole_universe_scoring()
    prot = test_shadow_trading_execution(df_ranked)
    test_live_ic_monitor_and_kill_switch(prot, df_ranked)
    print("\n" + "="*80)
    print("ALL STAGE 8 ACCEPTANCE TESTS PASSED SUCCESSFULLY.")
    print("="*80)
