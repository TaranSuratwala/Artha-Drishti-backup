"""
Paper Trade Analyzer v2 — Matches the fixed pipeline's composite confidence schema.
Validates that the confidence fix resolves the chronic low-confidence problem.
"""
import pandas as pd
import numpy as np
import os
import sys

def analyze_paper_trades(log_file: str = "paper_trades_log.csv") -> None:
    if not os.path.exists(log_file):
        print(f"Log file {log_file} does not exist yet. Run the pipeline first.")
        return

    df = pd.read_csv(log_file)
    if df.empty:
        print("No trades recorded yet.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print("=" * 60)
    print("  Paper Trading Performance Tear Sheet (v2)")
    print("=" * 60)

    total_trades = len(df)
    actual_trades = df[df['action_label'] != 'SQUARE_OFF']
    square_offs = df[df['action_label'] == 'SQUARE_OFF']

    print(f"\nTotal Orders Executed:     {total_trades}")
    print(f"  Directional Trades:      {len(actual_trades)}")
    print(f"  Square-Off Orders:       {len(square_offs)}")
    print(f"Unique Symbols Traded:     {df['symbol'].nunique()}")

    print("\n--- CONFIDENCE DIAGNOSTICS (v2 Fix Validation) ---")

    if 'composite_confidence' in df.columns:
        conf_col = 'composite_confidence'
    elif 'confidence' in df.columns:
        conf_col = 'confidence'
    else:
        print("No confidence column found in log.")
        return

    conf = actual_trades[conf_col].astype(float)
    print(f"  Composite Confidence Mean:   {conf.mean():.4f}")
    print(f"  Composite Confidence Median: {conf.median():.4f}")
    print(f"  Composite Confidence Std:    {conf.std():.4f}")
    print(f"  Composite Confidence Min:    {conf.min():.4f}")
    print(f"  Composite Confidence Max:    {conf.max():.4f}")

    # Confidence tier distribution
    high = (conf >= 0.65).sum()
    medium = ((conf >= 0.50) & (conf < 0.65)).sum()
    low = (conf < 0.50).sum()
    n = len(conf) if len(conf) > 0 else 1
    print(f"\n  Tier Distribution:")
    print(f"    HIGH   (>= 0.65): {high:>5d} ({high/n*100:.1f}%)")
    print(f"    MEDIUM (0.50-0.65): {medium:>5d} ({medium/n*100:.1f}%)")
    print(f"    LOW    (< 0.50):  {low:>5d} ({low/n*100:.1f}%)")

    # Direction confidence vs JEV quality breakdown
    if 'direction_confidence' in actual_trades.columns and 'jev_quality' in actual_trades.columns:
        dir_conf = actual_trades['direction_confidence'].astype(float)
        jev_q = actual_trades['jev_quality'].astype(float)
        print(f"\n  Direction Confidence Mean:    {dir_conf.mean():.4f}")
        print(f"  JEV Execution Quality Mean:  {jev_q.mean():.2f}/10")

    if 'direction_prob' in actual_trades.columns:
        dp = actual_trades['direction_prob'].astype(float)
        print(f"\n  Direction Probability Mean:   {dp.mean():.4f}")
        print(f"  Direction Probability Std:    {dp.std():.4f}")
        bullish = (dp > 0.5).sum()
        bearish = (dp <= 0.5).sum()
        print(f"  Bullish Signals: {bullish}  |  Bearish Signals: {bearish}")

    # Action breakdown
    print(f"\n--- ACTION DISTRIBUTION ---")
    action_counts = actual_trades['action_label'].value_counts()
    for action, count in action_counts.items():
        print(f"  {action:<15s}: {count:>5d} ({count/len(actual_trades)*100:.1f}%)")

    # Per-symbol breakdown
    print(f"\n--- PER-SYMBOL SUMMARY ---")
    for sym in sorted(actual_trades['symbol'].unique()):
        sym_data = actual_trades[actual_trades['symbol'] == sym]
        print(f"  {sym:<12s}: {len(sym_data):>4d} trades, "
              f"avg conf={sym_data[conf_col].astype(float).mean():.3f}")

    # Recent trades
    print(f"\nLast 5 Trades:")
    cols = ['timestamp', 'symbol', 'side', 'price', conf_col, 'action_label']
    available = [c for c in cols if c in df.columns]
    print(df[available].tail(5).to_string(index=False))

    print("\n" + "=" * 60)
    print("  FIX VALIDATION SUMMARY")
    print("=" * 60)
    if conf.mean() >= 0.50:
        print("  [PASS] Mean composite confidence >= 0.50")
    else:
        print("  [FAIL] Mean composite confidence < 0.50 -- investigate further")

    if high / n >= 0.20:
        print(f"  [PASS] {high/n*100:.1f}% of signals in HIGH tier (>= 20% target)")
    else:
        print(f"  [WARN] Only {high/n*100:.1f}% in HIGH tier (target >= 20%)")

    print("  [INFO] For full tear sheet: pip install quantstats")
    print("         qs.reports.html(returns, output='tearsheet.html')")


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else "paper_trades_log.csv"
    analyze_paper_trades(log)
