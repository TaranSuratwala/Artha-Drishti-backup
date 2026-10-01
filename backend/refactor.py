import re
import sys

def modify_mlpredictor():
    with open('MLPredictor.py', 'r', encoding='utf-8') as f:
        code = f.read()

    # 1. Update stoploss logic to use volatility
    code = re.sub(
        r'(rule_stoploss = current_price - sl_multiplier \* atr_20.*?)final_stoploss = 0\.6 \* rule_stoploss \+ 0\.4 \* pattern_stoploss',
        r'''\1
            pred_vol = pred_mean.get('volatility', atr_20) if isinstance(pred_mean, dict) else atr_20
            rule_stoploss = current_price - sl_multiplier * pred_vol
            final_stoploss = 0.6 * rule_stoploss + 0.4 * pattern_stoploss''',
        code, flags=re.DOTALL
    )

    # 2. Risk Reward for Bullish
    code = re.sub(
        r'(_rr_valid = \(final_target > buy_price\) and \(final_stoploss < buy_price\)\s+_risk = buy_price - final_stoploss)',
        r'''\1
                _reward = final_target - buy_price
                rr_ratio = _reward / max(_risk, 1e-5)''',
        code
    )

    # 3. Risk Reward for Bearish
    code = re.sub(
        r'(_rr_valid = \(final_target < buy_price\) and \(final_stoploss > buy_price\)\s+_risk = final_stoploss - buy_price)',
        r'''\1
                _reward = buy_price - final_target
                rr_ratio = _reward / max(_risk, 1e-5)''',
        code
    )

    # 4. Insert rr_ratio into trade_setup
    code = re.sub(
        r"('stop_loss': round\(final_stoploss, 2\) if not _p_suppress else None,)",
        r"\1\n                    'rr_ratio': round(rr_ratio, 2) if '_rr_valid' in locals() and _rr_valid else None,",
        code
    )

    # 5. Insert walk forward validation
    walk_forward_code = '''
        try:
            from walkforward import PurgedWalkForwardSplitter
            gap_days = int(CONFIG.get('purge_gap_calendar_days', CONFIG.get('purge_gap_size', 60))) if CONFIG.get('purge_gap', True) else 0
            splitter = PurgedWalkForwardSplitter(embargo_days=gap_days)
            windows = splitter.split(pd.Timestamp(all_dates[0]), pd.Timestamp(all_dates[-1]))
            final_window = windows[-1]
            train_end_date = final_window.train_end
            val_start_date = final_window.test_start
            val_end_date = final_window.test_end
            cal_start_date = val_end_date + pd.Timedelta(days=gap_days)
            cal_end_date = cal_start_date + pd.Timedelta(days=60)
            test_start_date = cal_end_date + pd.Timedelta(days=gap_days)
            logger.info(f"Applying Walk-Forward Split: Train->{train_end_date.date()} | Val {val_start_date.date()}->{val_end_date.date()}")
        except Exception as e:
            logger.warning(f"Failed to load WalkForwardSplitter, falling back to chronological split: {e}")
            train_cut_idx = max(int(len(all_dates) * 0.60) - 1, 0)
            val_cut_idx = max(int(len(all_dates) * 0.70) - 1, train_cut_idx)
            cal_cut_idx = max(int(len(all_dates) * 0.85) - 1, val_cut_idx)
            
            train_end_date = pd.Timestamp(all_dates[train_cut_idx])
            val_end_date = pd.Timestamp(all_dates[val_cut_idx])
            cal_end_date = pd.Timestamp(all_dates[cal_cut_idx])
            
            gap_days = int(CONFIG.get('purge_gap_calendar_days', CONFIG.get('purge_gap_size', 60))) if CONFIG.get('purge_gap', True) else 0
            val_start_date = train_end_date + pd.Timedelta(days=gap_days)
            cal_start_date = val_end_date + pd.Timedelta(days=gap_days)
            test_start_date = cal_end_date + pd.Timedelta(days=gap_days)
    '''

    code = re.sub(
        r'train_cut_idx = max\(int\(len\(all_dates\) \* 0\.60\) - 1, 0\).*?test_start_date = cal_end_date \+ pd\.Timedelta\(days=gap_days\)',
        walk_forward_code.strip(),
        code, flags=re.DOTALL
    )

    with open('MLPredictor.py', 'w', encoding='utf-8') as f:
        f.write(code)
        
if __name__ == '__main__':
    modify_mlpredictor()
