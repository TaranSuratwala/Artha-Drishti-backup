import sys
import os
import re

def patch_ml_predictor():
    with open('MLPredictor.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # M1. CONFIG dict. Change keys and add new ones.
    # The config is defined as CONFIG = { ... }
    # Let's insert the new keys at the beginning of the dict.
    if "'entry_mode': 'next_open'," not in content:
        config_patch = """'early_stop_metric': 'direction_rank_ic',
    'min_delta_direction': 0.02,             # IC*100 units
    'transaction_cost_pct': 0.22,            # 2x0.10% delivery STT + exchange/SEBI/stamp/GST
    'slippage_pct': 0.08,                    # total 0.30% round trip (was 0.13%)
    'purge_gap_calendar_days': 45, 'purge_gap_size': 45,   # >= 30-trading-day aux label span
    'stabilize_regime_features': False,      # serving cannot reproduce expanding ranks
    'entry_mode': 'next_open',               # 'next_open' | 'close'
    'label_jump_clip': 0.40,
    'allow_sell_signals': False,             # cash delivery cannot short
    'ensemble_include_gbdt': False,
    'drop_panel_wide_inputs': True,          # ablate once vs False; serving supports True only
    'nifty_hedge_cost_pct': 0.03,
    'frozen_nifty_path': None,"""
        content = re.sub(r"(CONFIG\s*=\s*\{)", r"\1\n    " + config_patch, content, count=1)
        # We should also replace the old 'early_stop_metric', 'min_delta_direction' etc if they exist.
        # It's safer to just let the new ones override if the python dict evaluates them sequentially,
        # but to be clean, let's remove previous instances.
        keys_to_remove = ['early_stop_metric', 'min_delta_direction', 'transaction_cost_pct', 'slippage_pct', 'purge_gap_calendar_days', 'purge_gap_size', 'stabilize_regime_features']
        for k in keys_to_remove:
            content = re.sub(rf"'{k}'\s*:[^,\n]+,", "", content)
            content = re.sub(rf"\"{k}\"\s*:[^,\n]+,", "", content)

    # M2. Imports. Add after the sys.path.insert(0, _backend_dir) block.
    if "from nse_research import" not in content:
        import_patch = """
from nse_research import (fast_daily_rank_ic, decile_spread_long, nw_lrvar, nw_t,
                          cohort_sharpe, robust_panelwide, add_panel_nse_features, load_fii_series)
"""
        content = re.sub(r"(sys\.path\.insert\(0,\s*_backend_dir\)[^\n]*)", r"\1" + import_patch, content)

    # M3. _load_nifty50_benchmark
    if "Loaded FROZEN Nifty benchmark" not in content:
        # find def _load_nifty50_benchmark and the try block inside it
        m3_patch1 = """_fp = CONFIG.get('frozen_nifty_path')
        if _fp and os.path.exists(_fp):
            logger.info(f"   Loaded FROZEN Nifty benchmark: {_fp}")
            return joblib.load(_fp)
        """
        content = re.sub(r"(def _load_nifty50_benchmark.*?try:\s*\n)", r"\1        " + m3_patch1.strip() + "\n", content, flags=re.DOTALL)
        
        m3_patch2 = """if _fp:
            os.makedirs(os.path.dirname(_fp) or '.', exist_ok=True); joblib.dump(nifty_map, _fp)
        return nifty_map"""
        content = re.sub(r"return nifty_map(?=\s*$|\s*except)", m3_patch2, content)

    # M4. load_or_engineer_features
    if "NSE panel features added" not in content:
        m4_patch = """
        try:
            result_df = add_panel_nse_features(result_df, fii=load_fii_series(self.engine))
            logger.info(f"   NSE panel features added; columns={len(result_df.columns)}")
        except Exception as e:
            logger.warning(f"   NSE panel features skipped: {e}")
"""
        content = re.sub(r"(result_df\s*=\s*pd\.concat\(all_dfs,\s*ignore_index=True\)[^\n]*)", r"\1" + m4_patch, content)

    # M5. train(), frozen samplers
    content = content.replace("np.random.choice(len(df), _sample_size, replace=False)", "np.random.RandomState(12345).choice(len(df), _sample_size, replace=False)")
    content = content.replace("np.random.choice(len(train_index), _ic_sample_size, replace=False)", "np.random.RandomState(12345).choice(len(train_index), _ic_sample_size, replace=False)")
    content = content.replace("np.random.choice(len(df), _ref_sample_n, replace=False)", "np.random.RandomState(12345).choice(len(df), _ref_sample_n, replace=False)")
    if "'amihud'" not in content:
        content = re.sub(r"(ABSOLUTE_FEATURES\s*=\s*\{[^\}]*)\}", r"\1, 'amihud'}", content)

    # M6. train(), robust panel-wide detector
    if "_panel_wide_auto = robust_panelwide" not in content:
        m6_target = r"_global_std =.*?(?=if _panel_wide_auto:)"
        m6_replace = """_panel_wide_auto = robust_panelwide(_probe_df, _candidates)
        _pw = set(_panel_wide_auto)
        _cs_cols = [c for c in _candidates if c not in _pw]
        
        """
        content = re.sub(r"for c in _candidates:.*?_cs_cols\.append\(c\)\s*(?=if _panel_wide_auto:)", m6_replace, content, flags=re.DOTALL)

    # M7. train(), drop panel-wide inputs
    if "drop_panel_wide_inputs: removed" not in content:
        m7_patch = """
        if CONFIG.get('drop_panel_wide_inputs', True):
            _pw_all = set(_KNOWN_PANEL_WIDE) | {'nifty_return_5d', 'nifty_return_10d', 'nifty_vol_5', 'nifty_vol_10',
                      'nifty_vol_20', 'crude_change_5d', 'crude_change_20d', 'usdinr_change_5d', 'usdinr_change_20d',
                      'india_vix_sma_10', 'vrp', 'vrp_zscore', 'vix_inverted'}
            _dropped = [c for c in feature_cols if c in _pw_all]
            feature_cols = [c for c in feature_cols if c not in _pw_all]
            logger.info(f"   drop_panel_wide_inputs: removed {len(_dropped)}: {_dropped}")
"""
        content = re.sub(r"(self\.feature_cols\s*=\s*feature_cols)", m7_patch + r"\n        \1", content)

    # M8. train(), indexing loop
    if "ticker_open_arrays, ticker_jump_arrays =" not in content:
        content = content.replace("for ticker in tqdm(tickers, desc=\"Indexing Tickers\"):", "ticker_open_arrays, ticker_jump_arrays = [], []\n        for ticker in tqdm(tickers, desc=\"Indexing Tickers\"):")
        m8_patch = """
            _cl = ticker_df['close'].values.astype(np.float64)
            _ratio = np.where(_cl > 0, close_arr / np.where(_cl > 0, _cl, 1.0), 1.0)
            _open = ticker_df['open'].values.astype(np.float64) * _ratio if 'open' in ticker_df.columns else close_arr.astype(np.float64)
            _lc = np.log(np.maximum(close_arr.astype(np.float64), 1e-8))
            _dlr = np.abs(np.diff(_lc, prepend=_lc[0]))
            ticker_open_arrays.append(_open.astype(np.float32))
            ticker_jump_arrays.append(pd.Series(_dlr).rolling(pred_days).max().shift(-pred_days).values)   # max|dlr| over t+1..t+h"""
        content = re.sub(r"(ticker_arrays\.append\(\(feat_arr, close_arr, high_arr, low_arr, nifty_arr, natr_arr\)\)[^\n]*)", r"\1" + m8_patch, content)

    # M9. train(), split loop
    if "_bad = np.nan_to_num(ticker_jump_arrays[t_idx]" not in content:
        m9_patch = """
                _bad = np.nan_to_num(ticker_jump_arrays[t_idx][seq_len - 1: seq_len - 1 + n_valid], nan=0.0) > float(CONFIG.get('label_jump_clip', 0.40))
                rows = rows[~_bad]
                if rows.size == 0:
                    continue"""
        content = re.sub(r"(rows\s*=\s*np\.arange\(n_valid, dtype=np\.int64\)[^\n]*)", r"\1" + m9_patch, content)

    # M10. train(), targets loop
    if "ticker_raw_excess_close_arrays = []" not in content:
        content = content.replace("ticker_raw_excess_arrays = []", "ticker_raw_excess_arrays = []\n        ticker_raw_excess_close_arrays = []")
        content = content.replace("ticker_raw_excess_arrays.append(np.zeros((0,), dtype=np.float32))", "ticker_raw_excess_arrays.append(np.zeros((0,), dtype=np.float32))\n                ticker_raw_excess_close_arrays.append(np.zeros((0,), dtype=np.float32))")
        
        m10_patch1 = """stock_returns_close = np.log(fut_prices / (cur_prices + 1e-8))
            if CONFIG.get('entry_mode', 'next_open') == 'next_open':
                _entry = ticker_open_arrays[i][cur_indices + 1].astype(np.float64)
                stock_returns = np.log(fut_prices / np.maximum(_entry, 1e-8))
            else:
                stock_returns = stock_returns_close
            _jc = np.nan_to_num(ticker_jump_arrays[i][cur_indices], nan=0.0) > float(CONFIG.get('label_jump_clip', 0.40))
            stock_returns = np.where(_jc, 0.0, stock_returns)
            stock_returns_close = np.where(_jc, 0.0, stock_returns_close)"""
        content = re.sub(r"stock_returns\s*=\s*np\.log\(fut_prices\s*/\s*\(cur_prices\s*\+\s*1e-8\)\)", m10_patch1, content)
        
        content = re.sub(r"(raw_excess_returns\s*=\s*price_changes\.copy\(\)[^\n]*)", r"\1\n            raw_excess_close = stock_returns_close - market_returns", content)
        content = re.sub(r"(ticker_raw_excess_arrays\.append\(raw_excess_returns\.astype\(np\.float32\)\)[^\n]*)", r"\1\n            ticker_raw_excess_close_arrays.append(raw_excess_close.astype(np.float32))", content)

    # M11. train(), close-entry returns and validation context
    if "_raw_close_flat_all =" not in content:
        m11_patch1 = """
        _raw_close_flat_all = np.concatenate(ticker_raw_excess_close_arrays) if ticker_raw_excess_close_arrays else np.zeros(0, np.float32)
        _pos_te = _flat_positions(test_index)
        self._test_raw_returns_close = _raw_close_flat_all[_pos_te].astype(np.float32) if _pos_te.size else np.zeros(0, np.float32)"""
        content = re.sub(r"(self\._test_raw_returns\s*=\s*_build_raw_returns\(test_index\).*?[^\n]*)", r"\1" + m11_patch1, content)

        m11_patch2 = """
        if val_index:
            _vi = np.asarray(val_index, dtype=np.int64).reshape(-1, 2)
            _val_dates = _cs_dates_flat[_cs_off[_vi[:, 0]] + _vi[:, 1]]
            _val_raw_rets = _build_raw_returns(val_index).astype(np.float64)
        else:
            _val_dates, _val_raw_rets = np.zeros(0, 'datetime64[ns]'), np.zeros(0)"""
        # We need to insert this after `self._test_dates = ...`
        content = re.sub(r"(self\._test_dates\s*=\s*.*?else:.*?self\._test_dates\s*=\s*.*?[^\n]*)", r"\1" + m11_patch2, content, flags=re.DOTALL)

    # M12. Early stop on rank IC
    if "'direction_rank_ic'" not in content:
        content = re.sub(r"(_maximizing_metrics\s*=\s*\{[^\}]*)\}", r"\1, 'direction_rank_ic'}", content)
        
        m12_patch1 = """
                dir_rank_ic = float(np.nan_to_num(fast_daily_rank_ic(
                    val_preds_np['direction'], _val_dates, _val_raw_rets,
                    min_names=int(CONFIG.get('rank_ic_min_names', 30))).mean())) * 100.0
                self.metrics_history['direction_rank_ic'].append(dir_rank_ic)"""
        content = re.sub(r"(dir_quality\s*=\s*[^\n]*\n)", r"\1" + m12_patch1 + "\n", content)
        
        content = re.sub(r"(elif early_metric == 'direction_quality':\s*raw_monitor = dir_quality)", r"\1\n            elif early_metric == 'direction_rank_ic':\n                raw_monitor = dir_rank_ic", content)
        content = content.replace("if CONFIG.get('use_gap_penalized_es', True):", "if CONFIG.get('use_gap_penalized_es', True) and early_metric != 'direction_rank_ic':")
        # epoch summary log line:
        content = re.sub(r"(logger\.info\(f\"Epoch \{epoch.*?dir_q=\{dir_quality[^\"]*)(\"\))", r"\1 | IC={dir_rank_ic:+.2f}\2", content)

    # M13. F1, units bug
    if "TRUE excess returns" not in content:
        m13_patch = """_policy_returns = np.array([], dtype=np.float64)
        try:
            _policy_returns = _build_raw_returns(cal_index).astype(np.float64)   # TRUE excess returns, not z-scores
        except Exception as _policy_e:
            logger.warning(f"   Policy holdout returns unavailable ({_policy_e}) — will fallback to test for threshold tuning")"""
        content = re.sub(r"_policy_returns\s*=\s*np\.array\(\[\], dtype=np\.float64\)\s*\n\s*try:.*?(?=except Exception as _policy_e:)", m13_patch + "\n        except Exception as _policy_e:\n", content, flags=re.DOTALL)
        # we might need to adjust the regex for the except block since we already replaced it, let's just do a string replace of the whole try block
        old_try_block = re.search(r"_policy_returns\s*=\s*np\.array\(\[\], dtype=np\.float64\).*?threshold tuning\"\)", content, flags=re.DOTALL).group(0)
        content = content.replace(old_try_block, m13_patch)

    # M14. Sharpe from daily cohorts and test logits
    if "_backtest['paper_trade'] = _paper_backtest" in content and "daily_cohort_net_of_cost" not in content:
        m14_patch = """
        _cost = float(CONFIG['transaction_cost_pct']) + float(CONFIG['slippage_pct'])
        _hedge = float(CONFIG.get('nifty_hedge_cost_pct', 0.03))
        _bl = _backtest_long_only
        if len(self._test_dates) == len(_best_test_probs):
            _bl['per_trade_sharpe'] = _bl.get('sharpe_ratio')
            _bl['sharpe_ratio'] = round(cohort_sharpe(_best_test_probs, self._test_dates, _actual_returns,
                                        _bt_buy_thr, CONFIG['pred_days'], _cost, _hedge), 2)
            _bl['sharpe_definition'] = 'daily_cohort_net_of_cost_and_nifty_hedge'
        self._test_logits = np.asarray(test_dir_logits, dtype=np.float32)"""
        content = re.sub(r"(_backtest\['paper_trade'\]\s*=\s*_paper_backtest[^\n]*)", r"\1" + m14_patch, content)

    # M15. _report_rank_ic, full replacement
    if "fast_daily_rank_ic(probs" not in content:
        m15_patch = """def _report_rank_ic(self, probs: np.ndarray) -> Dict[str, float]:
        \"\"\"Rank IC / ICIR / decile spread with Newey-West (overlapping 5d labels), median-based spread.\"\"\"
        dates, rets = getattr(self, '_test_dates', None), getattr(self, '_test_raw_returns', None)
        if dates is None or rets is None or len(dates) != len(probs):
            return {}
        h, mn = int(CONFIG.get('pred_days', 5)), int(CONFIG.get('rank_ic_min_names', 30))
        ic = fast_daily_rank_ic(probs, dates, rets, mn)
        if ic.empty:
            return {}
        sp = decile_spread_long(probs, dates, rets, 0.1, mn)
        m = float(ic.mean()); lr = nw_lrvar(ic.values, h - 1)
        out = {
            'rank_ic_mean': m, 'rank_ic_std': float(ic.std()),
            'rank_ic_nw_t': nw_t(ic, h - 1),
            'rank_ic_ir_annualized': float(m / np.sqrt(lr) * np.sqrt(252.0 / h)),
            'rank_ic_positive_rate_pct': float((ic > 0).mean() * 100),
            'decile_spread_mean_pct': float(sp.mean() * 100), 'decile_spread_t_stat': nw_t(sp, h - 1),
            'n_dates_evaluated': float(len(ic)),
        }
        rc = getattr(self, '_test_raw_returns_close', None)
        if rc is not None and len(rc) == len(probs):
            ic_c = fast_daily_rank_ic(probs, dates, rc, mn)
            out['rank_ic_mean_close_entry'] = float(ic_c.mean())
            out['ic_open_over_close'] = float(m / ic_c.mean()) if abs(ic_c.mean()) > 1e-9 else 0.0
        logger.info("\\n--- CROSS-SECTIONAL SIGNAL QUALITY (Newey-West, lag %d) ---", h - 1)
        logger.info(f"   Rank IC (entry={CONFIG.get('entry_mode')}): {m:+.4f}  NW-t={out['rank_ic_nw_t']:+.2f}  "
                    f"close-entry IC={out.get('rank_ic_mean_close_entry', float('nan')):+.4f}  "
                    f"open/close={out.get('ic_open_over_close', float('nan')):.2f}")
        logger.info(f"   ICIR (NW long-run var, annualized): {out['rank_ic_ir_annualized']:+.2f}  "
                    f"positive-rate {out['rank_ic_positive_rate_pct']:.1f}%")
        logger.info(f"   Median decile spread: {out['decile_spread_mean_pct']:+.3f}% per {h}d (NW-t={out['decile_spread_t_stat']:+.2f})")
        return out"""
        content = re.sub(r"def _report_rank_ic\(self, probs: np\.ndarray\).*?return out", m15_patch, content, flags=re.DOTALL)

    # M16. _generate_signal, SELL gate
    if "'sell_disabled_cash_long_only'" not in content:
        m16_patch = """_sell_enabled = bool(CONFIG.get('allow_sell_signals', False)) and (
            not CONFIG.get('use_data_driven_side_gating', True) or bool(getattr(self, '_sell_side_significant', False)))
        _sell_gate_reason = 'sell_disabled_cash_long_only'"""
        content = re.sub(r"if CONFIG\.get\('use_data_driven_side_gating', True\):.*?_sell_gate_reason\s*=\s*'data_driven_suppression'", m16_patch, content, flags=re.DOTALL)

    # M17. _ensemble_predict, GBDT off by default
    content = content.replace("if hasattr(self, 'lgbm_model') and self.lgbm_model is not None:", "if CONFIG.get('ensemble_include_gbdt', False) and getattr(self, 'lgbm_model', None) is not None:")
    content = content.replace("if hasattr(self, 'xgb_model') and self.xgb_model is not None:", "if CONFIG.get('ensemble_include_gbdt', False) and getattr(self, 'xgb_model', None) is not None:")

    with open('MLPredictor.py', 'w', encoding='utf-8') as f:
        f.write(content)

def patch_advanced_feature_engine():
    with open('AdvancedFeatureEngine.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # A1. Bounded range denominators. Add helper at module level, replace in four places.
    if "def _rng(df):" not in content:
        content = re.sub(r"(import numpy as np\nimport pandas as pd\n)", r"\1\ndef _rng(df):\n    return np.maximum(df['high'] - df['low'], 1e-4 * df['close'])\n\n", content)
        
        # Replace (df['high'] - df['low'] + 1e-10) with _rng(df)
        content = content.replace("(df['high'] - df['low'] + 1e-10)", "_rng(df)")

    # A2. _days_to_expiry weekday
    if "last_date.dayofweek != _wd" not in content:
        a2_patch = """_wd = 1 if dt >= pd.Timestamp('2025-09-01') else 3      # Tuesday expiries after the 2025 change
        while last_date.dayofweek != _wd:"""
        content = content.replace("while last_date.dayofweek != 3:", a2_patch)

    # A3. _fno_features, FII lag and new table
    if "fii_dii_net_flow" not in content:
        a3_patch = """fii_df = pd.read_sql(text("SELECT d AS date, fii_net AS fii_dii_net_flow FROM fii_dii ORDER BY d"), conn)
            if not fii_df.empty:
                fii_df['date'] = pd.to_datetime(fii_df['date']).shift(-1)      # flow of day d usable from the NEXT session
                fii_df = fii_df.dropna(subset=['date'])
                df = df.merge(fii_df, on='date', how='left')
                got_fii = True"""
        content = re.sub(r"fii_df\s*=\s*pd\.read_sql\(\"SELECT date, fii_net_value.*?got_fii\s*=\s*True", a3_patch, content, flags=re.DOTALL)

    with open('AdvancedFeatureEngine.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    patch_ml_predictor()
    patch_advanced_feature_engine()
    print("Patched successfully")
