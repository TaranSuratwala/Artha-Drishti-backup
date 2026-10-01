#!/usr/bin/env python
"""apply_patches.py — run in backend/. Patches MLPredictor.py and AdvancedFeatureEngine.py (backup .bak).
Per file: every anchor must match the expected number of times, otherwise NOTHING is written."""
import sys, shutil, pathlib

def B(ind, s):
    """First line unindented (anchor keeps its own indent); later lines get `ind` spaces."""
    L = s.strip('\n').split('\n')
    return L[0] + ''.join(('\n' + ' ' * ind + l) if l.strip() else '\n' for l in L[1:])

def patch(path, edits):
    p = pathlib.Path(path)
    src = p.read_text(encoding='utf-8')
    for name, old, new, count in edits:
        n = src.count(old)
        if (count is None and n < 1) or (count is not None and n != count):
            sys.exit(f"[{path}] patch '{name}': expected {count if count else '>=1'} match(es), found {n}. Nothing written.")
        src = src.replace(old, new)
    shutil.copy2(p, str(p) + '.bak')
    p.write_text(src, encoding='utf-8')
    print(f"{path}: {len(edits)} patches applied (backup {p.name}.bak)")

ML = []
A = ML.append

# ---------------- CONFIG ----------------
A(('cfg_es',  "'early_stop_metric': 'direction_balanced_accuracy',", "'early_stop_metric': 'direction_rank_ic',", 1))
A(('cfg_tc',  "'transaction_cost_pct': 0.10,", "'transaction_cost_pct': 0.22,", 1))
A(('cfg_sl',  "'slippage_pct': 0.03,", "'slippage_pct': 0.08,", 1))
A(('cfg_pg1', "'purge_gap_size': 15,", "'purge_gap_size': 45,", 1))
A(('cfg_pg2', "'purge_gap_calendar_days': 15,", "'purge_gap_calendar_days': 45,", 1))
A(('cfg_stab', "'stabilize_regime_features': True,", "'stabilize_regime_features': False,", 1))
A(('cfg_new', "'model_version_tag': '76.0.0',", B(4, '''
'model_version_tag': '76.0.0',
'entry_mode': 'next_open',              # label/PnL entry: 'next_open' | 'close'
'label_jump_clip': 0.40,                # mask windows with a split-like daily |log ret| > this
'allow_sell_signals': False,            # cash delivery cannot short
'ensemble_include_gbdt': False,         # GBDT probs are uncalibrated vs NN isotonic scale
'drop_panel_wide_inputs': False,        # ablate: False vs True, keep winner by NW rank IC
'nifty_hedge_cost_pct': 0.03,
'frozen_nifty_path': None,              # certifier sets this
'''), 1))

# ---------------- imports ----------------
A(('imports', "if _backend_dir not in sys.path:\n    sys.path.insert(0, _backend_dir)",
   "if _backend_dir not in sys.path:\n    sys.path.insert(0, _backend_dir)\n\n"
   "from nse_research import (fast_daily_rank_ic, decile_spread_long, nw_lrvar, nw_t, cohort_sharpe,\n"
   "                          robust_panelwide, add_panel_nse_features, load_fii_series)", 1))

# ---------------- frozen Nifty ----------------
A(('nifty_load', "date_min = pd.to_datetime(df['date']).min() - pd.Timedelta(days=30)", B(12, '''
_fp = CONFIG.get('frozen_nifty_path')
if _fp and os.path.exists(_fp):
    logger.info(f"   Loaded FROZEN Nifty benchmark: {_fp}")
    return joblib.load(_fp)
date_min = pd.to_datetime(df['date']).min() - pd.Timedelta(days=30)
'''), 1))
A(('nifty_dump', 'logger.info(f"   Loaded Nifty 50 benchmark: {len(nifty_map)} trading days")', B(12, '''
logger.info(f"   Loaded Nifty 50 benchmark: {len(nifty_map)} trading days")
_fp2 = CONFIG.get('frozen_nifty_path')
if _fp2:
    os.makedirs(os.path.dirname(_fp2) or '.', exist_ok=True)
    joblib.dump(nifty_map, _fp2)
'''), 1))

# ---------------- NSE panel features after engineering ----------------
A(('panel_feats', "result_df = pd.concat(all_dfs, ignore_index=True)", B(8, '''
result_df = pd.concat(all_dfs, ignore_index=True)
try:
    result_df = add_panel_nse_features(result_df, fii=load_fii_series(self.engine))
    logger.info(f"   NSE panel features added; columns={len(result_df.columns)}")
except Exception as _e:
    logger.warning(f"   NSE panel features skipped: {_e}")
'''), 1))

# ---------------- seed-independent feature selection samples ----------------
A(('rs1', "_sample_idx = np.random.choice(len(df), _sample_size, replace=False)",
          "_sample_idx = np.random.RandomState(12345).choice(len(df), _sample_size, replace=False)", 1))
A(('rs2', "_ic_sample_idx = np.random.choice(len(train_index), _ic_sample_size, replace=False)",
          "_ic_sample_idx = np.random.RandomState(12345).choice(len(train_index), _ic_sample_size, replace=False)", 1))
A(('rs3', "_ref_idx = np.random.choice(len(df), _ref_sample_n, replace=False)",
          "_ref_idx = np.random.RandomState(12345).choice(len(df), _ref_sample_n, replace=False)", 1))

# ---------------- raw amihud out; robust panel-wide detector ----------------
A(('amihud_excl', "exclude_cols.update(ABSOLUTE_FEATURES)", B(8, '''
exclude_cols.update(ABSOLUTE_FEATURES)
exclude_cols.update({'amihud'})   # replaced by log_amihud_20 (ranked, tail-safe)
'''), 1))
A(('pw1', "_global_std = df[_candidates].std(axis=0)",
   B(16, "_global_std = df[_candidates].std(axis=0)\n_iqr_pw = set(robust_panelwide(_probe_df, _candidates))"), 1))
A(('pw2', "if _frac_panelwide >= 0.8:", "if c in _iqr_pw:", 1))

# ---------------- optional: drop panel-wide inputs ----------------
A(('drop_pw', "self.feature_cols = feature_cols\n        n_features = len(feature_cols)", B(8, '''
if CONFIG.get('drop_panel_wide_inputs', False):
    _pw_all = set(_KNOWN_PANEL_WIDE) | {
        'nifty_return_5d', 'nifty_return_10d', 'nifty_vol_5', 'nifty_vol_10', 'nifty_vol_20',
        'crude_change_5d', 'crude_change_20d', 'usdinr_change_5d', 'usdinr_change_20d',
        'india_vix_sma_10', 'india_vix_change', 'vrp', 'vrp_zscore', 'vix_inverted',
        'fii_dii_net_flow', 'fii_dii_flow_trend',
        'month', 'quarter', 'day_of_month', 'week_of_year', 'day_of_week', 'is_month_start', 'is_month_end'}
    _dropped = [c for c in feature_cols if c in _pw_all]
    feature_cols = [c for c in feature_cols if c not in _pw_all]
    logger.info(f"   drop_panel_wide_inputs: removed {len(_dropped)}: {_dropped}")
self.feature_cols = feature_cols
n_features = len(feature_cols)
'''), 1))

# ---------------- indexing loop: open + jump arrays ----------------
A(('ix0', "ticker_keys_for_arrays: List[str] = []",
   B(8, "ticker_keys_for_arrays: List[str] = []\nticker_open_arrays: List[np.ndarray] = []\nticker_jump_arrays: List[np.ndarray] = []"), 1))
A(('ix1', "ticker_idx = len(ticker_arrays)", B(16, '''
_cl = ticker_df['close'].values.astype(np.float64)
_ratio = np.where(_cl > 0, close_arr / np.where(_cl > 0, _cl, 1.0), 1.0)
_open64 = (ticker_df['open'].values.astype(np.float64) * _ratio) if 'open' in ticker_df.columns else close_arr.astype(np.float64)
_lc = np.log(np.maximum(close_arr.astype(np.float64), 1e-8))
_jump = pd.Series(np.abs(np.diff(_lc, prepend=_lc[0]))).rolling(pred_days).max().shift(-pred_days).values
ticker_idx = len(ticker_arrays)
'''), 1))
A(('ix2', "ticker_arrays.append((feat_arr, close_arr, high_arr, low_arr, nifty_arr, natr_arr))", B(16, '''
ticker_arrays.append((feat_arr, close_arr, high_arr, low_arr, nifty_arr, natr_arr))
ticker_open_arrays.append(_open64.astype(np.float32))
ticker_jump_arrays.append(_jump)
'''), 1))

# ---------------- split loop: drop split-like windows ----------------
A(('split_mask', "rows = np.arange(n_valid, dtype=np.int64)\n            seq_end = date_arr[rows + seq_len - 1]", B(12, '''
rows = np.arange(n_valid, dtype=np.int64)
_bad = np.nan_to_num(ticker_jump_arrays[t_idx][rows + seq_len - 1], nan=0.0) > float(CONFIG.get('label_jump_clip', 0.40))
rows = rows[~_bad]
if rows.size == 0:
    continue
seq_end = date_arr[rows + seq_len - 1]
'''), 1))

# ---------------- targets: open-entry returns ----------------
A(('tg0', "ticker_raw_excess_arrays = []", B(8, "ticker_raw_excess_arrays = []\nticker_raw_excess_close_arrays = []"), 1))
A(('tg1', "ticker_raw_excess_arrays.append(np.zeros((0,), dtype=np.float32))",
   B(16, "ticker_raw_excess_arrays.append(np.zeros((0,), dtype=np.float32))\nticker_raw_excess_close_arrays.append(np.zeros((0,), dtype=np.float32))"), 1))
A(('tg2', "stock_returns = np.log(fut_prices / (cur_prices + 1e-8))", B(12, '''
stock_returns_close = np.log(fut_prices / (cur_prices + 1e-8))
if CONFIG.get('entry_mode', 'next_open') == 'next_open':
    _entry = ticker_open_arrays[i][cur_indices + 1].astype(np.float64)
    stock_returns = np.log(fut_prices / np.maximum(_entry, 1e-8))
else:
    stock_returns = stock_returns_close
_jc = np.nan_to_num(ticker_jump_arrays[i][cur_indices], nan=0.0) > float(CONFIG.get('label_jump_clip', 0.40))
stock_returns = np.where(_jc, 0.0, stock_returns)
stock_returns_close = np.where(_jc, 0.0, stock_returns_close)
'''), 1))
A(('tg3', "raw_excess_returns = price_changes.copy()",
   B(12, "raw_excess_returns = price_changes.copy()\nraw_excess_close = stock_returns_close - market_returns"), 1))
A(('tg4', "ticker_raw_excess_arrays.append(raw_excess_returns.astype(np.float32))",
   B(12, "ticker_raw_excess_arrays.append(raw_excess_returns.astype(np.float32))\nticker_raw_excess_close_arrays.append(raw_excess_close.astype(np.float32))"), 1))

# ---------------- close-entry test returns + validation context ----------------
A(('v1', "self._test_raw_returns = _build_raw_returns(test_index) if test_index else np.zeros(0, dtype=np.float32)", B(8, '''
self._test_raw_returns = _build_raw_returns(test_index) if test_index else np.zeros(0, dtype=np.float32)
_raw_close_flat_all = np.concatenate(ticker_raw_excess_close_arrays) if ticker_raw_excess_close_arrays else np.zeros(0, np.float32)
_pos_te = _flat_positions(test_index)
self._test_raw_returns_close = _raw_close_flat_all[_pos_te].astype(np.float32) if _pos_te.size else np.zeros(0, np.float32)
'''), 1))
A(('v2', 'logger.info(f"   Raw test returns saved: {len(self._test_raw_returns):,} samples, "', B(8, '''
if val_index:
    _vi = np.asarray(val_index, dtype=np.int64).reshape(-1, 2)
    _val_dates = _cs_dates_flat[_cs_off[_vi[:, 0]] + _vi[:, 1]]
    _val_raw_rets = _build_raw_returns(val_index).astype(np.float64)
else:
    _val_dates, _val_raw_rets = np.zeros(0, 'datetime64[ns]'), np.zeros(0)
logger.info(f"   Raw test returns saved: {len(self._test_raw_returns):,} samples, "
'''), 1))

# ---------------- early stopping on val rank IC ----------------
A(('es1', "best_score = -float('inf') if early_metric in _maximizing_metrics else float('inf')", B(8, '''
_maximizing_metrics.add('direction_rank_ic')
best_score = -float('inf') if early_metric in _maximizing_metrics else float('inf')
'''), 1))
A(('es2', "dir_quality = self._compute_direction_quality_score(epoch_metrics.get('direction_metrics', {}))", B(12, '''
dir_quality = self._compute_direction_quality_score(epoch_metrics.get('direction_metrics', {}))
dir_rank_ic = float(np.nan_to_num(fast_daily_rank_ic(
    val_preds_np['direction'], _val_dates, _val_raw_rets,
    min_names=int(CONFIG.get('rank_ic_min_names', 30))).mean())) * 100.0
self.metrics_history['direction_rank_ic'].append(dir_rank_ic)
'''), 1))
A(('es3', "raw_monitor = dir_quality",
   "raw_monitor = dir_quality\n            elif early_metric == 'direction_rank_ic':\n                raw_monitor = dir_rank_ic", 1))
A(('es4', "if CONFIG.get('use_gap_penalized_es', True):",
   "if CONFIG.get('use_gap_penalized_es', True) and early_metric != 'direction_rank_ic':", 1))

# ---------------- F1: real returns for threshold policy ----------------
A(('policy_ret', "self._conformal_calibration = self._fit_conformal_calibration(", B(12, '''
try:
    _policy_returns = _build_raw_returns(cal_index).astype(np.float64)   # TRUE excess returns, not z-scores
except Exception as _pol_e:
    logger.warning(f"   Raw policy returns unavailable ({_pol_e})")
self._conformal_calibration = self._fit_conformal_calibration(
'''), 1))

# ---------------- cohort Sharpe + test logits ----------------
A(('cohort', "_backtest['paper_trade'] = _paper_backtest", B(12, '''
_backtest['paper_trade'] = _paper_backtest
_cost = float(CONFIG['transaction_cost_pct']) + float(CONFIG['slippage_pct'])
_bl = _backtest_long_only
if len(self._test_dates) == len(_best_test_probs):
    _bl['per_trade_sharpe'] = _bl.get('sharpe_ratio')
    _bl['sharpe_ratio'] = round(cohort_sharpe(_best_test_probs, self._test_dates, _actual_returns,
                                _bt_buy_thr, CONFIG['pred_days'], _cost,
                                float(CONFIG.get('nifty_hedge_cost_pct', 0.03))), 2)
    _bl['sharpe_definition'] = 'daily_cohort_net_of_cost_and_nifty_hedge'
self._test_logits = np.asarray(test_dir_logits, dtype=np.float32)
'''), 1))

# ---------------- rank IC (Newey-West) ----------------
A(('ric', "def _report_rank_ic(self, probs: np.ndarray) -> Dict[str, float]:", B(4, '''
def _report_rank_ic(self, probs: np.ndarray) -> Dict[str, float]:
    """Rank IC / ICIR / decile spread; Newey-West for overlapping labels; median-based spread."""
    dates = getattr(self, '_test_dates', None)
    rets = getattr(self, '_test_raw_returns', None)
    if dates is None or rets is None or len(dates) != len(probs):
        return {}
    h = int(CONFIG.get('pred_days', 5)); mn = int(CONFIG.get('rank_ic_min_names', 30))
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
        'decile_spread_mean_pct': float(sp.mean() * 100),
        'decile_spread_t_stat': nw_t(sp, h - 1),
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
    logger.info(f"   Median decile spread: {out['decile_spread_mean_pct']:+.3f}% per {h}d "
                f"(NW-t={out['decile_spread_t_stat']:+.2f})")
    return out

def _report_rank_ic_legacy(self, probs: np.ndarray) -> Dict[str, float]:
'''), 1))

# ---------------- signals ----------------
A(('sell', "if not _sell_enabled:", B(12, '''
if not CONFIG.get('allow_sell_signals', False):
    _sell_enabled = False
    _sell_gate_reason = 'sell_disabled_cash_long_only'
if not _sell_enabled:
'''), 1))
A(('ens_l', "if hasattr(self, 'lgbm_model') and self.lgbm_model is not None:",
   "if CONFIG.get('ensemble_include_gbdt', False) and getattr(self, 'lgbm_model', None) is not None:", 1))
A(('ens_x', "if hasattr(self, 'xgb_model') and self.xgb_model is not None:",
   "if CONFIG.get('ensemble_include_gbdt', False) and getattr(self, 'xgb_model', None) is not None:", 1))

AFE = []
F = AFE.append
F(('rng_helper', "\nlogger = logging.getLogger(__name__)\n",
   "\nlogger = logging.getLogger(__name__)\n\n\ndef _rng(df):\n    \"\"\"High-low range floored at 1bp of price (locked/unadjusted rows no longer explode ratios).\"\"\"\n"
   "    return np.maximum(df['high'] - df['low'], 1e-4 * df['close'])\n", 1))
F(('rng_use', "(df['high'] - df['low'] + 1e-10)", "_rng(df)", None))
F(('exp_wd', "def _days_to_expiry(dt):",
   B(12, "def _days_to_expiry(dt):\n    _wd = 1 if dt >= pd.Timestamp('2025-09-01') else 3   # Tuesday expiries post-2025 change (verify vs NSE circular)"), 1))
F(('exp_loop', "while last_date.dayofweek != 3:", "while last_date.dayofweek != _wd:", 2))
F(('fii_sql', 'fii_df = pd.read_sql("SELECT date, fii_net_value as fii_dii_net_flow FROM fii_dii_flow", conn)',
   'fii_df = pd.read_sql(text("SELECT d AS date, fii_net AS fii_dii_net_flow FROM fii_dii ORDER BY d"), conn)', 1))
F(('fii_lag', "fii_df['date'] = pd.to_datetime(fii_df['date'])", B(20, '''
fii_df['date'] = pd.to_datetime(fii_df['date']).shift(-1)   # flow of day d usable from the NEXT session
fii_df = fii_df.dropna(subset=['date'])
'''), 1))

if __name__ == '__main__':
    patch('MLPredictor.py', ML)
    patch('AdvancedFeatureEngine.py', AFE)