"""nse_research.py — NSE labels, evaluation, panel features (shared by training and serving)."""
import numpy as np, pandas as pd

# ------------------------------ labels ------------------------------
def open_entry_cs_labels(O, C, h=5, clip=0.40, min_names=50, ls=0.02, mkt=None):
    """O,C: adjusted wide panels (date x ticker). Entry O[t+1], exit C[t+h]. mkt=(mO,mC) Series."""
    fwd = np.log(C.shift(-h) / O.shift(-1))
    if mkt is not None:
        mO, mC = mkt
        fwd = fwd.sub(np.log(mC.shift(-h) / mO.shift(-1)), axis=0)
    jump = np.log(C / C.shift(1)).abs().rolling(h).max().shift(-h)
    fwd = fwd.mask(jump > clip)
    n = fwd.notna().sum(axis=1)
    pct = fwd.rank(axis=1, pct=True)
    y = (pct > 0.5).astype(float) * (1 - 2 * ls) + ls
    ok = fwd.notna() & (n >= min_names).values[:, None]
    return fwd.where(ok), pct.where(ok), y.where(ok)

# ---------------------------- evaluation ----------------------------
def _frame(scores, dates, rets):
    df = pd.DataFrame({'d': np.asarray(dates), 's': np.asarray(scores, float), 'r': np.asarray(rets, float)})
    return df[np.isfinite(df.s) & np.isfinite(df.r)]

def fast_daily_rank_ic(scores, dates, rets, min_names=30):
    df = _frame(scores, dates, rets)
    df = df[df.groupby('d').s.transform('size') >= min_names]
    if df.empty:
        return pd.Series(dtype=float)
    g = df.groupby('d')
    a, b = g.s.rank(), g.r.rank()
    a = a - a.groupby(df.d).transform('mean'); b = b - b.groupby(df.d).transform('mean')
    num = (a * b).groupby(df.d).sum()
    den = np.sqrt((a * a).groupby(df.d).sum() * (b * b).groupby(df.d).sum())
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()

def decile_spread_long(scores, dates, rets, q=0.1, min_names=30):
    """Median-based top-minus-bottom (robust to residual split/outlier returns)."""
    df = _frame(scores, dates, rets).copy()
    df['p'] = df.groupby('d').s.rank(pct=True)
    sp = df[df.p > 1 - q].groupby('d').r.median() - df[df.p <= q].groupby('d').r.median()
    n = df.groupby('d').size()
    return sp[sp.index.isin(n[n >= min_names].index)].dropna()

def nw_lrvar(x, L=4):
    e = np.asarray(x, float); e = e[np.isfinite(e)]; n = len(e)
    if n < 3:
        return float('nan')
    e = e - e.mean(); L = min(L, n - 1)
    v = e @ e / n + 2 * sum((1 - k / (L + 1)) * (e[k:] @ e[:-k]) / n for k in range(1, L + 1))
    return max(v, 1e-18)

def nw_t(x, L=4):
    e = np.asarray(pd.Series(x).dropna(), float)
    return float(e.mean() / np.sqrt(nw_lrvar(e, L) / len(e))) if len(e) > 3 else 0.0

def ic_by_liquidity(score, fwd, C, V, k=3):
    """Wide panels. {tercile: (mean_ic, nw_t)}; 1 = least liquid."""
    b = np.ceil((C * V).rolling(20).median().rank(axis=1, pct=True) * k)
    out = {}
    for i in range(1, k + 1):
        ic = score.where(b == i).rank(axis=1).corrwith(fwd.where(b == i).rank(axis=1), axis=1)
        out[i] = (float(ic.mean()), nw_t(ic))
    return out

def cohort_sharpe(scores, dates, rets, thr, h, cost_pct, hedge_cost_pct=0.03, clip=0.30, top_frac=None):
    """Daily-cohort Sharpe: equal-weight cohort per date, non-overlapping h-day subsamples averaged."""
    df = pd.DataFrame({'d': np.asarray(dates), 's': np.asarray(scores, float),
                       'r': np.clip(np.nan_to_num(np.asarray(rets, float)), -clip, clip)})
    sel = (df.s > thr) if top_frac is None else (df.groupby('d').s.rank(pct=True) > 1 - top_frac)
    net = np.expm1(df.r[sel]).groupby(df.d[sel]).mean() - (cost_pct + hedge_cost_pct) / 100.0
    coh = net.reindex(np.sort(df.d.unique())).fillna(0.0)
    sh = [x.mean() / x.std() * np.sqrt(252.0 / h) for o in range(h)
          for x in [coh.iloc[o::h]] if len(x) > 2 and x.std() > 0]
    return float(np.mean(sh)) if sh else 0.0

def ensemble_gain(rho, K):
    """IC_K / IC_1 for K seeds with pairwise score correlation rho."""
    r = (1 - rho) / max(rho, 1e-6)
    return float(np.sqrt((1 + r) / (1 + r / K)))

# ------------------------------ features ------------------------------
def robust_panelwide(df, cols, date='date', thr=0.01):
    """IQR-ratio detector (immune to heavy tails); std-ratio fallback when global IQR is 0 (sparse flags)."""
    g = df.groupby(date)[cols]
    iqr_w = (g.quantile(.75) - g.quantile(.25)).median()
    iqr_g = df[cols].quantile(.75) - df[cols].quantile(.25)
    sd_w, sd_g = g.std().median(), df[cols].std()
    out = []
    for c in cols:
        if iqr_g[c] > 0:
            r = iqr_w[c] / iqr_g[c]
        else:
            r = sd_w[c] / sd_g[c] if sd_g[c] > 0 else 0.0
        if not np.isfinite(r) or r < thr:
            out.append(c)
    return out

def log_amihud(ret, C, V, w=20):
    dv = (C * V).clip(lower=1e5)                                    # INR turnover floor
    return np.log1p(1e9 * (ret.abs() / dv).rolling(w, min_periods=10).median())

def gap_feats(O, H, L, C, w=60):
    g = np.log(O / C.shift(1))
    gz = (g / g.clip(-.1, .1).rolling(w, min_periods=30).std()).clip(-4, 4)
    r1 = C / C.shift(1) - 1
    mx = r1.abs().clip(upper=.25).rolling(250, min_periods=60).max().fillna(.2)
    tier = pd.DataFrame(np.select([mx.values < .03, mx.values < .07, mx.values < .15],
                                  [.02, .05, .10], .20), index=mx.index, columns=mx.columns)
    return {'gap_z': gz,
            'up_lock': ((r1 >= .99 * tier) & (C >= .9999 * H)).astype(float),
            'dn_lock': ((r1 <= -.99 * tier) & (C <= 1.0001 * L)).astype(float)}

def flow_beta_feature(R, F, w=60, lag=1):
    """R: return panel; F: FII net / mean20|FII| (Series). Exposure (through t-1) x lagged shock."""
    mu = lambda x: x.rolling(w, min_periods=w // 2).mean()
    cov = mu(R.mul(F, axis=0)) - mu(R).mul(mu(F), axis=0)
    beta = cov.div(F.rolling(w, min_periods=w // 2).var(), axis=0).shift(1)
    Fz = ((F - F.rolling(120).mean()) / F.rolling(120).std()).shift(lag)
    return beta.mul(Fz, axis=0).clip(-5, 5)

def vix_stress_x_illiq(vix, illiq_rank):
    lv = np.log(vix); z = (lv - lv.rolling(60).mean()) / lv.rolling(60).std()
    return illiq_rank.mul((z - 1).clip(lower=0), axis=0)

def load_fii_series(engine):
    from sqlalchemy import text
    try:
        with engine.connect() as c:
            d = pd.read_sql(text("SELECT d, fii_net FROM fii_dii ORDER BY d"), c)
    except Exception:
        return None
    if d.empty:
        return None
    return pd.Series(d.fii_net.values, index=pd.to_datetime(d.d))

def add_panel_nse_features(df, fii=None):
    """df: long engineered frame with date,ticker,open,high,low,close,adj_close,volume[,india_vix]."""
    d = df[['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']].copy()
    d['date'] = pd.to_datetime(d['date'])
    W = {c: d.pivot(index='date', columns='ticker', values=c) for c in d.columns[2:]}
    ratio = (W['adj_close'] / W['close'].replace(0, np.nan)).clip(.01, 100).fillna(1.0)
    O, H, L, C = W['open'] * ratio, W['high'] * ratio, W['low'] * ratio, W['adj_close']
    V = W['volume'] / ratio
    ret = np.log(C / C.shift(1))
    la = log_amihud(ret, C, V)
    out = {'log_amihud_20': la, **gap_feats(O, H, L, C)}
    if fii is not None and len(fii):
        F = fii.reindex(C.index).ffill()
        F = F / F.abs().rolling(20, min_periods=5).mean()
        out['flow_beta_fii'] = flow_beta_feature(ret, F)
    if 'india_vix' in df.columns:
        vix = df.assign(date=pd.to_datetime(df['date'])).groupby('date')['india_vix'].first() * 100
        out['vix_stress_x_illiq'] = vix_stress_x_illiq(vix.reindex(C.index).ffill(), la.rank(axis=1, pct=True))
    feats = pd.concat({k: v.stack() for k, v in out.items()}, axis=1)
    feats.index.names = ['date', 'ticker']
    df = df.drop(columns=[c for c in feats.columns if c in df.columns]).copy()
    df['date'] = pd.to_datetime(df['date'])
    return df.merge(feats.reset_index(), on=['date', 'ticker'], how='left')