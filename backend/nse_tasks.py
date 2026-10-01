"""nse_tasks.py — Celery: prospective pre-open capture, EOD ingest, cross-sectional scoring, IC monitor."""
import os, logging
import numpy as np, pandas as pd, requests
from celery import Celery, chain
from celery.schedules import crontab
from sqlalchemy import create_engine, text
from nse_research import add_panel_nse_features, load_fii_series, fast_daily_rank_ic, open_entry_cs_labels

log = logging.getLogger(__name__)
app = Celery('nse', broker=os.environ['CELERY_BROKER'])
app.conf.timezone = 'Asia/Kolkata'
app.conf.beat_schedule = {
    'preopen': dict(task='nse.preopen_snap', schedule=crontab(minute='8-11', hour=9, day_of_week='1-5')),
    'eod':     dict(task='nse.eod_chain',   schedule=crontab(minute=30, hour=19, day_of_week='1-5')),
}
eng = create_engine(os.environ['DATABASE_URL'], pool_pre_ping=True)
H = int(os.getenv('PRED_DAYS', 5))
UA = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}

def _get_json(url):
    s = requests.Session(); s.headers.update(UA)
    s.get(os.getenv('NSE_HOME', 'https://www.nseindia.com'), timeout=10)     # cookie priming
    r = s.get(url, timeout=15); r.raise_for_status(); return r.json()

def _is_trading_day(d):                          # replace with exchange holiday calendar
    return d.weekday() < 5

@app.task(name='nse.preopen_snap')
def preopen_snap():
    ts = pd.Timestamp.now(tz='Asia/Kolkata')
    if not _is_trading_day(ts):
        return 0
    js = _get_json(os.environ['NSE_PREOPEN_URL'])
    rows = []
    for it in js.get('data', []):                                            # ADAPT keys to your feed
        m, d = it.get('metadata', {}), it.get('detail', {}).get('preOpenMarket', {})
        rows.append(dict(ts=ts, symbol=m.get('symbol'), iep=m.get('iep', d.get('IEP')),
                         tot_buy=d.get('totalBuyQuantity'), tot_sell=d.get('totalSellQuantity'),
                         prev_close=m.get('previousClose')))
    if rows:
        pd.DataFrame(rows).dropna(subset=['symbol', 'iep']).to_sql(
            'preopen_snap', eng, if_exists='append', index=False, method='multi')
    return len(rows)

@app.task(name='nse.ingest_fii_dii')
def ingest_fii_dii():
    df = pd.DataFrame(_get_json(os.environ['NSE_FIIDII_URL']))               # [{category,date,netValue},...]
    df['d'] = pd.to_datetime(df['date'], format=os.getenv('FIIDII_DATE_FMT', '%d-%b-%Y')).dt.date
    df['v'] = pd.to_numeric(df['netValue'], errors='coerce')
    df['who'] = np.where(df['category'].str.upper().str.contains('FII'), 'fii', 'dii')
    piv = df.pivot_table(index='d', columns='who', values='v', aggfunc='sum')
    with eng.begin() as c:
        for d, r in piv.iterrows():
            c.execute(text("INSERT INTO fii_dii(d,fii_net,dii_net) VALUES(:d,:f,:i) "
                           "ON CONFLICT(d) DO UPDATE SET fii_net=:f, dii_net=:i"),
                      dict(d=d, f=float(r.get('fii', np.nan)), i=float(r.get('dii', np.nan))))

@app.task(name='nse.ingest_bhav')
def ingest_bhav(csv_path=None):
    """UDiFF F&O bhavcopy CSV; column names per current file — VERIFY."""
    df = pd.read_csv(csv_path or os.environ['NSE_FO_BHAV_CSV'])
    m = {'STF': 'FUTSTK', 'STO': 'OPTSTK'}
    df = df[df['FinInstrmTp'].isin(m)]
    out = pd.DataFrame(dict(d=pd.to_datetime(df['TradDt']).dt.date, symbol=df['TckrSymb'],
        instr=df['FinInstrmTp'].map(m), expiry=pd.to_datetime(df['XpryDt']).dt.date, strike=df['StrkPric'],
        opt_type=df['OptnTp'], oi=df['OpnIntrst'], chg_oi=df['ChngInOpnIntrst'], settle=df['SttlmPric']))
    out.to_sql('fo_bhav', eng, if_exists='append', index=False, method='multi', chunksize=20000)

@app.task(name='nse.refresh_feats')
def refresh_feats():
    return True                                    # views are live; hook for materialised feature tables

def _load_recent(n=320):
    q = text("""SELECT ticker,date,open,high,low,close,volume,adj_close FROM nse_stocks
                WHERE date >= (SELECT max(date) FROM nse_stocks) - :n * interval '1 day' ORDER BY ticker,date""")
    with eng.connect() as c:
        return pd.read_sql(q, c, params={'n': int(n * 1.6)})

@app.task(name='nse.score_universe')
def score_universe():
    """Cross-sectional batch scoring on the REAL same-day universe (same feature path as training)."""
    from concurrent.futures import ProcessPoolExecutor
    from MLPredictor import UnifiedStockPredictor, _engineer_one_ticker, rolling_zscore_matrix, CONFIG
    import torch
    raw = _load_recent()
    groups = [(t, g) for t, g in raw.groupby('ticker') if len(g) >= CONFIG['seq_len'] + 60]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(_engineer_one_ticker, [t for t, _ in groups], [g for _, g in groups]))
    df = pd.concat([r[0] for r in res if r[0] is not None], ignore_index=True)
    df = add_panel_nse_features(df, fii=load_fii_series(eng))
    asof = df['date'].max()

    seed_dirs = [d for d in os.getenv('SEED_DIRS', '').split(',') if d] or [None]
    p = UnifiedStockPredictor()
    per_seed = []
    for sd in seed_dirs:
        if sd:
            p._set_artifact_base_dir(sd)
        p._load_model()
        cols = list(p.feature_cols)
        for c in cols:
            if c not in df:
                df[c] = 0.0
        rk = [c for c in getattr(p, '_cross_sectional_ranked_cols', []) if c in cols]
        d2 = df[['date', 'ticker'] + cols].copy()
        d2[rk] = (df.groupby('date')[rk].rank(pct=True).astype(np.float32) - 0.5) * 2.0
        X, syms = [], []
        for t, g in d2.groupby('ticker'):
            a = rolling_zscore_matrix(g[cols].to_numpy(np.float64), 252, 30)[-CONFIG['seq_len']:]
            if len(a) == CONFIG['seq_len'] and g['date'].iloc[-1] == asof:
                X.append(a); syms.append(t)
        X = torch.from_numpy(np.stack(X)).float()
        out = []
        p.model.eval()
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                out.append(p.model(X[i:i + 2048].to(p.device))['direction'].float().cpu().numpy().ravel())
        per_seed.append(pd.Series(np.concatenate(out), index=syms).rank(pct=True))
    score = pd.concat(per_seed, axis=1).mean(axis=1)                        # rank-average across seeds
    rank_pct = score.rank(pct=True)
    turnover = raw.assign(v=raw.close * raw.volume).groupby('ticker').v.apply(lambda s: s.tail(20).median())
    liq = turnover.reindex(score.index) >= float(os.getenv('LIQ_MIN_INR', 2e7))
    res_df = pd.DataFrame(dict(d=pd.Timestamp(asof).date(), symbol=score.index, score=score.values,
                               rank_pct=rank_pct.values, is_top=((rank_pct >= 0.90) & liq).values,
                               model_ver=str(getattr(p, '_model_version', ''))))
    with eng.begin() as c:
        c.execute(text("DELETE FROM scores WHERE d=:d"), dict(d=res_df.d.iloc[0]))
    res_df.to_sql('scores', eng, if_exists='append', index=False, method='multi')
    return int(res_df.is_top.sum())

@app.task(name='nse.verify_ic')
def verify_ic(backtest_ic=0.045):
    """Realised open-entry IC for score dates with H sessions of outcomes; rolling-20 alert."""
    with eng.connect() as c:
        sc = pd.read_sql(text("SELECT d,symbol,score FROM scores WHERE d >= current_date - 90"), c)
        px = pd.read_sql(text("SELECT ticker,date,open,close,adj_close FROM nse_stocks WHERE date >= current_date - 120"), c)
    if sc.empty:
        return
    px['date'] = pd.to_datetime(px['date'])
    r = (px.adj_close / px.close).clip(.01, 100).fillna(1)
    px['O'], px['C'] = px.open * r, px.adj_close
    O = px.pivot(index='date', columns='ticker', values='O'); C = px.pivot(index='date', columns='ticker', values='C')
    fwd, _, _ = open_entry_cs_labels(O, C, h=H)
    f = fwd.stack().rename('fwd').reset_index(); f.columns = ['d', 'symbol', 'fwd']
    sc['d'] = pd.to_datetime(sc['d'])
    m = sc.merge(f, on=['d', 'symbol']).dropna()
    ic = fast_daily_rank_ic(m.score, m.d, m.fwd, min_names=100)
    if ic.empty:
        return
    roll = ic.rolling(20, min_periods=10).mean()
    with eng.begin() as c:
        for d, v in ic.items():
            c.execute(text("INSERT INTO ic_monitor(d,ic,n,roll20_ic) VALUES(:d,:v,0,:r) "
                           "ON CONFLICT(d) DO UPDATE SET ic=:v, roll20_ic=:r"),
                      dict(d=d.date(), v=float(v), r=float(roll.get(d, np.nan))))
    rr = roll.dropna()
    if len(rr) >= 10 and (rr.tail(10) < 0.5 * backtest_ic).all():
        log.warning("IC DECAY: rolling-20 IC < half of backtest for 10 consecutive sessions -> retrain")

@app.task(name='nse.eod_chain')
def eod_chain():
    if not _is_trading_day(pd.Timestamp.now(tz='Asia/Kolkata')):
        return
    chain(ingest_bhav.si(), ingest_fii_dii.si(), refresh_feats.si(), score_universe.si(), verify_ic.si())()