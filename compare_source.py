#!/usr/bin/env python3
"""Compare yfinance download vs DB rows fetched by UnifiedStockPredictor._fetch_recent_ticker_data
Reports missing rows, mismatched close/adj_close/volume, duplicate dates.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from MLPredictor import UnifiedStockPredictor
import yfinance as yf
import pandas as pd

TICKER = os.environ.get('TICKER', 'RELIANCE.NS')
LOOKBACK = int(os.environ.get('LOOKBACK', 120))

p = UnifiedStockPredictor()
print('Predictor initialized. Using DB engine:', getattr(p, 'engine', None))

# Fetch DB data
db_df = p._fetch_recent_ticker_data(TICKER, LOOKBACK)
if db_df is None or db_df.empty:
    print('DB: no data returned for', TICKER)
else:
    db_df['date'] = pd.to_datetime(db_df['date'])
    db_df = db_df.sort_values('date').reset_index(drop=True)
    print('DB rows:', len(db_df), 'from', db_df['date'].min(), 'to', db_df['date'].max())

# Fetch yfinance
start = (pd.Timestamp.today() - pd.Timedelta(days=LOOKBACK*2)).strftime('%Y-%m-%d')
end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
print('Downloading yfinance', TICKER, 'start', start, 'end', end)
raw = yf.download(TICKER, start=start, end=end, interval='1d', progress=False, auto_adjust=False, repair=True, threads=False)
if raw is None or raw.empty:
    print('yfinance: no data')
    sys.exit(0)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
yf_df = raw.reset_index().rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'})
yf_df['date'] = pd.to_datetime(yf_df['date'])
print('yfinance rows:', len(yf_df), 'from', yf_df['date'].min(), 'to', yf_df['date'].max())

# Align on dates present in either
if db_df is None or db_df.empty:
    common = []
else:
    db_dates = set(db_df['date'].dt.normalize())
    yf_dates = set(yf_df['date'].dt.normalize())
    only_yf = sorted(list(yf_dates - db_dates))
    only_db = sorted(list(db_dates - yf_dates))
    common = sorted(list(db_dates & yf_dates))
    print('Dates only in yfinance:', len(only_yf))
    print('Dates only in DB:', len(only_db))
    print('Dates common:', len(common))

# Compare common dates for differences
if common:
    diffs = []
    for d in common[-30:]:  # check last 30 common
        yf_row = yf_df[yf_df['date'].dt.normalize() == d].iloc[0]
        db_row = db_df[db_df['date'].dt.normalize() == d].iloc[0]
        row_issues = {}
        for col in ['open','high','low','close','adj_close','volume']:
            yv = float(yf_row.get(col, 0) or 0)
            dv = float(db_row.get(col, 0) or 0)
            if col == 'volume':
                if int(yv) != int(dv):
                    row_issues[col] = (dv, int(yv))
            else:
                if abs(yv - dv) > 1e-6 and round(abs((yv - dv)/max(abs(yv),1e-8)),4) > 0.0001:
                    row_issues[col] = (dv, yv)
        if row_issues:
            diffs.append((d.date(), row_issues))
    print('Mismatched rows in last common dates:', len(diffs))
    for d, issues in diffs[:10]:
        print(d, issues)

# Print last few DB rows and yfinance rows for visual inspection
print('\nLast 5 DB rows:')
if db_df is None or db_df.empty:
    print('  (no DB rows)')
else:
    print(db_df.tail(5).to_string(index=False))

print('\nLast 5 yfinance rows:')
print(yf_df.tail(5).to_string(index=False))

# Check for repeated dates in DB
if db_df is not None and not db_df.empty:
    dup_dates = db_df['date'].dt.normalize().duplicated().sum()
    print('\nDB duplicate date count:', dup_dates)

print('\nDone')
