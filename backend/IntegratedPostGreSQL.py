import yfinance as yf
import pandas as pd
import requests
import io
import time
import logging
import sys
import csv
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from tqdm import tqdm
import numpy as np
from typing import List, Optional, Dict, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- WINDOWS CONSOLE FIX ---
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from dotenv import load_dotenv
import urllib.parse
# Load .env file from the current working directory or parents
load_dotenv()
# Also explicitly try to load from the project root if it exists
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, '.env'))

# --- CONFIGURATION ---
class Config:
    DB_URL = os.getenv("DATABASE_URL")

    NSE_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    
    CHUNK_SIZE = 50
    MAX_WORKERS = 4
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = (5, 10)  # (connect_timeout, read_timeout) in seconds
    
    DB_POOL_SIZE = 10
    DB_MAX_OVERFLOW = 20
    RATE_LIMIT_DELAY = 0.5
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.nseindia.com/',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
    }

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nse_pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NSEDataPipeline:
    def __init__(self, db_url: str = Config.DB_URL):
        if not db_url:
            raise RuntimeError(
                "DB_URL is not set. Set the DB_URL environment variable to your "
                "Postgres connection string, e.g. postgresql://user:password@host:5432/dbname"
            )
        self.engine = create_engine(
            db_url, 
            pool_pre_ping=True, 
            pool_size=Config.DB_POOL_SIZE, 
            max_overflow=Config.DB_MAX_OVERFLOW,
            pool_recycle=3600
        )
        self.session = requests.Session()
        self.session.headers.update(Config.HEADERS)
        self.init_database()
        self.failed_tickers = []
        self.success_count = 0
        self.error_count = 0
        logger.info("NSE Data Pipeline initialized")

    def init_database(self):
        """Initialize database schema"""
        # Attempt to create extension outside of a transaction block
        try:
            with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
        except Exception as e:
            logger.warning(f"TimescaleDB extension not supported: {e}")
            
        with self.engine.begin() as conn:
            # Main table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS nse_stocks (
                    date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    ticker VARCHAR(20) NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    adj_close DOUBLE PRECISION,
                    volume BIGINT,
                    split_factor DOUBLE PRECISION DEFAULT 1.0,
                    delivery_qty BIGINT DEFAULT 0,
                    delivery_percentage DOUBLE PRECISION DEFAULT 0.0,
                    traded_qty BIGINT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker, date)
                );
            """))

        # Attempt to create hypertable in a separate transaction so failures do not crash other table creation
        try:
            with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text("""
                    SELECT create_hypertable('nse_stocks', 'date', 
                        if_not_exists => TRUE,
                        chunk_time_interval => INTERVAL '1 month');
                """))
        except Exception:
            pass 

        with self.engine.begin() as conn:

            # Stock Metadata
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stock_metadata (
                    ticker VARCHAR(20) PRIMARY KEY,
                    company_name VARCHAR(200),
                    isin VARCHAR(20),
                    last_fetched_date DATE,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            
            # Engineered Features Table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS engineered_features (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    adj_close DOUBLE PRECISION,
                    volume BIGINT,
                    returns DOUBLE PRECISION,
                    log_returns DOUBLE PRECISION,
                    rsi_7 DOUBLE PRECISION,
                    rsi_14 DOUBLE PRECISION,
                    rsi_21 DOUBLE PRECISION,
                    macd DOUBLE PRECISION,
                    macd_signal DOUBLE PRECISION,
                    macd_histogram DOUBLE PRECISION,
                    bb_upper DOUBLE PRECISION,
                    bb_middle DOUBLE PRECISION,
                    bb_lower DOUBLE PRECISION,
                    bb_width DOUBLE PRECISION,
                    bb_position DOUBLE PRECISION,
                    atr_14 DOUBLE PRECISION,
                    volatility_2 DOUBLE PRECISION,
                    sma_5 DOUBLE PRECISION,
                    sma_10 DOUBLE PRECISION,
                    sma_20 DOUBLE PRECISION,
                    sma_50 DOUBLE PRECISION,
                    sma_100 DOUBLE PRECISION,
                    sma_200 DOUBLE PRECISION,
                    ema_5 DOUBLE PRECISION,
                    ema_10 DOUBLE PRECISION,
                    ema_12 DOUBLE PRECISION,
                    ema_20 DOUBLE PRECISION,
                    ema_50 DOUBLE PRECISION,
                    ema_100 DOUBLE PRECISION,
                    ema_200 DOUBLE PRECISION,
                    adx DOUBLE PRECISION,
                    stochastic_k DOUBLE PRECISION,
                    stochastic_d DOUBLE PRECISION,
                    cci DOUBLE PRECISION,
                    williams_r DOUBLE PRECISION,
                    obv DOUBLE PRECISION,
                    obv_ema DOUBLE PRECISION,
                    pattern_head_shoulders BOOLEAN,
                    pattern_double_top BOOLEAN,
                    pattern_double_bottom BOOLEAN,
                    pattern_triangle BOOLEAN,
                    pattern_flag BOOLEAN,
                    support_level DOUBLE PRECISION,
                    resistance_level DOUBLE PRECISION,
                    trend_channel DOUBLE PRECISION,
                    volume_sentiment DOUBLE PRECISION,
                    liquidity_score DOUBLE PRECISION,
                    strength_score DOUBLE PRECISION,
                    signal_strength DOUBLE PRECISION,
                    future_return_5d DOUBLE PRECISION,
                    future_direction_5d INTEGER,
                    feature_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (ticker, date)
                );
            """))
            
            # FII/DII Flow Table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fii_dii_flow (
                    date DATE PRIMARY KEY,
                    fii_buy_value DOUBLE PRECISION,
                    fii_sell_value DOUBLE PRECISION,
                    fii_net_value DOUBLE PRECISION,
                    dii_buy_value DOUBLE PRECISION,
                    dii_sell_value DOUBLE PRECISION,
                    dii_net_value DOUBLE PRECISION,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))

            # FO Ban List Table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fo_ban_list (
                    date DATE NOT NULL,
                    ticker VARCHAR(50) NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (date, ticker)
                );
            """))

            # Options Metrics Table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS options_metrics (
                    date DATE NOT NULL,
                    ticker VARCHAR(50) NOT NULL,
                    put_call_ratio DOUBLE PRECISION,
                    iv_near_month DOUBLE PRECISION,
                    iv_next_month DOUBLE PRECISION,
                    iv_term_structure_slope DOUBLE PRECISION,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (date, ticker)
                );
            """))
            
            # Add indexes for performance
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_nse_stocks_ticker_date 
                ON nse_stocks(ticker, date DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_nse_stocks_date 
                ON nse_stocks(date DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_engineered_features_ticker_date 
                ON engineered_features(ticker, date DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_engineered_features_date 
                ON engineered_features(date DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_engineered_features_ticker 
                ON engineered_features(ticker);
            """))

    # --- METHODS REQUIRED BY SCREENER ---
    def get_latest_data(self, limit=None):
        """Fetches the most recent record for every unique ticker."""
        # DISTINCT ON (ticker) ensures we get one row per stock (the latest one)
        query_str = """
            SELECT DISTINCT ON (ticker) * FROM nse_stocks 
            ORDER BY ticker, date DESC
        """
        # FIX (SQL-injection hardening): `limit` was f-string-interpolated directly
        # into the query text. It's internally-sourced today, but any future caller
        # that forwards a UI/API-supplied limit unchanged would inject arbitrary SQL.
        # Bind it as a parameter (and belt-and-suspenders cast to int) instead.
        params = {}
        if limit:
            limit = int(limit)  # raises cleanly on non-numeric input instead of injecting it
            query_str = f"SELECT * FROM ({query_str}) AS sub LIMIT :limit"
            params['limit'] = limit
            
        with self.engine.connect() as conn:
            result = conn.execute(text(query_str), params)
            # Convert SQLAlchemy Rows to list of dicts
            return [dict(row._mapping) for row in result]

    def get_ticker_history(self, ticker):
        """Fetches full history for a specific ticker."""
        query = text("SELECT * FROM nse_stocks WHERE ticker = :ticker ORDER BY date ASC")
        with self.engine.connect() as conn:
            result = conn.execute(query, {"ticker": ticker})
            return [dict(row._mapping) for row in result]

    def get_bulk_latest_features(self, tickers: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Fetch latest engineered features row for multiple (or all) tickers in one query."""
        if tickers:
            query = text("""
                SELECT DISTINCT ON (ticker) *
                FROM engineered_features
                WHERE ticker = ANY(:tickers)
                ORDER BY ticker, date DESC
            """)
            params = {"tickers": tickers}
        else:
            query = text("""
                SELECT DISTINCT ON (ticker) *
                FROM engineered_features
                ORDER BY ticker, date DESC
            """)
            params = {}
            
        with self.engine.connect() as conn:
            result = conn.execute(query, params)
            return {row.ticker: dict(row._mapping) for row in result}

    def get_bulk_ticker_history(self, tickers: Optional[List[str]] = None, limit_per_ticker: int = 250) -> Dict[str, List[Dict]]:
        """Fetch recent history for multiple tickers in one query using a window function."""
        if tickers:
            query = text(f"""
                WITH RankedData AS (
                    SELECT *,
                           ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY date DESC) as rn
                    FROM nse_stocks
                    WHERE ticker = ANY(:tickers)
                )
                SELECT * FROM RankedData WHERE rn <= :limit ORDER BY ticker, date ASC
            """)
            params = {"tickers": tickers, "limit": limit_per_ticker}
        else:
            query = text(f"""
                WITH RankedData AS (
                    SELECT *,
                           ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY date DESC) as rn
                    FROM nse_stocks
                )
                SELECT * FROM RankedData WHERE rn <= :limit ORDER BY ticker, date ASC
            """)
            params = {"limit": limit_per_ticker}
            
        with self.engine.connect() as conn:
            result = conn.execute(query, params)
            
            history_map = {}
            for row in result:
                ticker = row.ticker
                if ticker not in history_map:
                    history_map[ticker] = []
                # Ensure ascending order per ticker since we ordered by ticker, date ASC
                # Need to remove the 'rn' column from the dict if possible, or just leave it
                d = dict(row._mapping)
                d.pop('rn', None)
                history_map[ticker].append(d)
                
            return history_map

    def get_feature_freshness(self) -> Dict[str, datetime]:
        """Get the latest feature calculation date for all tickers to check staleness."""
        query = text("SELECT ticker, MAX(date) as latest_date FROM engineered_features GROUP BY ticker")
        with self.engine.connect() as conn:
            result = conn.execute(query)
            return {row.ticker: row.latest_date for row in result}


    def _detect_trading_gaps(self, df: pd.DataFrame) -> Dict[str, List]:
        """Detect gaps in trading data for each ticker"""
        gaps = {}
        
        if df.empty or 'ticker' not in df.columns or 'date' not in df.columns:
            return gaps
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy().sort_values('date')
            
            if len(ticker_data) < 2:
                continue
            
            # Calculate gaps between consecutive dates
            date_diffs = ticker_data['date'].diff().dt.days
            
            # Normal gaps should be 1-3 days (weekends/holidays)
            # Gaps > 3 days might indicate missing data
            large_gaps = date_diffs[date_diffs > 3]
            
            if not large_gaps.empty:
                gap_dates = ticker_data.loc[large_gaps.index, 'date'].tolist()
                gaps[ticker] = gap_dates
                if len(gap_dates) <= 5:
                    logger.info(f"ℹ️  Trading gap(s) detected for {ticker}: {len(gap_dates)} gap(s)")
        
        return gaps

    def _get_data_quality_metrics(self, df: pd.DataFrame, ticker: str = None) -> Dict:
        """Generate data quality metrics"""
        metrics = {
            'total_records': len(df),
            'unique_tickers': df['ticker'].nunique() if 'ticker' in df.columns else 0,
            'date_range': None,
            'null_counts': {},
            'outliers_detected': 0,
            'validation_status': 'PASS',
            'warnings': []
        }
        
        if df.empty:
            metrics['validation_status'] = 'FAIL'
            metrics['warnings'].append('Empty dataframe')
            return metrics
        
        if 'date' in df.columns:
            metrics['date_range'] = {
                'start': str(df['date'].min()),
                'end': str(df['date'].max())
            }
        
        # Check for nulls
        for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    metrics['null_counts'][col] = null_count
                    if null_count > len(df) * 0.1:  # > 10% nulls
                        metrics['validation_status'] = 'WARNING'
                        metrics['warnings'].append(f'{col}: {null_count} null values ({null_count/len(df)*100:.1f}%)')
        
        # Detect outliers in close price
        if 'close' in df.columns and len(df) > 10:
            close_pct_change = df['close'].pct_change().abs()
            outliers = (close_pct_change > 0.30).sum()  # > 30% change
            metrics['outliers_detected'] = int(outliers)
            if outliers > 0:
                metrics['validation_status'] = 'WARNING'
                metrics['warnings'].append(f'Detected {outliers} extreme price movements (>30%)')
        
        return metrics

    def get_data_quality_report(self, limit_tickers: int = None) -> Dict:
        """Generate comprehensive data quality report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'HEALTHY',
            'total_records': 0,
            'total_tickers': 0,
            'warnings': [],
            'ticker_status': []
        }
        
        try:
            with self.engine.connect() as conn:
                # Get overall stats
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT ticker) as total_tickers,
                        COUNT(DISTINCT DATE(date)) as trading_days,
                        MIN(date) as earliest_date,
                        MAX(date) as latest_date,
                        SUM(CASE WHEN close IS NULL OR close <= 0 THEN 1 ELSE 0 END) as invalid_closes,
                        SUM(CASE WHEN volume IS NULL OR volume < 0 THEN 1 ELSE 0 END) as invalid_volumes
                    FROM nse_stocks
                """))
                
                row = result.fetchone()
                if row:
                    report['total_records'] = row.total_records
                    report['total_tickers'] = row.total_tickers
                    report['trading_days'] = row.trading_days
                    report['date_range'] = {
                        'start': str(row.earliest_date),
                        'end': str(row.latest_date)
                    }
                    
                    if row.invalid_closes > 0:
                        report['overall_status'] = 'WARNING'
                        report['warnings'].append(f'Found {row.invalid_closes} invalid close prices')
                    
                    if row.invalid_volumes > 0:
                        report['overall_status'] = 'WARNING'
                        report['warnings'].append(f'Found {row.invalid_volumes} invalid volumes')
                
                # Get per-ticker stats
                ticker_query = text("""
                    SELECT 
                        ticker,
                        COUNT(*) as record_count,
                        MIN(date) as first_date,
                        MAX(date) as last_date,
                        ROUND(AVG(close), 2) as avg_price,
                        ROUND(AVG(volume), 0) as avg_volume,
                        COUNT(DISTINCT DATE(date)) as trading_days,
                        SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_closes,
                        SUM(CASE WHEN volume = 0 THEN 1 ELSE 0 END) as zero_volumes
                    FROM nse_stocks
                    GROUP BY ticker
                    ORDER BY record_count DESC
                """)
                
                if limit_tickers:
                    ticker_query = text(f"""
                        SELECT 
                            ticker,
                            COUNT(*) as record_count,
                            MIN(date) as first_date,
                            MAX(date) as last_date,
                            ROUND(AVG(close), 2) as avg_price,
                            ROUND(AVG(volume), 0) as avg_volume,
                            COUNT(DISTINCT DATE(date)) as trading_days,
                            SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_closes,
                            SUM(CASE WHEN volume = 0 THEN 1 ELSE 0 END) as zero_volumes
                        FROM nse_stocks
                        GROUP BY ticker
                        ORDER BY record_count DESC
                        LIMIT {limit_tickers}
                    """)
                
                result = conn.execute(ticker_query)
                for row in result:
                    ticker_info = {
                        'ticker': row.ticker,
                        'records': row.record_count,
                        'date_range': f"{row.first_date} to {row.last_date}",
                        'avg_price': row.avg_price,
                        'avg_volume': int(row.avg_volume),
                        'trading_days': row.trading_days,
                        'quality_issues': []
                    }
                    
                    if row.null_closes > 0:
                        ticker_info['quality_issues'].append(f'{row.null_closes} null closes')
                    if row.zero_volumes > 0:
                        ticker_info['quality_issues'].append(f'{row.zero_volumes} zero volumes')
                    
                    report['ticker_status'].append(ticker_info)
        
        except Exception as e:
            logger.error(f"Failed to generate data quality report: {e}")
            report['overall_status'] = 'ERROR'
            report['warnings'].append(str(e))
        
        return report

    def reconcile_data(self, ticker: str = None) -> Dict:
        """Reconcile and fix data quality issues"""
        reconciliation_report = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'issues_found': 0,
            'issues_fixed': 0,
            'details': []
        }
        
        try:
            with self.engine.begin() as conn:
                if ticker:
                    # Reconcile specific ticker
                    query = text("""
                        SELECT ticker, date, open, high, low, close, adj_close, volume
                        FROM nse_stocks
                        WHERE ticker = :ticker
                        ORDER BY date
                    """)
                    result = conn.execute(query, {'ticker': ticker})
                else:
                    # Check for duplicate records
                    dup_query = text("""
                        SELECT ticker, date, COUNT(*) as cnt
                        FROM nse_stocks
                        GROUP BY ticker, date
                        HAVING COUNT(*) > 1
                    """)
                    result = conn.execute(dup_query)
                    dups = list(result)
                    
                    if dups:
                        reconciliation_report['issues_found'] = len(dups)
                        reconciliation_report['details'].append(f"Found {len(dups)} duplicate records")
                        
                        # Remove duplicates (keep latest)
                        for dup in dups[:10]:  # Fix first 10
                            delete_query = text("""
                                DELETE FROM nse_stocks
                                WHERE ticker = :ticker AND date = :date AND
                                      ctid NOT IN (SELECT ctid FROM nse_stocks 
                                                   WHERE ticker = :ticker AND date = :date
                                                   ORDER BY updated_at DESC LIMIT 1)
                            """)
                            conn.execute(delete_query, {'ticker': dup.ticker, 'date': dup.date})
                            reconciliation_report['issues_fixed'] += 1
                
                # Check for null prices
                null_query = text("""
                    SELECT ticker, COUNT(*) as null_count
                    FROM nse_stocks
                    WHERE close IS NULL OR close <= 0
                    GROUP BY ticker
                """)
                result = conn.execute(null_query)
                null_records = list(result)
                
                if null_records:
                    for record in null_records[:5]:
                        reconciliation_report['details'].append(
                            f"Ticker {record.ticker}: {record.null_count} records with null/invalid close"
                        )
                        reconciliation_report['issues_found'] += record.null_count
                
                # Check data consistency
                consistency_query = text("""
                    SELECT ticker, COUNT(*) as inconsistent_count
                    FROM nse_stocks
                    WHERE high < low OR high < close OR low > close
                    GROUP BY ticker
                """)
                result = conn.execute(consistency_query)
                consistency_issues = list(result)
                
                if consistency_issues:
                    for issue in consistency_issues[:5]:
                        reconciliation_report['details'].append(
                            f"Ticker {issue.ticker}: {issue.inconsistent_count} OHLC consistency issues"
                        )
                        reconciliation_report['issues_found'] += issue.inconsistent_count
                        
                        # Fix consistency issues
                        fix_query = text("""
                            UPDATE nse_stocks
                            SET high = GREATEST(open, high, low, close),
                                low = LEAST(open, high, low, close)
                            WHERE ticker = :ticker AND (high < low OR high < close OR low > close)
                        """)
                        conn.execute(fix_query, {'ticker': issue.ticker})
                        reconciliation_report['issues_fixed'] += issue.inconsistent_count
        
        except Exception as e:
            logger.error(f"Data reconciliation error: {e}")
            reconciliation_report['error'] = str(e)
        
        return reconciliation_report

    def store_engineered_features(self, ticker: str, df_engineered: pd.DataFrame):
        """
        Store engineered features in the engineered_features table with Schema Evolution.
        Automatically adds missing columns to the table.
        """
        try:
            if df_engineered.empty:
                logger.warning(f"Empty engineered features for {ticker}")
                return
            
            # Prepare data
            df_to_store = df_engineered.copy()
            
            # Ensure date column is datetime and timezone-naive
            if 'date' in df_to_store.columns:
                df_to_store['date'] = pd.to_datetime(df_to_store['date'])
                if hasattr(df_to_store['date'].dtype, 'tz') and df_to_store['date'].dt.tz is not None:
                    df_to_store['date'] = df_to_store['date'].dt.tz_localize(None)
            
            # Add metadata columns if missing
            if 'ticker' not in df_to_store.columns:
                df_to_store['ticker'] = ticker
            
            df_to_store['feature_count'] = len(df_to_store.columns)
            df_to_store['updated_at'] = datetime.now()

            # FIX: sanitize column names ONCE, up front, and rename the dataframe
            # to match. Previously ALTER TABLE used a sanitized `safe_col` name
            # while the later INSERT/COPY/SET-clause used the original, unsanitized
            # `col` -- if a feature column ever needed sanitizing (any char outside
            # [A-Za-z0-9_]), the table would get a column named `safe_col` but the
            # INSERT would try to write to `col`, which doesn't exist -> runtime
            # SQL error. Renaming here means every downstream reference uses the
            # same (already-safe) name, so they can never diverge.
            rename_map = {}
            for c in df_to_store.columns:
                safe_c = "".join(ch for ch in c if ch.isalnum() or ch == '_')
                if safe_c and safe_c != c:
                    rename_map[c] = safe_c
            if rename_map:
                logger.warning(f"Sanitizing {len(rename_map)} column name(s) for SQL safety: {rename_map}")
                df_to_store = df_to_store.rename(columns=rename_map)
            
            # --- SCHEMA EVOLUTION ---
            # 1. Get existing columns in DB
            with self.engine.connect() as conn:
                # Get current table columns
                result = conn.execute(text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'engineered_features'
                """))
                existing_cols = {row[0]: row[1] for row in result}
            
            # 2. Identify missing columns in DF
            df_cols = [c for c in df_to_store.columns if c not in existing_cols]
            
            # 3. Add missing columns
            if df_cols:
                logger.info(f"⚡ Schema Evolution: Adding {len(df_cols)} new columns to engineered_features")
                with self.engine.begin() as conn:
                    for col in df_cols:
                        # Map pandas types to SQL types
                        dtype = df_to_store[col].dtype
                        if pd.api.types.is_integer_dtype(dtype):
                            sql_type = "BIGINT"
                        elif pd.api.types.is_float_dtype(dtype):
                            sql_type = "DOUBLE PRECISION"
                        elif pd.api.types.is_datetime64_any_dtype(dtype):
                            sql_type = "TIMESTAMP WITHOUT TIME ZONE"
                        elif pd.api.types.is_bool_dtype(dtype):
                            sql_type = "BOOLEAN"
                        else:
                            sql_type = "TEXT"
                            
                        # col is already sanitized (renamed up front) -- guaranteed
                        # non-empty and SQL-safe, so no re-sanitization/mismatch risk here.
                        conn.execute(text(f'ALTER TABLE engineered_features ADD COLUMN IF NOT EXISTS "{col}" {sql_type}'))
                        # Update our local knowledge
                        existing_cols[col] = sql_type

            # --- DATA STORAGE ---
            # Get valid columns again to be safe
            valid_cols = list(existing_cols.keys())
            cols_to_insert = [col for col in df_to_store.columns if col in valid_cols]
            
            if not cols_to_insert:
                return

            df_to_insert = df_to_store[cols_to_insert].copy()
            
            # Handle boolean conversion for Postgres
            for col in df_to_insert.columns:
                if df_to_insert[col].dtype == 'bool':
                    df_to_insert[col] = df_to_insert[col].astype(bool)
            
            raw_conn = self.engine.raw_connection()
            try:
                cursor = raw_conn.cursor()
                
                # Create temp table
                cursor.execute("DROP TABLE IF EXISTS temp_ef_upload")
                cols_def = ', '.join([f'"{col}" TEXT' for col in cols_to_insert])
                cursor.execute(f"CREATE TEMP TABLE temp_ef_upload ({cols_def}) ON COMMIT DROP")
                
                # COPY data
                csv_buffer = io.StringIO()
                df_to_insert.to_csv(csv_buffer, sep='\t', header=False, index=False, na_rep='', quoting=csv.QUOTE_NONE)
                csv_buffer.seek(0)
                
                cols_names = ', '.join([f'"{col}"' for col in cols_to_insert])
                cursor.copy_expert(f"COPY temp_ef_upload ({cols_names}) FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t')", csv_buffer)
                
                # Generate SET clause for UPDATE
                # We exclude primary keys (ticker, date) and created_at from update
                update_cols = [c for c in cols_to_insert if c not in ('ticker', 'date', 'created_at')]
                set_clause = ', '.join([f'"{col}" = CAST(EXCLUDED."{col}" AS {existing_cols.get(col, "TEXT")})' for col in update_cols])
                
                # Build casting for SELECT
                select_casts = []
                for col in cols_to_insert:
                    sql_type = existing_cols.get(col, "TEXT")
                    select_casts.append(f'CAST("{col}" AS {sql_type})')
                select_clause = ', '.join(select_casts)

                # INSERT ... ON CONFLICT DO UPDATE
                query = f"""
                    INSERT INTO engineered_features ({cols_names})
                    SELECT {select_clause} FROM temp_ef_upload
                    ON CONFLICT (ticker, date) 
                    DO UPDATE SET
                        {set_clause},
                        updated_at = CURRENT_TIMESTAMP
                """
                cursor.execute(query)
                
                raw_conn.commit()
                logger.info(f"✅ Stored {len(df_to_insert)} rows of features for {ticker}")
                
            except Exception as e:
                raw_conn.rollback()
                logger.error(f"Failed to copy features: {e}")
                raise
            finally:
                cursor.close()
                raw_conn.close()

        except Exception as e:
            logger.error(f"Error storing engineered features for {ticker}: {e}")
            raise
    # ------------------------------------

    def get_all_nse_symbols(self) -> List[str]:
        """Fetch NSE symbol list with retry logic"""
        logger.info("Fetching NSE symbol list...")
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.session.get(
                    Config.NSE_LIST_URL, 
                    timeout=Config.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text))
                df.columns = [c.strip() for c in df.columns]
                
                # Upsert Metadata
                metadata_df = df[['SYMBOL', 'NAME OF COMPANY', 'ISIN NUMBER']].copy()
                metadata_df.columns = ['ticker', 'company_name', 'isin']
                metadata_df = metadata_df[metadata_df['ticker'].str.len() > 0]
                metadata_df = metadata_df.drop_duplicates(subset=['ticker'])
                
                with self.engine.begin() as conn:
                    params = [{'t': row['ticker'], 'c': row['company_name'], 'i': row['isin']} for _, row in metadata_df.iterrows()]
                    if params:
                        conn.execute(text("""
                            INSERT INTO stock_metadata (ticker, company_name, isin, last_updated)
                            VALUES (:t, :c, :i, CURRENT_TIMESTAMP)
                            ON CONFLICT (ticker) DO UPDATE SET 
                                company_name = EXCLUDED.company_name,
                                isin = EXCLUDED.isin,
                                is_active = TRUE,
                                last_updated = CURRENT_TIMESTAMP
                        """), params)
                
                symbols = [f"{s}.NS" for s in metadata_df['ticker'].unique()]
                logger.info(f"Found {len(symbols)} active NSE symbols")
                return symbols
                
            except KeyboardInterrupt:
                logger.warning("NSE fetch interrupted by user, falling back to database...")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{Config.MAX_RETRIES} failed: {e}")
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(min(0.1 * (attempt + 1), 0.5))
                else:
                    logger.error(f"Failed to fetch NSE list after {Config.MAX_RETRIES} attempts")
        
        # --- DATABASE FALLBACK ---
        # If NSE website is unreachable/blocked, use existing tickers from database
        return self._get_symbols_from_db()

    def fetch_and_store_fii_dii_flow(self) -> bool:
        """
        FIX (real gap found in review, not a bug in existing code): the
        `fii_dii_flow`, `options_metrics`, and `fo_ban_list` tables are created
        by init_database() but NOTHING in this file ever wrote to them. Every
        consumer downstream already handles that correctly and honestly —
        AdvancedFeatureEngine.py's `_fno_features`/`_short_interest_features`
        fall back to neutral, non-fabricated placeholder values (0.0 net flow,
        1.0 balanced PCR, etc.) exactly as documented there, and MLPredictor.py
        tracks `fno_data_available`/`short_interest_data_available` flags so
        nothing downstream mistakes a placeholder for a real reading. So this
        was never silently wrong — it was silently EMPTY: the model has simply
        never seen a single real value for what its own feature-engineering
        docstrings describe as a Pillar-2/3 alpha source (institutional
        FII/DII flow is one of the most-watched sentiment signals in Indian
        equities). This fills that gap for `fii_dii_flow` — the one of the
        three with a stable, well-known public source — via NSE's FII/DII
        JSON widget endpoint (the same one used by several open-source NSE
        wrapper libraries, e.g. nsepython's `nse_fiidii`).

        Important limitations (documented rather than hidden):
        - This endpoint serves RECENT (typically last ~1-2 trading days)
          provisional data for the live widget, not a multi-year history. It
          cannot retroactively backfill years of training history — running
          this daily is what builds that history GOING FORWARD. Historical
          backfill would need NSE's separate archive reports
          (nseindia.com/reports/fii-dii); that's a distinct, less stable
          scrape target and is intentionally left out of this best-effort
          pass rather than guessed at.
        - NSE's API schema/anti-bot behavior changes without notice and is
          not covered by this repo's test suite (no network access to
          nseindia.com from an automated sandbox). If NSE changes the JSON
          shape, this fails closed (logs a warning, writes nothing, the
          existing neutral-placeholder fallback keeps working exactly as
          before) — it can degrade to "still empty" but can never corrupt
          `fii_dii_flow` with malformed rows or crash the pipeline.
        - NSE's `/api/*` endpoints require a warmed session (cookies from an
          initial homepage GET) or they typically 401/403 — mirrored below.
        """
        payload = None
        for attempt in range(3):
            try:
                # Use a clean session with minimal headers to bypass NSE anti-bot
                temp_session = requests.Session()
                temp_session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                    'Accept': 'application/json, text/plain, */*'
                })
                
                # Warmup (ignore errors here, as cookies might still be set even on 403)
                try:
                    temp_session.get("https://www.nseindia.com", timeout=Config.REQUEST_TIMEOUT)
                except Exception:
                    pass
                    
                resp = temp_session.get(
                    "https://www.nseindia.com/api/fiidiiTradeReact",
                    timeout=Config.REQUEST_TIMEOUT
                )
                resp.raise_for_status()
                
                if not resp.text.strip():
                    raise ValueError("Empty response received from NSE")
                    
                payload = resp.json()
                break  # Success
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"FII/DII fetch skipped (endpoint unreachable or schema changed): {e}")
                    return False
                time.sleep(2)

        if not isinstance(payload, list) or not payload:
            logger.warning("FII/DII fetch returned no rows (empty/unexpected payload) — skipping.")
            return False

        try:
            df = pd.DataFrame(payload)
            # Tolerate the couple of key-casing variants NSE has shipped historically
            # (documented by community wrappers) instead of hard-failing on one.
            rename_map = {
                'category': 'category', 'Category': 'category',
                'date': 'date', 'Date': 'date',
                'buyValue': 'buy_value', 'Buy_Value': 'buy_value',
                'sellValue': 'sell_value', 'Sell_Value': 'sell_value',
                'netValue': 'net_value', 'Net_Value': 'net_value',
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            required = {'category', 'date', 'net_value'}
            if not required.issubset(df.columns):
                logger.warning(f"FII/DII payload missing expected columns {required - set(df.columns)} "
                                f"(NSE schema likely changed) — skipping this run, leaving table untouched.")
                return False

            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['date'])
            df['buy_value'] = pd.to_numeric(df.get('buy_value'), errors='coerce')
            df['sell_value'] = pd.to_numeric(df.get('sell_value'), errors='coerce')
            df['net_value'] = pd.to_numeric(df.get('net_value'), errors='coerce')

            df['category_norm'] = df['category'].astype(str).str.upper().str.strip()
            fii = df[df['category_norm'].str.startswith('FII') | df['category_norm'].str.startswith('FPI')]
            dii = df[df['category_norm'].str.startswith('DII')]

            by_date = {}
            for _, r in fii.iterrows():
                d = r['date'].date()
                by_date.setdefault(d, {})
                by_date[d].update({
                    'fii_buy_value': r['buy_value'], 'fii_sell_value': r['sell_value'],
                    'fii_net_value': r['net_value'],
                })
            for _, r in dii.iterrows():
                d = r['date'].date()
                by_date.setdefault(d, {})
                by_date[d].update({
                    'dii_buy_value': r['buy_value'], 'dii_sell_value': r['sell_value'],
                    'dii_net_value': r['net_value'],
                })

            if not by_date:
                logger.warning("FII/DII payload parsed but no FII/DII category rows matched — skipping.")
                return False

            _val_keys = ('fii_buy_value', 'fii_sell_value', 'fii_net_value',
                         'dii_buy_value', 'dii_sell_value', 'dii_net_value')
            params = [{'d': d, **{k: vals.get(k) for k in _val_keys}} for d, vals in by_date.items()]
            with self.engine.begin() as conn:
                for p in params:
                    conn.execute(text("""
                        INSERT INTO fii_dii_flow (date, fii_buy_value, fii_sell_value, fii_net_value,
                                                   dii_buy_value, dii_sell_value, dii_net_value, updated_at)
                        VALUES (:d, :fii_buy_value, :fii_sell_value, :fii_net_value,
                                :dii_buy_value, :dii_sell_value, :dii_net_value, CURRENT_TIMESTAMP)
                        ON CONFLICT (date) DO UPDATE SET
                            fii_buy_value = COALESCE(EXCLUDED.fii_buy_value, fii_dii_flow.fii_buy_value),
                            fii_sell_value = COALESCE(EXCLUDED.fii_sell_value, fii_dii_flow.fii_sell_value),
                            fii_net_value = COALESCE(EXCLUDED.fii_net_value, fii_dii_flow.fii_net_value),
                            dii_buy_value = COALESCE(EXCLUDED.dii_buy_value, fii_dii_flow.dii_buy_value),
                            dii_sell_value = COALESCE(EXCLUDED.dii_sell_value, fii_dii_flow.dii_sell_value),
                            dii_net_value = COALESCE(EXCLUDED.dii_net_value, fii_dii_flow.dii_net_value),
                            updated_at = CURRENT_TIMESTAMP
                    """), p)
            logger.info(f"✅ FII/DII flow: upserted {len(params)} date row(s)")
            return True
        except Exception as e:
            logger.warning(f"FII/DII parse/store failed (table left untouched): {e}")
            return False

    def _get_symbols_from_db(self) -> List[str]:
        """Fallback: fetch existing tickers from nse_stocks table when NSE website is unreachable"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT DISTINCT ticker FROM nse_stocks WHERE ticker IS NOT NULL ORDER BY ticker"
                ))
                tickers = [f"{row[0]}.NS" for row in result]
                if tickers:
                    logger.info(f"Database fallback: loaded {len(tickers)} existing tickers")
                else:
                    logger.warning("No tickers found in database fallback - database may be empty")
                return tickers
        except Exception as e:
            logger.error(f"Database fallback also failed: {e}")
            return []

    def get_ticker_date_ranges(self, tickers: List[str]) -> Dict[str, datetime]:
        """Get last fetch date for each ticker"""
        clean_tickers = set(t.replace('.NS', '') for t in tickers if t)
        ticker_map = {}

        if not clean_tickers:
            logger.info("Found existing data for 0 tickers")
            return ticker_map

        try:
            with self.engine.connect() as conn:
                # Simple query with no parameterization - fetch all, filter in Python
                result = conn.execute(text(
                    "SELECT ticker, MAX(date) as last_date FROM nse_stocks GROUP BY ticker"
                ))
                for row in result:
                    if row.ticker in clean_tickers and row.last_date:
                        ticker_map[row.ticker] = row.last_date
        except Exception as e:
            logger.warning(f"Could not fetch ticker date ranges: {e}")
        
        logger.info(f"Found existing data for {len(ticker_map)} tickers")
        return ticker_map

    def download_batch(self, batch_tickers: List[str], period: str = None, 
                      start=None, end=None) -> pd.DataFrame:
        """Download stock data with improved error handling and validation"""
        try:
            # Removed unconditional sleep for rate limiting as yfinance handles this or it should be managed at the caller level.
            
            # auto_adjust=False ensures we get BOTH Close and Adj Close columns.
            if start and end:
                data = yf.download(
                    batch_tickers, start=start, end=end, interval='1d', 
                    progress=False, group_by='ticker', auto_adjust=False, 
                    threads=False, repair=True, actions=True 
                )
            else:
                p = period if period else '5y'
                data = yf.download(
                    batch_tickers, period=p, interval='1d', 
                    progress=False, group_by='ticker', auto_adjust=False, 
                    threads=False, repair=True, actions=True 
                )

            if data.empty:
                logger.warning(f"No data returned for batch: {', '.join(batch_tickers)}")
                return pd.DataFrame()

            # ── FIX: numpy 2.x + yfinance compatibility ──
            # yfinance returns read-only numpy arrays causing
            # ValueError('output array is read-only') on operations.
            # Deep-copy the DataFrame to make all arrays writable.
            data = data.copy(deep=True)
            for col in data.columns:
                if hasattr(data[col].values, 'flags') and not data[col].values.flags.writeable:
                    data[col] = data[col].values.copy()

            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data = data.stack(level=0, future_stack=True)
                data.index.names = ['Date', 'ticker']
                data = data.reset_index()
            else:
                data = data.reset_index()
                if len(batch_tickers) == 1:
                    data['ticker'] = batch_tickers[0]

            # Second copy after reshaping to ensure all derived columns are writable
            data = data.copy(deep=True)

            logger.info(f"✓ Downloaded data for {data['ticker'].nunique() if 'ticker' in data.columns else 0} tickers")
            return data
            
        except Exception as e:
            error_msg = str(e)
            if "No data found" not in error_msg and "No timezone found" not in error_msg:
                logger.warning(f"Batch download issue: {error_msg[:100]}")
            return pd.DataFrame()

    def _validate_price_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced validation for price consistency and outliers"""
        if df.empty:
            return df
        
        before_count = len(df)
        
        # 1. Basic sanity checks
        df = df[
            (df['close'] > 0) &
            np.isfinite(df['close']) &
            (df['high'] >= df['low']) &
            (df['high'] >= df['close']) &
            (df['high'] >= df['open']) &
            (df['low'] <= df['close']) &
            (df['low'] <= df['open']) &
            (df['open'] > 0) &
            (df['high'] > 0) &
            (df['low'] > 0) &
            np.isfinite(df['open']) &
            np.isfinite(df['high']) &
            np.isfinite(df['low'])
        ].copy()
        
        # 2. Detect impossible price gaps (stock split candidates) FIRST, so the
        # outlier filter below can exempt them. A real 1:2 split is a ~50%
        # single-day move -- 15-25 sigma vs. a 20-day window -- which would
        # otherwise get deleted as a "bad tick" by the z-score filter, corrupting
        # exactly the raw close/adj_close history the split-adjustment fix in
        # AdvancedFeatureEngine.py depends on to compute a correct ratio.
        if len(df) > 0:
            df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
            price_change_pct = df.groupby('ticker')['close'].pct_change().abs() * 100
            split_candidate = price_change_pct > 30
            n_split = int(split_candidate.sum())
            if n_split > 0:
                for ticker in df.loc[split_candidate, 'ticker'].unique():
                    logger.warning(f"⚠️  Potential stock split detection for {ticker} - large price movement detected")

        # 3. Detect and handle extreme outliers (likely data errors)
        # FIX: this block computed rolling mean/std per ticker for exactly this
        # purpose, then discarded them and marked every row valid unconditionally
        # -- the ">5 std deviations" outlier flagging described in the comment
        # never actually ran. Wire it up: compute a z-score per column and drop
        # rows where any price column is a >5-sigma outlier vs. its own trailing
        # 20-day window (causal -- uses only past/current data, no look-ahead),
        # excluding rows already identified as split candidates above.
        if len(df) > 0:
            outlier_mask = pd.Series(False, index=df.index)
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    grp = df.groupby('ticker')[col]
                    roll_mean = grp.transform(lambda s: s.rolling(window=20, min_periods=5).mean())
                    roll_std = grp.transform(lambda s: s.rolling(window=20, min_periods=5).std())
                    with np.errstate(divide='ignore', invalid='ignore'):
                        z = (df[col] - roll_mean).abs() / roll_std.replace(0, np.nan)
                    col_outliers = (z > 5).fillna(False)
                    outlier_mask |= col_outliers
                    df['_valid_' + col] = ~col_outliers
            outlier_mask &= ~split_candidate  # never drop a plausible split/bonus day
            n_outliers = int(outlier_mask.sum())
            if n_outliers > 0:
                logger.warning(f"Removed {n_outliers} rows with a >5-sigma price outlier "
                                f"(vs. trailing 20-day mean/std, excluding split candidates) — likely bad ticks")
                df = df[~outlier_mask].copy()
        
        removed = before_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} invalid records due to price validation")
        
        return df

    def _validate_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced volume validation"""
        if df.empty:
            return df
        
        before_count = len(df)
        
        # 1. Volume must be non-negative
        df['volume'] = df['volume'].clip(lower=0)
        
        # 2. Detect abnormal volume spikes (> 5x median)
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            if len(ticker_data) > 20:
                median_vol = ticker_data['volume'].median()
                if median_vol > 0:
                    max_expected = median_vol * 5
                    spike_rows = ticker_data[ticker_data['volume'] > max_expected]
                    if not spike_rows.empty:
                        logger.debug(f"Volume spike(s) detected for {ticker}: {len(spike_rows)} records")
        
        # 3. Handle zero volume
        zero_vol_rows = len(df[df['volume'] == 0])
        if zero_vol_rows > 0:
            logger.warning(f"Found {zero_vol_rows} records with zero volume - keeping for completeness")
        
        return df

    def _validate_adj_close(self, df: pd.DataFrame) -> pd.DataFrame:
        """Improved adj_close handling"""
        if df.empty:
            return df
        
        if 'adj_close' not in df.columns or 'close' not in df.columns:
            return df
        
        # 1. Fill missing adj_close with close
        missing_adj = df[df['adj_close'].isna()]
        if not missing_adj.empty:
            logger.info(f"Filling {len(missing_adj)} missing adj_close values with close price")
            df.loc[df['adj_close'].isna(), 'adj_close'] = df.loc[df['adj_close'].isna(), 'close']
        
        # 2. Validate adj_close is positive and finite
        # Note: Previous check against high/low was incorrect as pre-split adj_close will be lower than raw low
        invalid_adj = df[(df['adj_close'] <= 0) | (~np.isfinite(df['adj_close']))]
        if not invalid_adj.empty:
            logger.warning(f"Correcting {len(invalid_adj)} invalid adj_close values")
            df.loc[invalid_adj.index, 'adj_close'] = df.loc[invalid_adj.index, 'close']
        
        return df

    def _calculate_split_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate split factor with better handling"""
        if df.empty:
            return df
        
        if 'close' not in df.columns or 'adj_close' not in df.columns:
            df['split_factor'] = 1.0
            return df
        
        with np.errstate(divide='ignore', invalid='ignore'):
            df['split_factor'] = np.where(
                (df['adj_close'] > 0) & (df['close'] > 0) & 
                np.isfinite(df['adj_close']) & np.isfinite(df['close']),
                (df['close'] / df['adj_close']).round(6),
                1.0
            )
        
        # Validate split factor
        df.loc[df['split_factor'] <= 0, 'split_factor'] = 1.0
        df.loc[df['split_factor'] > 100, 'split_factor'] = 1.0  # Reduced threshold
        df.loc[~np.isfinite(df['split_factor']), 'split_factor'] = 1.0
        
        # Log significant splits
        significant_splits = df[(df['split_factor'] != 1.0) & (df['split_factor'] > 1.01)]
        if not significant_splits.empty:
            for ticker in significant_splits['ticker'].unique():
                splits = significant_splits[significant_splits['ticker'] == ticker]
                logger.info(f"Stock split adjustments found for {ticker}: {len(splits)} dates")
        
        return df

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate dataframe with enhanced error handling"""
        if df.empty: 
            return df
        
        try:
            # Standardize columns
            df.columns = [str(c).lower().strip() for c in df.columns]
            
            rename_map = {
                'date': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 
                'close': 'close', 'adj close': 'adj_close', 'volume': 'volume', 
                'ticker': 'ticker', 'dividends': 'dividends', 'stock splits': 'stock_splits'
            }
            df = df.rename(columns=rename_map)
            
            # Clean Ticker
            df['ticker'] = df['ticker'].astype(str).str.replace('.NS', '', regex=False).str.strip()
            
            # Handle Date with timezone awareness
            if df['date'].dtype == 'object':
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Remove timezone if present
            if hasattr(df['date'].dtype, 'tz') and df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            elif df['date'].dtype != 'datetime64[ns]':
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Remove rows with invalid dates
            df = df.dropna(subset=['date'])
            
            if df.empty:
                return df
            
            # Numeric conversion for price columns
            price_cols = ['open', 'high', 'low', 'close', 'adj_close']
            for col in price_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Keep negative values for now - will validate later
                    df.loc[df[col] < 0, col] = np.nan
            
            # Ensure adj_close exists
            if 'adj_close' not in df.columns or df['adj_close'].isna().all():
                if 'close' in df.columns:
                    logger.warning("adj_close column missing or all NaN! Using close as fallback")
                    df['adj_close'] = df['close']
                else:
                    logger.error("Neither adj_close nor close column found!")
                    return pd.DataFrame()

            # Integer columns
            int_cols = ['volume', 'delivery_qty', 'traded_qty']
            for col in int_cols:
                if col not in df.columns:
                    df[col] = 0
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    df[col] = df[col].fillna(0)
                    df[col] = df[col].round(0)
                    df[col] = df[col].astype('int64')

            # Add missing columns
            if 'delivery_percentage' not in df.columns:
                df['delivery_percentage'] = 0.0

            # --- ENHANCED VALIDATION PIPELINE ---
            # 1. Validate adj_close
            df = self._validate_adj_close(df)
            
            # 2. Calculate split factor
            df = self._calculate_split_factor(df)
            
            # 3. Validate price logic
            df = self._validate_price_logic(df)
            
            # 4. Validate volume
            df = self._validate_volume(df)
            
            # Remove duplicates (keep last occurrence)
            df = df.drop_duplicates(subset=['ticker', 'date'], keep='last')
            
            # Final data check
            if df.empty:
                logger.warning("All records filtered out during validation")
                return df
            
            required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 
                            'adj_close', 'volume', 'split_factor', 'delivery_qty', 
                            'delivery_percentage', 'traded_qty']
            
            df['updated_at'] = datetime.now()
            
            # Log validation summary
            logger.info(f"✓ Processed {len(df)} valid records from yfinance")
            
            return df[required_cols + ['updated_at']]
            
        except Exception as e:
            logger.error(f"Error processing dataframe: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _validate_stock_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Validate stock data before database insertion."""
        if df is None or df.empty:
            return df
        
        original_len = len(df)
        
        # 1. Remove duplicate timestamps
        date_col = 'date' if 'date' in df.columns else df.index.name
        if date_col and date_col in df.columns:
            dupes = df.duplicated(subset=[date_col], keep='last')
            if dupes.any():
                logger.warning(f"[{ticker}] Removed {dupes.sum()} duplicate timestamps")
                df = df[~dupes]
        
        # 2. Flag zero-volume days
        if 'volume' in df.columns:
            zero_vol = (df['volume'] == 0) | df['volume'].isna()
            if zero_vol.any():
                logger.warning(f"[{ticker}] {zero_vol.sum()} zero-volume days detected")
        
        # 3. Check price continuity (flag >20% single-day moves)
        if 'close' in df.columns and len(df) > 1:
            pct_change = df['close'].pct_change().abs()
            extreme = pct_change > 0.20
            if extreme.any():
                logger.warning(f"[{ticker}] {extreme.sum()} extreme price moves (>20%) detected - possible corporate action")
        
        if len(df) < original_len:
            logger.info(f"[{ticker}] Data validation: {original_len} -> {len(df)} rows")
        
        return df

    def fast_bulk_upsert(self, df: pd.DataFrame):
        """Optimized bulk upsert"""
        if df.empty: 
            return
            
        ticker_val = df['ticker'].iloc[0] if 'ticker' in df.columns else 'BULK'
        df = self._validate_stock_data(df, ticker_val)

        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N', 
                 quoting=csv.QUOTE_NONE, escapechar='\\')
        output.seek(0)

        raw_conn = self.engine.raw_connection()
        try:
            cursor = raw_conn.cursor()
            
            cursor.execute("DROP TABLE IF EXISTS temp_stock_upload")
            cursor.execute("""
                CREATE TEMP TABLE temp_stock_upload (
                    date TIMESTAMP, ticker VARCHAR, open FLOAT, high FLOAT, low FLOAT, 
                    close FLOAT, adj_close FLOAT, volume BIGINT, split_factor FLOAT, 
                    delivery_qty BIGINT, delivery_percentage FLOAT, traded_qty BIGINT,
                    updated_at TIMESTAMP
                ) ON COMMIT DROP
            """)
            
            cursor.copy_expert(
                "COPY temp_stock_upload FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t', NULL '\\N')", 
                output
            )
            
            cursor.execute("""
                INSERT INTO nse_stocks (
                    date, ticker, open, high, low, close, adj_close, volume, 
                    split_factor, delivery_qty, delivery_percentage, traded_qty, updated_at
                )
                SELECT * FROM temp_stock_upload
                ON CONFLICT (ticker, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    adj_close = EXCLUDED.adj_close,
                    volume = EXCLUDED.volume,
                    split_factor = EXCLUDED.split_factor,
                    delivery_qty = EXCLUDED.delivery_qty,
                    delivery_percentage = EXCLUDED.delivery_percentage,
                    traded_qty = EXCLUDED.traded_qty,
                    updated_at = CURRENT_TIMESTAMP;
            """)
            
            # Update metadata (compatible with existing schema)
            cursor.execute("""
                INSERT INTO stock_metadata (ticker, last_fetched_date, last_updated)
                SELECT ticker, MAX(date)::date, CURRENT_TIMESTAMP
                FROM temp_stock_upload
                GROUP BY ticker
                ON CONFLICT (ticker) DO UPDATE SET
                    last_fetched_date = EXCLUDED.last_fetched_date,
                    last_updated = CURRENT_TIMESTAMP;
            """)
            
            raw_conn.commit()
            rows_inserted = len(df)
            self.success_count += rows_inserted
            
        except Exception as e:
            raw_conn.rollback()
            self.error_count += 1
            logger.error(f"Bulk insert failed: {e}")
            raise
        finally:
            cursor.close()
            raw_conn.close()

    def run_pipeline(self, symbols: List[str], force_full_refresh: bool = False):
        """Main pipeline with progress tracking, graceful shutdown, and retry logic.
        
        Fixes spurious KeyboardInterrupt on Windows caused by worker-thread
        network errors (connection resets, SSL timeouts, yfinance rate-limits)
        propagating through ThreadPoolExecutor.as_completed().
        """
        logger.info(f"🚀 Starting pipeline for {len(symbols)} symbols")

        # FIX (real gap — see fetch_and_store_fii_dii_flow docstring): exchange-
        # wide, so fetch once per pipeline run, not once per ticker. Best-effort:
        # any failure here must never block the price-data pipeline that follows.
        try:
            self.fetch_and_store_fii_dii_flow()
        except Exception as e:
            logger.warning(f"FII/DII flow step failed, continuing without it: {e}")

        last_dates = self.get_ticker_date_ranges(symbols)
        today = datetime.now()
        
        full_download_tickers = []
        incremental_tickers = []
        
        for sym in symbols:
            clean_sym = sym.replace('.NS', '')
            
            if force_full_refresh or clean_sym not in last_dates:
                full_download_tickers.append(sym)
            else:
                last_date = last_dates[clean_sym]
                days_since_update = (today - last_date).days
                
                if days_since_update > 1:
                    incremental_tickers.append(sym)
        
        tasks = []
        
        # Full Downloads
        for i in range(0, len(full_download_tickers), Config.CHUNK_SIZE):
            batch = full_download_tickers[i:i + Config.CHUNK_SIZE]
            tasks.append({
                'tickers': batch, 
                'period': '5y', 
                'start': None, 
                'end': None,
                'download_type': 'FULL'
            })
            
        # Incremental Downloads
        if incremental_tickers:
            start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
            for i in range(0, len(incremental_tickers), Config.CHUNK_SIZE):
                batch = incremental_tickers[i:i + Config.CHUNK_SIZE]
                tasks.append({
                    'tickers': batch, 
                    'period': None, 
                    'start': start_date, 
                    'end': today.strftime('%Y-%m-%d'),
                    'download_type': 'INCREMENTAL'
                })

        logger.info(f"📊 Processing: {len(full_download_tickers)} Full | {len(incremental_tickers)} Incremental | {len(tasks)} batches")
        
        failed_tasks = []  # Collect for retry
        
        try:
            with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                future_to_batch = {
                    executor.submit(
                        self.download_batch, 
                        t['tickers'], 
                        t['period'], 
                        t['start'], 
                        t['end']
                    ): t for t in tasks
                }
                
                for future in tqdm(as_completed(future_to_batch), total=len(tasks), 
                                 desc="📥 Downloading & Storing", unit="batch"):
                    try:
                        raw_df = future.result(timeout=180)
                        if not raw_df.empty:
                            gaps = self._detect_trading_gaps(raw_df)
                            clean_df = self.process_dataframe(raw_df)
                            
                            if not clean_df.empty:
                                self.fast_bulk_upsert(clean_df)
                                metrics = self._get_data_quality_metrics(clean_df)
                                if metrics['validation_status'] != 'PASS':
                                    logger.warning(f"⚠️  Quality check: {metrics['validation_status']} - {metrics['warnings']}")
                            else:
                                batch_info = future_to_batch[future]
                                logger.warning(f"No valid records after processing for batch: {batch_info['tickers']}")
                    except TimeoutError:
                        batch_info = future_to_batch[future]
                        logger.warning(f"⏱️ Timeout for {batch_info['download_type']} batch ({len(batch_info['tickers'])} tickers) — will retry")
                        failed_tasks.append(batch_info)
                    except Exception as e:
                        batch_info = future_to_batch[future]
                        err_msg = str(e)[:120]
                        logger.error(f"Worker failed for {batch_info['download_type']} ({len(batch_info['tickers'])} tickers): {err_msg}")
                        failed_tasks.append(batch_info)
        except KeyboardInterrupt:
            logger.warning("⚠️ Pipeline interrupted — saving progress and shutting down gracefully...")
            # executor.__exit__ will call shutdown(wait=True), which may hang.
            # We let it finish the current in-flight downloads but don't retry.
            failed_tasks.clear()
        except Exception as e:
            logger.error(f"Pipeline executor error: {e}")
        
        # ---- Retry failed batches sequentially (avoids thread-pool issues) ----
        if failed_tasks:
            logger.info(f"🔄 Retrying {len(failed_tasks)} failed batches sequentially...")
            for task in failed_tasks:
                try:
                    raw_df = self.download_batch(
                        task['tickers'], task['period'], task['start'], task['end']
                    )
                    if not raw_df.empty:
                        clean_df = self.process_dataframe(raw_df)
                        if not clean_df.empty:
                            self.fast_bulk_upsert(clean_df)
                            logger.info(f"   ✅ Retry succeeded for {len(task['tickers'])} tickers")
                            continue
                    logger.warning(f"   ⚠️ Retry returned no data for {task['tickers'][:3]}...")
                    self.failed_tickers.extend(task['tickers'])
                except Exception as e:
                    logger.error(f"   ❌ Retry also failed: {str(e)[:80]}")
                    self.failed_tickers.extend(task['tickers'])

        self.print_statistics()

    def print_statistics(self):
        """Print pipeline execution statistics with quality metrics"""
        logger.info("=" * 70)
        logger.info("📈 PIPELINE STATISTICS")
        logger.info("=" * 70)
        logger.info(f"✅ Successful operations: {self.success_count}")
        logger.info(f"❌ Failed operations: {self.error_count}")
        logger.info(f"⚠️  Failed tickers: {len(self.failed_tickers)}")
        
        if self.failed_tickers:
            failed_clean = [t.replace('.NS', '') for t in self.failed_tickers[:20]]
            logger.warning(f"Failed tickers: {', '.join(failed_clean)}")
            if len(self.failed_tickers) > 20:
                logger.warning(f"... and {len(self.failed_tickers) - 20} more")
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        COUNT(DISTINCT ticker) as total_stocks,
                        COUNT(*) as total_records,
                        MIN(date) as earliest_date,
                        MAX(date) as latest_date,
                        ROUND(AVG(volume), 0) as avg_volume,
                        SUM(CASE WHEN volume > 0 THEN 1 ELSE 0 END) as records_with_volume
                    FROM nse_stocks
                """))
                row = result.fetchone()
                if row:
                    logger.info(f"📊 Database: {row.total_stocks} stocks, {row.total_records:,} records")
                    logger.info(f"📅 Date range: {row.earliest_date} to {row.latest_date}")
                    logger.info(f"📊 Avg volume: {int(row.avg_volume):,} | Records with volume: {row.records_with_volume:,}")
                    
                    # Data quality percentage
                    quality_pct = (row.records_with_volume / row.total_records * 100) if row.total_records > 0 else 0
                    quality_status = "✅ EXCELLENT" if quality_pct > 95 else "⚠️  GOOD" if quality_pct > 85 else "❌ NEEDS ATTENTION"
                    logger.info(f"📈 Data Quality: {quality_status} ({quality_pct:.1f}% valid records)")
        except Exception as e:
            logger.error(f"Failed to fetch statistics: {e}")
        
        logger.info("=" * 70)

if __name__ == "__main__":
    start_time = time.time()
    
    import argparse
    parser = argparse.ArgumentParser(description='NSE Stock Data Pipeline')
    parser.add_argument('--force-refresh', action='store_true', 
                       help='Force full refresh of all stocks')
    args = parser.parse_args()
    
    pipeline = NSEDataPipeline()
    symbols = pipeline.get_all_nse_symbols()
    
    if symbols:
        pipeline.run_pipeline(symbols, force_full_refresh=args.force_refresh)
    else:
        logger.error("No symbols found. Exiting.")
    
    elapsed_time = time.time() - start_time
    logger.info(f"🏁 Process completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")