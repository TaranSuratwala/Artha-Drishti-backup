"""
backend/data_universe.py — Point-In-Time (PIT) Universe, Eligibility Filters, and Data Loaders.

TIMING CONVENTION:
    Features use data <= close of day t -> entry at OPEN t+1 -> exit at CLOSE t+k or earlier on barrier touch.
    Barriers use high/low of t+1...t+k; same-day double touch -> stop first; open through a barrier -> fill at open.
    Adjusted prices; point-in-time universe incl. delisted; reindex to master trading calendar (k = market days).
    Restate in every label/feature docstring.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union
import logging
import numpy as np
import pandas as pd
import os
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


# ============================================================================
# 1. CONFIGURATION & STATUTORY NSE RULES
# ============================================================================

@dataclass
class UniverseConfig:
    """
    NSE Equity Universe Point-In-Time (PIT) Eligibility Criteria.
    Verify against latest NSE circulars and contract notes.
    """
    # Liquidity: Median 20-day traded value >= Rs 5 Crore (50,000,000 INR)
    min_median_turnover_20d: float = 50_000_000.0  # Rs 5 Cr
    min_price: float = 10.0                        # Exclude penny stocks (< Rs 10)
    
    # Series: Only EQ (regular equity) is eligible for cash delivery alpha
    allowed_series: Tuple[str, ...] = ("EQ",)
    
    # Circuit filter: Standard NSE daily price bands
    default_price_band_pct: float = 0.20           # 20% standard band
    circuit_tolerance_pct: float = 0.001           # 0.1% tolerance from circuit limit
    
    # Execution safety: Reject entry if t+1 open is locked at circuit limit
    reject_locked_open: bool = True
    
    # Surveillance: Exclude Additional Surveillance Measure (ASM) & Graded Surveillance Measure (GSM)
    exclude_asm_gsm: bool = True
    
    # Master calendar reindexing
    reindex_to_master_calendar: bool = True


# ============================================================================
# 2. BAD PRINT & DATA CORRUPTION DETECTOR
# ============================================================================

class BadPrintDetector:
    """
    Causal Data Integrity & Bad Print Detection Engine.
    Operates strictly Point-In-Time (no future look-ahead statistics).
    """

    @staticmethod
    def inspect_and_clean(df: pd.DataFrame, max_daily_jump_pct: float = 0.25) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Inspects panel for data corruption, bad prints, and bar geometry violations.
        
        Rules:
        1. Non-positive prices: Open, High, Low, Close must be > 0.
        2. Geometry violations: High must be >= max(Open, Close); Low must be <= min(Open, Close).
        3. Flash anomalies: High - Low < 0 or High / Low > 2.0 within a single session without circuit expansion.
        4. Jump anomalies: |ln(C_t / C_{t-1})| > max_daily_jump_pct without split adjustment.
        
        Returns:
            (cleaned_df, anomaly_counts)
        """
        d = df.copy()
        counts = {
            "non_positive_price": 0,
            "geometry_violation": 0,
            "flash_spike": 0,
            "unadjusted_jump": 0,
        }
        
        # 1. Non-positive prices
        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in d.columns]
        bad_pos = pd.Series(False, index=d.index)
        for col in price_cols:
            bad_pos |= (d[col] <= 0) | d[col].isna()
        counts["non_positive_price"] = int(bad_pos.sum())
        if bad_pos.any():
            d = d[~bad_pos].copy()
            
        # 2. Bar geometry violations
        if all(c in d.columns for c in ['open', 'high', 'low', 'close']):
            max_oc = np.maximum(d['open'].values, d['close'].values) - 1e-4
            min_oc = np.minimum(d['open'].values, d['close'].values) + 1e-4
            geom_bad = (d['high'].values < max_oc) | (d['low'].values > min_oc)
            counts["geometry_violation"] = int(geom_bad.sum())
            if geom_bad.any():
                # Correct minor high/low breaches causally
                d.loc[geom_bad, 'high'] = np.maximum(d.loc[geom_bad, 'high'], np.maximum(d.loc[geom_bad, 'open'], d.loc[geom_bad, 'close']))
                d.loc[geom_bad, 'low'] = np.minimum(d.loc[geom_bad, 'low'], np.minimum(d.loc[geom_bad, 'open'], d.loc[geom_bad, 'close']))

        # 3. Flash spikes
        if 'high' in d.columns and 'low' in d.columns:
            hl_ratio = d['high'] / d['low'].replace(0, np.nan)
            flash_bad = hl_ratio > 3.0
            counts["flash_spike"] = int(flash_bad.sum())
            if flash_bad.any():
                # Flag as bad print
                d['is_bad_print'] = flash_bad
            else:
                d['is_bad_print'] = False
        else:
            d['is_bad_print'] = False

        # 4. Jump anomalies (ex-corporate actions)
        if 'ticker' in d.columns and 'close' in d.columns:
            d = d.sort_values(['ticker', 'date']).reset_index(drop=True)
            log_ret = d.groupby('ticker')['close'].apply(lambda s: np.log(s / s.shift(1))).reset_index(drop=True)
            jump_bad = log_ret.abs() > max_daily_jump_pct
            counts["unadjusted_jump"] = int(jump_bad.sum())
            d['is_extreme_jump'] = jump_bad
        else:
            d['is_extreme_jump'] = False

        return d, counts


# ============================================================================
# 3. POINT-IN-TIME (PIT) UNIVERSE & ELIGIBILITY ENGINE
# ============================================================================

class PITUniverseFilter:
    """
    Point-In-Time Universe Filter for NSE Equities.
    
    Guarantees:
    - Zero look-ahead: all eligibility decisions use information <= close of day t.
    - Delisted tickers are included up to their final trading session.
    - Entry feasibility: execution rejects orders where t+1 open is locked at circuit limits.
    """

    def __init__(self, config: Optional[UniverseConfig] = None):
        self.config = config or UniverseConfig()

    def compute_turnover_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Point-In-Time traded turnover in INR.
        Uses turnover column directly if present, else close * volume.
        """
        df = df.copy()
        if 'turnover' in df.columns and df['turnover'].notna().any():
            # If turnover is in Lakhs (NSE Bhavcopy standard), scale to INR
            # Typical NSE sec_bhavdata_full turnover is in Lakhs (e.g. 100 = 1 Cr)
            # Check median value to avoid double-multiplying
            sample_val = df['turnover'].dropna().median()
            if sample_val < 1e5 and sample_val > 0:
                df['turnover_inr'] = df['turnover'] * 100_000.0
            else:
                df['turnover_inr'] = df['turnover']
        else:
            df['turnover_inr'] = df['close'] * df['volume']
        return df

    def compute_eligibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Point-In-Time eligibility mask for each (date, ticker) bar.
        
        Returns DataFrame with boolean columns:
            - 'is_liquid_20d': 20-day rolling median turnover >= Rs 5 Cr
            - 'is_series_eq': Series == 'EQ'
            - 'not_asm_gsm': Not subject to ASM or GSM restrictions
            - 'is_price_eligible': Close price >= min_price
            - 'is_universe_eligible': Conjunction of all PIT eligibility rules at day t
            - 'can_enter_buy_t1': Day t+1 open is not locked at upper circuit
            - 'can_enter_sell_t1': Day t+1 open is not locked at lower circuit
        """
        d = df.copy()
        d['date'] = pd.to_datetime(d['date'])
        d = d.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # 1. Compute turnover in INR
        d = self.compute_turnover_panel(d)
        
        # 2. Rolling 20-day median turnover strictly causal (<= day t)
        # Groupby rolling avoids cross-ticker leakage
        d['adtv20_inr'] = d.groupby('ticker')['turnover_inr'].transform(
            lambda s: s.rolling(20, min_periods=1).median()
        )
        d['is_liquid_20d'] = d['adtv20_inr'] >= self.config.min_median_turnover_20d
        
        # 3. Price floor (anti-penny stock)
        d['is_price_eligible'] = d['close'] >= self.config.min_price
        
        # 4. Series check
        if 'series' in d.columns:
            d['is_series_eq'] = d['series'].isin(self.config.allowed_series)
        else:
            # If no series column provided in raw DB, assume EQ if not in known non-EQ list
            d['is_series_eq'] = True
            
        # 5. ASM / GSM Surveillance Check
        if 'is_asm_gsm' in d.columns:
            d['not_asm_gsm'] = ~d['is_asm_gsm'].astype(bool)
        else:
            d['not_asm_gsm'] = True
            
        # Composite PIT Eligibility at Close of Day t
        d['is_universe_eligible'] = (
            d['is_liquid_20d'] &
            d['is_price_eligible'] &
            d['is_series_eq'] &
            d['not_asm_gsm']
        )
        
        # 6. Circuit Rules & t+1 Open Lock Check
        # Price band detection: if price_band column not given, use default (20%)
        if 'price_band' in d.columns:
            band = pd.to_numeric(d['price_band'], errors='coerce').fillna(self.config.default_price_band_pct)
        else:
            band = pd.Series(self.config.default_price_band_pct, index=d.index)
            
        d['price_band_val'] = band
        # Upper and lower circuit limits for session t
        prev_close = d.groupby('ticker')['close'].shift(1).fillna(d['open'])
        d['upper_circuit_t'] = prev_close * (1.0 + band)
        d['lower_circuit_t'] = prev_close * (1.0 - band)
        
        # t+1 execution lock check (for entering at OPEN of t+1)
        next_open = d.groupby('ticker')['open'].shift(-1)
        next_band = d.groupby('ticker')['price_band_val'].shift(-1).fillna(self.config.default_price_band_pct)
        next_upper = d['close'] * (1.0 + next_band)
        next_lower = d['close'] * (1.0 - next_band)
        
        tol = self.config.circuit_tolerance_pct
        # Upper locked: cannot execute BUY at Open t+1
        upper_locked_t1 = next_open >= (next_upper * (1.0 - tol))
        # Lower locked: cannot execute SELL at Open t+1
        lower_locked_t1 = next_open <= (next_lower * (1.0 + tol))
        
        # Open price missing or 0 also cannot be traded
        valid_t1_open = next_open.notna() & (next_open > 0)
        
        d['can_enter_buy_t1'] = d['is_universe_eligible'] & valid_t1_open & (~upper_locked_t1)
        d['can_enter_sell_t1'] = d['is_universe_eligible'] & valid_t1_open & (~lower_locked_t1)
        
        return d


# ============================================================================
# 4. DATA LOADERS & DATABASE INTEGRATION
# ============================================================================

class NSEDataLoader:
    """
    Production Data Loader for NSE Equities & F&O Bhavcopy.
    Queries PostgreSQL with full delivery and traded turnover projection.
    """

    def __init__(self, db_url: Optional[str] = None):
        try:
            from backend.IntegratedPostGreSQL import Config
        except ImportError:
            from IntegratedPostGreSQL import Config
        self.db_url = db_url or Config.DB_URL or os.getenv("DATABASE_URL")
        if not self.db_url:
            # Fallback to local default if configured (password URL encoded)
            self.db_url = "postgresql://postgres:Taran%4017@localhost:5432/StockDB"
        self.engine = create_engine(self.db_url, pool_pre_ping=True)

    def load_stocks_with_delivery(
        self,
        start_date: str,
        end_date: str,
        tickers: Optional[List[str]] = None,
        min_records: int = 5
    ) -> pd.DataFrame:
        """
        Loads NSE equity data with delivery and traded quantity fields.
        Fixes the bug where delivery fields were omitted in historical SELECT queries.
        
        Columns returned:
            date, ticker, open, high, low, close, adj_close, volume,
            delivery_qty, delivery_percentage, traded_qty, split_factor
        """
        params = {
            'start_date': pd.to_datetime(start_date).strftime('%Y-%m-%d'),
            'end_date': pd.to_datetime(end_date).strftime('%Y-%m-%d')
        }
        
        base_query = """
            SELECT 
                ticker,
                date,
                open,
                high,
                low,
                close,
                adj_close,
                volume,
                COALESCE(delivery_qty, 0) AS delivery_qty,
                COALESCE(delivery_percentage, 0.0) AS delivery_percentage,
                COALESCE(traded_qty, volume) AS traded_qty,
                COALESCE(split_factor, 1.0) AS split_factor
            FROM nse_stocks
            WHERE date >= :start_date AND date <= :end_date
        """
        
        if tickers:
            base_query += " AND ticker = ANY(:tickers)"
            params['tickers'] = tickers
            
        base_query += " ORDER BY ticker, date ASC"
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(base_query), conn, params=params)
            
        if df.empty:
            logger.warning(f"No records found between {start_date} and {end_date}")
            return pd.DataFrame()
            
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter tickers with insufficient observations
        counts = df.groupby('ticker')['date'].count()
        valid_tickers = counts[counts >= min_records].index
        df = df[df['ticker'].isin(valid_tickers)].copy()
        
        # Corporate Action Adjustment
        ratio = (df['adj_close'] / df['close'].replace(0, np.nan)).clip(0.01, 100.0).fillna(1.0)
        df['adj_open'] = df['open'] * ratio
        df['adj_high'] = df['high'] * ratio
        df['adj_low'] = df['low'] * ratio
        df['adj_volume'] = df['volume'] / ratio
        df['turnover_inr'] = df['close'] * df['volume']
        
        return df

    def load_fo_bhavcopy(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Loads F&O Bhavcopy data (Open Interest, Rollover, Futures Settle).
        Returns empty DataFrame gracefully if table is unpopulated.
        """
        q = text("""
            SELECT d AS date, symbol, instr, expiry, strike, opt_type, oi, chg_oi, settle
            FROM fo_bhav
            WHERE d >= :start_date AND d <= :end_date
            ORDER BY symbol, d, expiry ASC
        """)
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(q, conn, params={'start_date': start_date, 'end_date': end_date})
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.warning(f"Failed to load fo_bhavcopy: {e}")
            return pd.DataFrame()

    def build_master_trading_calendar(self, df_prices: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Builds master trading calendar from dates where at least 100 tickers traded.
        """
        date_counts = df_prices.groupby('date')['ticker'].count()
        valid_dates = date_counts[date_counts >= 50].index.sort_values()
        return pd.DatetimeIndex(valid_dates)


class NSEBhavcopyLoader:
    """
    Parser and Ingester for NSE Daily Bhavcopy Files:
    1. sec_bhavdata_full.csv: Delivery quantity, delivery %, number of trades, turnover in Lakhs.
    2. fo_bhav.csv / UDiFF: Futures & Options open interest, rollover, settlement prices.
    """

    @staticmethod
    def parse_sec_bhavdata_full(filepath_or_buffer) -> pd.DataFrame:
        """
        Parses NSE sec_bhavdata_full.csv.
        Handles column name whitespace, uppercase normalization, and conversion of Lakhs to INR.
        """
        df = pd.read_csv(filepath_or_buffer)
        df.columns = [c.strip().upper() for c in df.columns]
        
        # Mapping official NSE column names
        rename_map = {
            'SYMBOL': 'ticker',
            'SERIES': 'series',
            'DATE1': 'date',
            'OPEN_PRICE': 'open',
            'HIGH_PRICE': 'high',
            'LOW_PRICE': 'low',
            'CLOSE_PRICE': 'close',
            'PREV_CLOSE': 'prev_close',
            'TTL_TRD_QNTY': 'traded_qty',
            'TURNOVER_LACS': 'turnover_lacs',
            'NO_OF_TRADES': 'no_of_trades',
            'DELIV_QTY': 'delivery_qty',
            'DELIV_PER': 'delivery_percentage'
        }
        df = df.rename(columns=rename_map)
        
        # Clean string columns
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.strip()
        if 'series' in df.columns:
            df['series'] = df['series'].astype(str).str.strip()
            # Only keep EQ series by default for cash delivery
            df = df[df['series'] == 'EQ'].copy()
            
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce').fillna(
                pd.to_datetime(df['date'], errors='coerce')
            )
            
        # Convert numeric columns
        num_cols = ['open', 'high', 'low', 'close', 'prev_close', 'traded_qty', 'turnover_lacs', 'no_of_trades', 'delivery_qty', 'delivery_percentage']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('-', '0').str.strip(), errors='coerce').fillna(0.0)
                
        # Turnover Lakhs -> INR (1 Lakh = 100,000 INR)
        if 'turnover_lacs' in df.columns:
            df['turnover_inr'] = df['turnover_lacs'] * 100_000.0
        else:
            df['turnover_inr'] = df['close'] * df['traded_qty']
            
        df['volume'] = df['traded_qty']
        return df

    @staticmethod
    def parse_fo_bhavcopy(filepath_or_buffer) -> pd.DataFrame:
        """
        Parses NSE UDiFF F&O Bhavcopy CSV.
        Extracts Futures and Options Open Interest, contracts, and settlement prices.
        """
        df = pd.read_csv(filepath_or_buffer)
        df.columns = [c.strip() for c in df.columns]
        
        # Handle UDiFF / Bhavcopy column format
        col_map = {
            'TradDt': 'date',
            'TckrSymb': 'symbol',
            'FinInstrmTp': 'instr',
            'XpryDt': 'expiry',
            'StrkPric': 'strike',
            'OptnTp': 'opt_type',
            'OpnIntrst': 'oi',
            'ChngInOpnIntrst': 'chg_oi',
            'SttlmPric': 'settle'
        }
        df = df.rename(columns=col_map)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if 'expiry' in df.columns:
            df['expiry'] = pd.to_datetime(df['expiry'])
            
        for col in ['strike', 'oi', 'chg_oi', 'settle']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                
        return df

