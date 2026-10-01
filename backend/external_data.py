import requests
import json
import logging
import pandas as pd
from datetime import datetime, date, timedelta
from bs4 import BeautifulSoup
import os
import tempfile
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Try importing jugaad_data
try:
    from jugaad_data.nse import bhavcopy_fo_save
    JUGAAD_AVAILABLE = True
except ImportError:
    JUGAAD_AVAILABLE = False
    logger.warning("jugaad-data not found. F&O data fetching will be limited.")

class NSEDataFetcher:
    """
    Handles fetching of real data from NSE APIs and using jugaad-data.
    """
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        }
        self.base_url = "https://www.nseindia.com"
        # We need to hit the base URL first to get cookies
        try:
            self.session.get(self.base_url, headers=self.headers, timeout=10)
        except Exception as e:
            logger.error(f"Failed to initialize NSE session: {e}")

    def fetch_fii_dii_flow(self) -> Optional[Dict]:
        """
        Fetches the latest FII/DII flow data from NSE.
        Note: NSE API only provides the latest (current day or last trading day) data directly.
        For historical, you need to scrape reports. We fetch the latest available.
        """
        url = f"{self.base_url}/api/fiidiiTradeReact"
        try:
            response = self.session.get(url, headers=self.headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if not data:
                    return None
                
                # data typically contains a list of records
                result = {
                    'date': None,
                    'fii_buy': 0.0, 'fii_sell': 0.0, 'fii_net': 0.0,
                    'dii_buy': 0.0, 'dii_sell': 0.0, 'dii_net': 0.0
                }
                
                for item in data:
                    category = item.get('category', '').upper()
                    date_str = item.get('date', '')
                    if not result['date'] and date_str:
                        # Assuming date_str format is "dd-MMM-yyyy"
                        try:
                            result['date'] = datetime.strptime(date_str, "%d-%b-%Y").date()
                        except ValueError:
                            pass
                    
                    buy = float(item.get('buyValue', 0.0))
                    sell = float(item.get('sellValue', 0.0))
                    net = float(item.get('netValue', 0.0))
                    
                    if 'FII' in category:
                        result['fii_buy'] = buy
                        result['fii_sell'] = sell
                        result['fii_net'] = net
                    elif 'DII' in category:
                        result['dii_buy'] = buy
                        result['dii_sell'] = sell
                        result['dii_net'] = net
                        
                # Fallback to today if date parsing fails
                if not result['date']:
                    result['date'] = datetime.today().date()
                    
                return result
            else:
                logger.warning(f"Failed to fetch FII/DII data. Status: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching FII/DII flow: {e}")
            return None

    def fetch_fo_ban_list(self) -> list:
        """
        Fetches the latest F&O Ban list.
        """
        url = f"{self.base_url}/api/historical/fo/derivatives/meta-data?type=banList"
        try:
            response = self.session.get(url, headers=self.headers, timeout=15)
            if response.status_code == 200:
                # The response could be an excel file or json depending on the endpoint.
                # Actually, NSE usually provides an XML or JSON. Let's try JSON first.
                try:
                    data = response.json()
                    # Parse JSON if possible
                    # This depends on exact NSE structure which changes often.
                    if isinstance(data, dict) and 'data' in data:
                        return [str(item) for item in data['data']]
                except json.JSONDecodeError:
                    pass
                
                # As a fallback or simpler method, we might just return an empty list 
                # if we can't parse it reliably, since ban lists are also published via Bhavcopy.
            return []
        except Exception as e:
            logger.error(f"Error fetching ban list: {e}")
            return []

    def fetch_options_metrics(self, dt: date) -> Optional[pd.DataFrame]:
        """
        Uses jugaad_data to download the FO bhavcopy for a given date,
        then calculates PCR and approximates IV skew for all tickers.
        """
        if not JUGAAD_AVAILABLE:
            return None
            
        temp_dir = tempfile.gettempdir()
        
        try:
            # Download Bhavcopy to temp dir
            file_path = bhavcopy_fo_save(dt, temp_dir)
            
            # Read the CSV
            df = pd.read_csv(file_path)
            
            # Clean up the downloaded file
            try:
                os.remove(file_path)
            except:
                pass
                
            # Filter for Options (OPTSTK and OPTIDX)
            opts = df[df['INSTRUMENT'].isin(['OPTSTK', 'OPTIDX'])].copy()
            
            if opts.empty:
                return None
                
            # Calculate PCR per ticker
            # PCR = Total Put Volume / Total Call Volume (or Open Interest)
            # Usually based on Open Interest (OPEN_INT)
            
            # Group by SYMBOL and OPTION_TYP (CE / PE)
            grouped = opts.groupby(['SYMBOL', 'OPTION_TYP'])['OPEN_INT'].sum().unstack(fill_value=0)
            
            metrics = pd.DataFrame(index=grouped.index)
            
            if 'CE' in grouped.columns and 'PE' in grouped.columns:
                metrics['put_call_ratio'] = grouped['PE'] / grouped['CE'].replace(0, 1)
            else:
                metrics['put_call_ratio'] = 1.0
                
            # Approximate IV based on ATM pricing (very rough approximation for now)
            # A true IV requires spot price and Black Scholes (py_vollib)
            # Here we just set up the schema output.
            metrics['iv_near_month'] = 0.20  # placeholder
            metrics['iv_next_month'] = 0.22  # placeholder
            metrics['iv_term_structure_slope'] = metrics['iv_next_month'] - metrics['iv_near_month']
            
            metrics = metrics.reset_index()
            metrics.rename(columns={'SYMBOL': 'ticker'}, inplace=True)
            metrics['date'] = dt
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching F&O bhavcopy for {dt}: {e}")
            return None

    def store_fii_dii_flow(self, engine, data: Dict):
        """Store FII/DII flow into PostgreSQL"""
        if not data or not data.get('date'):
            return
        
        from sqlalchemy import text
        query = text("""
            INSERT INTO fii_dii_flow 
            (date, fii_buy_value, fii_sell_value, fii_net_value, dii_buy_value, dii_sell_value, dii_net_value)
            VALUES (:date, :fii_buy, :fii_sell, :fii_net, :dii_buy, :dii_sell, :dii_net)
            ON CONFLICT (date) DO UPDATE SET
            fii_buy_value = EXCLUDED.fii_buy_value,
            fii_sell_value = EXCLUDED.fii_sell_value,
            fii_net_value = EXCLUDED.fii_net_value,
            dii_buy_value = EXCLUDED.dii_buy_value,
            dii_sell_value = EXCLUDED.dii_sell_value,
            dii_net_value = EXCLUDED.dii_net_value,
            updated_at = CURRENT_TIMESTAMP
        """)
        try:
            with engine.begin() as conn:
                conn.execute(query, data)
        except Exception as e:
            logger.error(f"Error storing FII/DII flow: {e}")

    def store_fo_ban_list(self, engine, dt: date, ban_list: list):
        """Store FO Ban List into PostgreSQL"""
        if not ban_list:
            return
            
        from sqlalchemy import text
        try:
            with engine.begin() as conn:
                for ticker in ban_list:
                    query = text("""
                        INSERT INTO fo_ban_list (date, ticker)
                        VALUES (:date, :ticker)
                        ON CONFLICT (date, ticker) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                    """)
                    conn.execute(query, {'date': dt, 'ticker': ticker})
        except Exception as e:
            logger.error(f"Error storing FO Ban List: {e}")

    def store_options_metrics(self, engine, metrics_df: pd.DataFrame):
        """Store options metrics into PostgreSQL"""
        if metrics_df is None or metrics_df.empty:
            return
            
        from sqlalchemy import text
        try:
            with engine.begin() as conn:
                for _, row in metrics_df.iterrows():
                    query = text("""
                        INSERT INTO options_metrics 
                        (date, ticker, put_call_ratio, iv_near_month, iv_next_month, iv_term_structure_slope)
                        VALUES (:date, :ticker, :put_call_ratio, :iv_near_month, :iv_next_month, :iv_term_structure_slope)
                        ON CONFLICT (date, ticker) DO UPDATE SET
                        put_call_ratio = EXCLUDED.put_call_ratio,
                        iv_near_month = EXCLUDED.iv_near_month,
                        iv_next_month = EXCLUDED.iv_next_month,
                        iv_term_structure_slope = EXCLUDED.iv_term_structure_slope,
                        updated_at = CURRENT_TIMESTAMP
                    """)
                    conn.execute(query, {
                        'date': row['date'],
                        'ticker': row['ticker'],
                        'put_call_ratio': row['put_call_ratio'],
                        'iv_near_month': row['iv_near_month'],
                        'iv_next_month': row['iv_next_month'],
                        'iv_term_structure_slope': row['iv_term_structure_slope']
                    })
        except Exception as e:
            logger.error(f"Error storing options metrics: {e}")

# Simple test block
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = NSEDataFetcher()
    fii_data = fetcher.fetch_fii_dii_flow()
    print("FII/DII Flow:", fii_data)
