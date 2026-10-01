import pandas as pd
from sqlalchemy import create_engine, text
from backend.IntegratedPostGreSQL import NSEDataPipeline

db = NSEDataPipeline()
print(pd.__version__)
print(db.engine)
with db.engine.connect() as conn:
    try:
        df = pd.read_sql('SELECT ticker, date, close FROM nse_stocks LIMIT 5', conn)
        print("Success with conn!")
    except Exception as e:
        print(f"Failed with conn: {e}")

try:
    df = pd.read_sql('SELECT ticker, date, close FROM nse_stocks LIMIT 5', db.engine)
    print("Success with engine!")
except Exception as e:
    print(f"Failed with engine: {e}")

