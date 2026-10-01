import pandas as pd
from sqlalchemy import text
from backend.IntegratedPostGreSQL import NSEDataPipeline

db = NSEDataPipeline()
with db.engine.connect() as conn:
    result = conn.execute(text('SELECT ticker, date, close FROM nse_stocks LIMIT 5'))
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    print(df)
    print("Success with conn.execute!")
