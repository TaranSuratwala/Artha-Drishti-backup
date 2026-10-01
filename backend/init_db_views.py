import os
import sqlalchemy
from sqlalchemy import text
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_views():
    load_dotenv()
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL not found in environment.")
        return

    logger.info(f"Connecting to database...")
    engine = sqlalchemy.create_engine(db_url)
    
    with engine.begin() as conn:
        logger.info("Applying Flow feature view...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fii_dii (
                d DATE PRIMARY KEY,
                fii_net DOUBLE PRECISION,
                dii_net DOUBLE PRECISION
            );
        """))
        conn.execute(text("""
            CREATE OR REPLACE VIEW flow_feat AS 
            WITH l AS (
                SELECT d, 
                       LAG(fii_net) OVER (ORDER BY d) as f1, 
                       LAG(dii_net) OVER (ORDER BY d) as g1 
                FROM fii_dii
            )
            SELECT d, 
                   (f1 - AVG(f1) OVER w) / NULLIF(STDDEV(f1) OVER w, 0) AS fii_z_l1, 
                   (g1 - AVG(g1) OVER w) / NULLIF(STDDEV(g1) OVER w, 0) AS dii_z_l1
            FROM l 
            WINDOW w AS (ORDER BY d ROWS BETWEEN 119 PRECEDING AND CURRENT ROW);
        """))
        
        logger.info("Applying Pre-open feature view...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS preopen_snap (
                ts TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                iep DOUBLE PRECISION,
                tot_buy BIGINT,
                tot_sell BIGINT,
                prev_close DOUBLE PRECISION
            );
        """))
        
        try:
            conn.execute(text("SELECT create_hypertable('preopen_snap','ts',chunk_time_interval=>interval '30 days', if_not_exists=>TRUE);"))
        except Exception as e:
            pass

        conn.execute(text("""
            CREATE OR REPLACE VIEW preopen_feat AS 
            SELECT 
                d, 
                symbol, 
                ln(iep / pc) AS iep_gap, 
                imb, 
                percent_rank() OVER (PARTITION BY d ORDER BY imb) AS imb_cs, 
                (d >= DATE '2026-09-07') AS post_rule_change
            FROM (
                SELECT 
                    (ts AT TIME ZONE 'Asia/Kolkata')::date AS d, 
                    symbol, 
                    last(iep, ts) AS iep, 
                    last(prev_close, ts) AS pc, 
                    (last(tot_buy, ts) - last(tot_sell, ts))::float / NULLIF(last(tot_buy, ts) + last(tot_sell, ts), 0) AS imb
                FROM preopen_snap 
                WHERE (ts AT TIME ZONE 'Asia/Kolkata')::time BETWEEN '09:08' AND '09:12'
                GROUP BY 1, 2
            ) s;
        """))
        
        logger.info("Applying F&O feature view...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fo_bhav (
                d DATE,
                symbol TEXT,
                instr TEXT,
                expiry DATE,
                strike DOUBLE PRECISION,
                opt_type TEXT,
                oi BIGINT,
                chg_oi BIGINT,
                settle DOUBLE PRECISION
            );
        """))
        
        conn.execute(text("""
            CREATE OR REPLACE VIEW fo_feat AS 
            WITH nx AS (
                SELECT d, symbol, min(expiry) AS ex 
                FROM fo_bhav 
                WHERE instr = 'OPTSTK' AND expiry >= d 
                GROUP BY 1, 2
            ), pcr AS (
                SELECT 
                    b.d, 
                    b.symbol, 
                    sum(b.oi) FILTER (WHERE b.opt_type = 'PE')::float / 
                    NULLIF(sum(b.oi) FILTER (WHERE b.opt_type = 'CE'), 0) AS pcr_oi
                FROM fo_bhav b 
                JOIN nx ON b.d = nx.d AND b.symbol = nx.symbol AND b.expiry = nx.ex 
                WHERE b.instr = 'OPTSTK' 
                GROUP BY 1, 2
            ), fut AS (
                SELECT DISTINCT ON (d, symbol) 
                    d, symbol, oi, chg_oi, settle 
                FROM fo_bhav 
                WHERE instr = 'FUTSTK' 
                ORDER BY d, symbol, expiry
            )
            SELECT 
                f.d, 
                f.symbol, 
                ln(pcr_oi) AS ln_pcr, 
                f.chg_oi::float / NULLIF(f.oi - f.chg_oi, 0) AS d_oi_pct, 
                (pcr_oi IS NOT NULL) AS fo_avail 
            FROM fut f 
            LEFT JOIN pcr USING (d, symbol);
        """))

    logger.info("All views created successfully.")

if __name__ == '__main__':
    apply_views()
