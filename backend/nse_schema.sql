-- Prereq: CREATE EXTENSION IF NOT EXISTS timescaledb;
-- nse_stocks must have a unique key that includes the time column (ticker,date).
SELECT create_hypertable('nse_stocks','date', migrate_data=>true, if_not_exists=>true);

CREATE TABLE IF NOT EXISTS fii_dii(d date PRIMARY KEY, fii_net double precision, dii_net double precision);

CREATE TABLE IF NOT EXISTS preopen_snap(ts timestamptz NOT NULL, symbol text NOT NULL,
  iep double precision, tot_buy bigint, tot_sell bigint, prev_close double precision);
SELECT create_hypertable('preopen_snap','ts', chunk_time_interval=>interval '30 days', if_not_exists=>true);

CREATE TABLE IF NOT EXISTS fo_bhav(d date NOT NULL, symbol text NOT NULL, instr text NOT NULL, expiry date,
  strike double precision, opt_type text, oi bigint, chg_oi bigint, settle double precision);
SELECT create_hypertable('fo_bhav','d', if_not_exists=>true);

CREATE TABLE IF NOT EXISTS scores(d date NOT NULL, symbol text NOT NULL, score double precision,
  rank_pct double precision, is_top boolean, model_ver text, PRIMARY KEY(d,symbol));
SELECT create_hypertable('scores','d', if_not_exists=>true);

CREATE TABLE IF NOT EXISTS ic_monitor(d date PRIMARY KEY, ic double precision, n int, roll20_ic double precision);

CREATE OR REPLACE VIEW flow_feat AS
WITH l AS (SELECT d, LAG(fii_net) OVER (ORDER BY d) f1, LAG(dii_net) OVER (ORDER BY d) g1 FROM fii_dii)
SELECT d, (f1-AVG(f1) OVER w)/NULLIF(STDDEV(f1) OVER w,0) AS fii_z_l1,
          (g1-AVG(g1) OVER w)/NULLIF(STDDEV(g1) OVER w,0) AS dii_z_l1
FROM l WINDOW w AS (ORDER BY d ROWS BETWEEN 119 PRECEDING AND CURRENT ROW);

CREATE OR REPLACE VIEW preopen_feat AS
SELECT d, symbol, ln(iep/pc) AS iep_gap, imb,
       percent_rank() OVER (PARTITION BY d ORDER BY imb) AS imb_cs,
       (d >= DATE '2026-09-07') AS post_rule
FROM (SELECT (ts AT TIME ZONE 'Asia/Kolkata')::date d, symbol,
             last(iep,ts) iep, last(prev_close,ts) pc,
             (last(tot_buy,ts)-last(tot_sell,ts))::float / NULLIF(last(tot_buy,ts)+last(tot_sell,ts),0) imb
      FROM preopen_snap
      WHERE (ts AT TIME ZONE 'Asia/Kolkata')::time BETWEEN '09:08' AND '09:12'
      GROUP BY 1,2) s;

CREATE OR REPLACE VIEW fo_feat AS
WITH nx AS (SELECT d,symbol,min(expiry) ex FROM fo_bhav WHERE instr='OPTSTK' AND expiry>=d GROUP BY 1,2),
pcr AS (SELECT b.d,b.symbol,
        sum(b.oi) FILTER (WHERE b.opt_type='PE')::float / NULLIF(sum(b.oi) FILTER (WHERE b.opt_type='CE'),0) AS pcr_oi
        FROM fo_bhav b JOIN nx ON b.d=nx.d AND b.symbol=nx.symbol AND b.expiry=nx.ex
        WHERE b.instr='OPTSTK' GROUP BY 1,2),
fut AS (SELECT DISTINCT ON (d,symbol) d,symbol,oi,chg_oi,settle
        FROM fo_bhav WHERE instr='FUTSTK' ORDER BY d,symbol,expiry)
SELECT f.d, f.symbol, ln(pcr_oi) AS ln_pcr,
       f.chg_oi::float/NULLIF(f.oi-f.chg_oi,0) AS d_oi_pct, (pcr_oi IS NOT NULL) AS fo_avail
FROM fut f LEFT JOIN pcr USING (d,symbol);