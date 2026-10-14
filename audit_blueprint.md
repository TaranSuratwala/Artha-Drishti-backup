# Artha-Drishti: Stage 0 Audit & Blueprint

## Audit of Existing Backend Files

| File | Current Role | Defects Found | Action | Priority |
| :--- | :--- | :--- | :--- | :--- |
| `AdvancedFeatureEngine.py` | Feature extraction (multi-horizon) | Uses point features per ticker without full cross-sectional scaling. Rollings use `min`/`max` scaling that leaks look-ahead if not strictly historical. Lacks robust collinearity control and actual cross-sectional relative strength vs market/sector neutralisation at a point in time. | Deprecate & Replace with `nse_features.py` and `cs_transforms.py`. | High |
| `FeatureEngineering.py` | Legacy feature engineering | Similar to above; hardcoded horizons, overlapping logic with advanced engine, basic `target_is_up_5d` (naive target). | Deprecate & Replace. | High |
| `MLPredictor.py` | Model training & prediction | Massive monolithic file (~945KB). Uses split-set/K-fold CV but lacks rigorous Combinatorial Purged CV (CPCV). Prone to look-ahead via overlaps (does not embargo properly across horizons). Targets are naive classes instead of rank-Gauss relative alpha. | Deprecate & Replace with `multi_horizon_models.py` and `purged_cv.py`. | High |
| `IntegratedPostGreSQL.py` | Database schema & ingestion | `engineered_features` is a wide table hardcoding columns (e.g., `rsi_7`, `macd`). `options_metrics`, `fo_ban_list`, `fii_dii_flow` are completely empty. Fails to record point-in-time parameters. Missing `no_of_trades`, `turnover_lacs`, `open_interest` in NSE cash tables. | Extend schema: EAV or JSONB for point-in-time features, add missing execution data sources. | Medium |
| `tasks.py` | Celery scheduling | Runs per-ticker model training sequentially. Cannot perform cross-sectional modeling if models are trained per-ticker. | Extend: Switch to daily cross-sectional multi-horizon training/inference batches. Add fitted-parameter versioning. | Medium |
| `RiskAnalytics.py` | Risk metrics | Likely operates on point-in-time individual trades rather than fully unified portfolio construction. | Replace/Extend: Integrate with Stage 5 friction model and Stage 6 reporting. | Low |

## Blueprint for Architecture Upgrade

### 1. Database Schema Changes (`IntegratedPostGreSQL.py`)
- **Point-in-Time Features Table:** Shift from a wide schema (`engineered_features`) to a more dynamic storage (e.g., JSONB or unpivoted `(date, ticker, feature_name, feature_value)`) to allow adding features without schema migrations. 
- **Fitted Parameter Versioning:** Add a `model_registry` table storing pipeline state (e.g., FFD window parameters `d` per ticker, scaling factors, cross-sectional cluster centroids) so inference is strictly out-of-sample.
- **Execution Data Data Loaders:** Integrate actual fetching/storage of NSE `sec_bhavdata_full` (Delivery %, `no_of_trades`, `turnover_lacs`) and F&O bhavcopy (Open Interest, Rollover). 

### 2. Scheduling Changes (`tasks.py`)
- Move from `train_model_task(ticker)` to `train_cross_sectional_model(date_range)`.
- Schedule daily data ingestion for both Cash and F&O segments before feature engineering.
- Implement explicit separation between "training" (which versions scalers and model weights) and "inference" (which applies point-in-time scalers to today's vector).

### 3. Pipeline Modules
- **`fracdiff.py`**: Stage 1a (Fractional differentiation).
- **`nse_features.py`**: Stage 1b (Microstructure, Volatility, Options metrics, India VIX regime).
- **`cs_transforms.py`**: Stage 1c (Cross-sectional normalisation, beta neutralisation).
- **`orthogonalization.py`**: Stage 1d (Hierarchical clustering, sequential Gram-Schmidt).
- **`labeling.py`**: Stage 2 (Dynamic triple barrier, Meta-labeling, cross-sectional relative-alpha targets, Sample weights).
- **`purged_cv.py`**: Stage 3 (Combinatorial Purged CV with embargo).
- **`multi_horizon_models.py`**: Stage 4 (Baseline LightGBM, Multi-task PyTorch with uncertainty).
- **`frictions.py`**: Stage 5 (Execution costs, impact models, hard gates).
- **`evaluation.py`**: Stage 6 (Deflated Sharpe, Probability of Backtest Overfitting, tearsheets).
