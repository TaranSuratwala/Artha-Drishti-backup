"""
Daily Automated Trading Pipeline
Combines Universe Screening, ML Prediction, and RL/Paper Trading execution.
"""
import os
import sys
import logging
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from IntegratedPostGreSQL import NSEDataPipeline
from MLPredictor import UnifiedStockPredictor
from PaperBroker import PaperBroker
from OrderManager import OrderManager
from BrokerAdapter import OrderRequest, OrderSide, BrokerOrderType, ProductType

class DailyTradingPipeline:
    def __init__(self):
        # 1. Initialize core infrastructure
        self.db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/StockDB")
        self.pipeline = NSEDataPipeline(self.db_url)
        
        # 2. Load the ML Predictor
        self.predictor = UnifiedStockPredictor(self.db_url)
        self.predictor._load_model()
        if self.predictor.model is None:
            logger.error("ML Model not found. Please train the model first.")
            sys.exit(1)
            
        # 3. Connect the Paper Trading Layer
        self.broker = PaperBroker(initial_capital=1000000.0, data_pipeline=self.pipeline)
        self.broker.connect({})
        
        # We need a dummy config for OrderManager
        class DummyConfig:
            MAX_ORDER_VALUE = 200000.0
            MAX_DAILY_LOSS = 20000.0
            MAX_POSITION_SIZE_PCT = 20.0
            MAX_ORDERS_PER_DAY = 20
            REQUIRE_CONFIRMATION_ABOVE = 1000000.0
            
        self.order_manager = OrderManager(broker=self.broker, config=DummyConfig())

    def get_tradeable_universe(self):
        """Step 1: Filter out illiquid/penny stocks to save ML inference time."""
        logger.info("Fetching universe of stocks...")
        latest_data = self.pipeline.get_latest_data()
        df = pd.DataFrame([dict(row) for row in latest_data])
        
        # Filter rules: Price > ₹50, Volume > 100,000
        filtered = df[(df['close'] > 50) & (df['volume'] > 100000)]
        tickers = filtered['ticker'].unique().tolist()
        logger.info(f"Filtered universe from {len(df['ticker'].unique())} to {len(tickers)} tradeable stocks.")
        
        return tickers

    def generate_predictions(self, tickers):
        """Step 2: Run the ML Predictor on the filtered universe."""
        logger.info("Running ML inference across the tradeable universe...")
        predictions = []
        
        for idx, ticker in enumerate(tickers):
            try:
                # Get historical data for the ticker
                history = self.pipeline.get_ticker_history(ticker)
                if not history or len(history) < 50:
                    continue
                    
                df_hist = pd.DataFrame([dict(row) for row in history])
                
                # Predict using the ML Engine
                pred = self.predictor.predict(ticker)
                if pred and 'signal_policy' in pred:
                    policy = pred['signal_policy']
                    
                    # 'confidence' is returned as a percentage (e.g. 54.1), so divide by 100
                    bull_prob = policy.get('confidence', 0) / 100.0
                    
                    predictions.append({
                        'ticker': ticker,
                        'current_price': df_hist.iloc[-1]['close'],
                        'bull_prob': bull_prob,
                        'details': policy
                    })
                    
            except Exception as e:
                logger.debug(f"Failed to predict {ticker}: {e}")
                
            if idx > 0 and idx % 20 == 0:
                logger.info(f"Processed {idx}/{len(tickers)} stocks...")
                
        return pd.DataFrame(predictions)

    def execute_trades(self, predictions_df, max_positions=5):
        """Step 3: Select top confident stocks and place paper orders."""
        if predictions_df.empty:
            logger.warning("No predictions generated.")
            return

        # Sort by ML confidence
        top_buys = predictions_df.sort_values(by='bull_prob', ascending=False)
        
        # Take the top N candidates (relative ranking instead of absolute thresholds)
        high_conviction = top_buys.head(max_positions)
        
        logger.info(f"Evaluating top {len(high_conviction)} candidates using Confidence Engine...")
        
        # Calculate Position Sizing (e.g., divide available capital equally)
        funds = self.broker.get_funds()
        available_cap = funds['available_capital']
        
        from confidence_engine import ConfidenceEngine, NSEContext, SizingConfig, size_position
        
        # Initialize the new ConfidenceEngine
        conf_engine = ConfidenceEngine(cfg=SizingConfig(equity=available_cap))
        
        if len(high_conviction) > 0:
            for _, row in high_conviction.iterrows():
                ticker = row['ticker']
                price = row['current_price']
                prob = row['bull_prob']
                details = row['details']
                
                # Setup NSE Context
                ctx = NSEContext(
                    price=price,
                    adv_inr=50000000,  # Example: 5Cr ADV
                    sigma_d=0.02,      # Example: 2% daily vol
                    spread_bps=10.0,
                    vix_z=0.0
                )
                
                # Run the Confidence Engine sizing (net-of-cost Kelly)
                # We mock draws with an array of identical probabilities if actual draws aren't available
                sizing_dec = conf_engine.assess(
                    draws=[prob] * 10,
                    vix_z=ctx.vix_z,
                    gain=0.05,
                    loss=0.02,
                    cost=0.002
                )
                
                # Size position
                pos = size_position(
                    p_win_lcb=sizing_dec['p_shrunk'],
                    gain=0.05,
                    loss=0.02,
                    edge_conf=sizing_dec['edge_confidence'],
                    ctx=ctx,
                    cfg=conf_engine.cfg
                )
                
                if not pos.trade:
                    logger.info(f"Skipping {ticker}: ConfidenceEngine veto -> {pos.reason}")
                    continue
                    
                qty = pos.shares
                logger.info(f"Placing BUY order for {ticker}: {qty} shares @ ~₹{price:.2f} (Kelly Fraction: {pos.fraction:.2%})")
                
                req = OrderRequest(
                    symbol=ticker,
                    side=OrderSide.BUY,
                    quantity=qty,
                    order_type=BrokerOrderType.MARKET,
                    price=price,
                    product=ProductType.CNC
                )
                
                # Send to OrderManager (which checks daily loss limits, writes atomic logs, etc.)
                resp = self.order_manager.validate_and_place(req)
                logger.info(f"Order Status: {resp.status.value}")

if __name__ == "__main__":
    pipeline = DailyTradingPipeline()
    universe = pipeline.get_tradeable_universe()
    preds = pipeline.generate_predictions(universe)
    pipeline.execute_trades(preds)
