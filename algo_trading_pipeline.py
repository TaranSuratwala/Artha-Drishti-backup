"""
Artha Drishti - Algorithmic Trading Execution Pipeline (v2 — Confidence Fix)
Target: NSE via KiteConnect / PaperBrokerAdapter
Author: Principal Quantitative Developer

FIXES APPLIED (v2):
  1. Model stub now matches real MultiTargetStockModel output interface (dict of logits)
  2. Confidence derived from calibrated direction probability, matching MLPredictor.py L12936-12939
  3. JEV pretrade gate heuristic fallback integrated as execution-quality confidence axis
  4. ADCI-style composite confidence replaces raw random confidence
  5. MC Dropout simulation for realistic probability distribution
"""

import asyncio
import logging
import time
import json
import csv
import os
import math
import concurrent.futures
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import joblib

# Optional: use uvloop for high-performance event loop
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("ArthaDrishti.Pipeline")

# =====================================================================
# Domain Models & Interfaces
# =====================================================================

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class BrokerOrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class ProductType(Enum):
    CNC = "CNC"
    MIS = "MIS"

class OrderRequest:
    def __init__(self, symbol: str, quantity: int, side: OrderSide,
                 order_type: BrokerOrderType, product: ProductType, price: float = 0.0):
        self.symbol = symbol
        self.quantity = quantity
        self.side = side
        self.order_type = order_type
        self.product = product
        self.price = price

class PaperBrokerAdapter:
    """Mock for BrokerAdapter.py"""
    async def place_order(self, request: OrderRequest) -> str:
        logger.info(f"[EXECUTION] Order placed: {request.side.name} {request.quantity} {request.symbol}")
        return "MOCK_ORDER_123"


# =====================================================================
# Model Stub — Now matches REAL MultiTargetStockModel output interface
# =====================================================================

class MultiTargetStockModelStub(nn.Module):
    """
    Stub that matches the REAL MultiTargetStockModel.forward() return
    signature: a dict of raw logits, NOT a tuple of classification heads.

    The real model (MLPredictor.py L3254-3294) returns:
      {
        'direction':    Tensor[B, 1]  — raw logit (pre-sigmoid)
        'price':        Tensor[B, 3]  — quantile predictions [P10, P50, P90]
        'target':       Tensor[B, 1]  — target move (post-softplus)
        'volatility':   Tensor[B, 1]  — vol prediction (post-softplus)
        'direction_3d': Tensor[B, 1]  — 3-day horizon logit
        'direction_7d': Tensor[B, 1]  — 7-day horizon logit
      }

    There is NO confidence head, NO risk head, NO action head in the neural net.
    All of those are derived procedurally downstream.
    """
    def __init__(self, input_features: int = 150, seq_len: int = 40, hidden_dim: int = 128):
        super().__init__()
        flat_dim = input_features * seq_len
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.direction_head = nn.Linear(hidden_dim, 1)
        self.price_head = nn.Linear(hidden_dim, 3)
        self.target_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.volatility_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.direction_3d_head = nn.Linear(hidden_dim, 1)
        self.direction_7d_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = x.size(0)
        flat = x.view(batch, -1)
        h = self.encoder(flat)
        return {
            'direction': self.direction_head(h),            # raw logit
            'price': self.price_head(h),                    # [P10, P50, P90]
            'target': self.target_head(h),                  # softplus
            'volatility': self.volatility_head(h),          # softplus
            'direction_3d': self.direction_3d_head(h),      # raw logit
            'direction_7d': self.direction_7d_head(h),      # raw logit
        }


# =====================================================================
# JEV Heuristic Confidence Evaluator (from jev_pretrade_gate.py)
# =====================================================================

@dataclass
class ExecutionQuality:
    """Lightweight execution quality assessment (JEV heuristic fallback)."""
    quality_score: float        # 0.0 - 10.0
    is_executable: bool
    reasons: List[str] = field(default_factory=list)

class JevHeuristicEvaluator:
    """
    Lightweight port of the heuristic fallback from jev_pretrade_gate.py
    (L209-272). Evaluates execution environment quality independent of
    the ML model's directional probability, providing an orthogonal
    confidence axis that doesn't suffer from calibration compression.
    """

    def evaluate(
        self,
        direction_prob: float,
        direction_std: float,
        signal_type: str,
        volume_ratio: float = 1.0,
        spread_bps: float = 25.0,
        psi_drift: float = 0.0,
        atr_pct: float = 2.0,
    ) -> ExecutionQuality:
        quality = 7.0
        reasons: List[str] = []
        is_executable = True

        # PSI drift gate
        if psi_drift > 1.0:
            is_executable = False
            quality -= 5.0
            reasons.append(f"PSI={psi_drift:.2f}>critical(1.0)")
        elif psi_drift > 0.5:
            quality -= 3.0
            reasons.append(f"PSI={psi_drift:.2f}>severe(0.5)")

        # Ensemble/MC disagreement gate
        if direction_std > 0.08:
            quality -= 2.0
            reasons.append(f"MC_std={direction_std:.3f}>0.08")

        # Liquidity gate
        if volume_ratio < 0.3:
            quality -= 2.0
            is_executable = False
            reasons.append(f"vol_ratio={volume_ratio:.1f}<0.3")

        # Spread gate
        if spread_bps > 50:
            quality -= 2.0
            reasons.append(f"spread={spread_bps:.0f}bps>50")

        # ML probability gate
        if signal_type == "BUY" and direction_prob < 0.55:
            quality -= 1.0
            reasons.append(f"weak_BUY_prob={direction_prob:.3f}")
        elif signal_type == "SELL" and direction_prob > 0.45:
            quality -= 1.0
            reasons.append(f"weak_SELL_prob={direction_prob:.3f}")

        # Volatility regime penalty
        if atr_pct > 5.0:
            quality -= 1.5
            reasons.append(f"high_volatility_atr={atr_pct:.1f}%")

        quality = max(0.0, min(10.0, quality))

        return ExecutionQuality(
            quality_score=quality,
            is_executable=is_executable,
            reasons=reasons,
        )


# =====================================================================
# 1. Asynchronous WebSocket Data Ingestion
# =====================================================================

class KiteDataStreamer:
    """
    Manages async WS connection to KiteTicker.
    Implements exponential backoff, heartbeat, and tick-to-candle aggregation.
    """
    def __init__(self, symbols: List[str], api_key: str):
        self.symbols = symbols
        self.api_key = api_key
        self.is_connected = False
        self.tick_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 8
        self.last_tick_time = time.time()

    async def connect(self) -> None:
        self.is_connected = True
        logger.info(f"[STREAMER] Connected to WS for symbols: {self.symbols}")
        asyncio.create_task(self._mock_stream_producer())
        asyncio.create_task(self._circuit_breaker_monitor())

    async def _mock_stream_producer(self) -> None:
        """Mock WebSocket listener putting ticks into an async queue."""
        while self.is_connected:
            await asyncio.sleep(0.5)
            for symbol in self.symbols:
                tick = {
                    "symbol": symbol,
                    "ltp": float(np.random.normal(1500, 2)),
                    "volume": int(np.random.randint(1, 100)),
                    "timestamp": time.time()
                }
                await self.tick_queue.put(tick)
                self.last_tick_time = time.time()

    async def _circuit_breaker_monitor(self) -> None:
        """Monitors heartbeat/ping-pong."""
        while self.is_connected:
            await asyncio.sleep(5)
            if time.time() - self.last_tick_time > 30:
                logger.error("[STREAMER] Connection lost (>30s no tick). Triggering reconnect.")
                await self.reconnect()

    async def reconnect(self) -> None:
        self.is_connected = False
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.critical("[STREAMER] Max reconnect attempts reached. Halting stream.")
            return
        base_delay = min(60, 2 ** self.reconnect_attempts)
        jitter = np.random.uniform(0, 1)
        delay = base_delay + jitter
        logger.warning(f"[STREAMER] Reconnecting in {delay:.2f} seconds...")
        await asyncio.sleep(delay)
        self.reconnect_attempts += 1
        await self.connect()


# =====================================================================
# 2. Real-Time Feature Pipeline
# =====================================================================

class StreamingFeatureEngine:
    """
    Maintains sliding window of OHLCV candles and computes technical indicators.
    """
    def __init__(self, lookback: int = 40):
        self.lookback = lookback
        self.history: Dict[str, deque] = {}
        # Load pre-trained artifacts (commented for mock)
        # self.scaler = joblib.load("backend/unified_models/feature_scaler.pkl")
        # self.feature_cols = joblib.load("backend/unified_models/feature_cols.pkl")
        self.feature_cols = [f"f_{i}" for i in range(150)]

    def aggregate_tick(self, symbol: str, tick: Dict[str, Any]) -> None:
        """Aggregates tick data into 1-minute OHLCV candles."""
        if symbol not in self.history:
            self.history[symbol] = deque(maxlen=250)
        candle = {
            "open": tick["ltp"], "high": tick["ltp"] + 1, "low": tick["ltp"] - 1,
            "close": tick["ltp"], "volume": tick["volume"], "timestamp": tick["timestamp"]
        }
        self.history[symbol].append(candle)

    def get_model_inputs(self, symbol: str) -> Optional[np.ndarray]:
        """Returns normalized feature tensor of shape (T, Features)."""
        if symbol not in self.history or len(self.history[symbol]) < self.lookback:
            return None
        # Mock feature generation — in production, compute via pandas_ta
        features = np.random.randn(len(self.history[symbol]), len(self.feature_cols))
        window = features[-self.lookback:]
        if window.shape != (self.lookback, len(self.feature_cols)):
            logger.error(f"[FEATURE ENGINE] Dimension mismatch for {symbol}")
            return None
        return window


# =====================================================================
# 3. Non-Blocking Neural Inference — FIXED CONFIDENCE DERIVATION
# =====================================================================

class AsyncNeuralPredictor:
    """
    Offloads PyTorch forward passes to a ThreadPoolExecutor.

    CRITICAL FIX: Confidence is now DERIVED from calibrated direction
    probability using the same formula as MLPredictor.py L12936-12939:

        direction_confidence = |P(bull) - threshold| / denom

    This replaces the incorrect phantom confidence head that was returning
    random uniform values (mean ~0.50), causing chronic low confidence.

    Additionally integrates JEV heuristic evaluator for composite scoring.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.50,   # Lowered: now correctly derived
        dir_decision_threshold: float = 0.50,  # Matches _optimal_dir_threshold
        temperature: float = 1.0,              # Post-training calibration T
        mc_samples: int = 5,                   # Monte Carlo dropout passes
        buy_threshold: float = 0.55,           # Dynamic buy threshold from checkpoint
        sell_threshold: float = 0.35,          # Dynamic sell threshold from checkpoint
    ):
        self.model = MultiTargetStockModelStub().eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.confidence_threshold = confidence_threshold
        self.dir_threshold = dir_decision_threshold
        self.temperature = temperature
        self.mc_samples = mc_samples
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.jev_evaluator = JevHeuristicEvaluator()

    def _calibrated_sigmoid(self, logit: float) -> float:
        """Apply temperature-scaled sigmoid: sigma(logit / T)."""
        scaled = logit / max(self.temperature, 1e-4)
        scaled = np.clip(scaled, -30, 30)
        return float(1.0 / (1.0 + math.exp(-scaled)))

    def _derive_direction_confidence(self, direction_prob: float) -> float:
        """
        Exact replica of MLPredictor.py L12936-12939:
            _dir_thr = _optimal_dir_threshold
            is_bullish = direction_prob > _dir_thr
            _confidence_denom = (1.0 - _dir_thr) if is_bullish else _dir_thr
            direction_confidence = min(|direction_prob - _dir_thr| / denom, 1.0)
        """
        thr = self.dir_threshold
        is_bullish = direction_prob > thr
        denom = (1.0 - thr) if is_bullish else thr
        return min(abs(direction_prob - thr) / max(denom, 1e-8), 1.0)

    def _compute_composite_confidence(
        self,
        direction_confidence: float,
        jev_quality: float,
        direction_std: float,
    ) -> float:
        """
        ADCI-inspired composite confidence (simplified 3-axis):
          - 50% ML direction confidence (derived from |P - threshold|)
          - 30% JEV execution quality (0-10 normalized to 0-1)
          - 20% prediction certainty (inverse of MC dropout std)
        """
        ml_axis = direction_confidence
        jev_axis = jev_quality / 10.0
        certainty_axis = max(1.0 - direction_std / 0.15, 0.0)

        composite = 0.50 * ml_axis + 0.30 * jev_axis + 0.20 * certainty_axis
        return min(max(composite, 0.0), 1.0)

    def _predict_sync(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Run MC Dropout inference, calibrate probabilities, derive confidence
        using the same methodology as the real MLPredictor.predict().
        """
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # === Phase 1: MC Dropout forward passes ===
        # Enable dropout during inference (matching MLPredictor.py L12753-12755)
        self.model.train()  # re-enables dropout
        direction_probs: List[float] = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                preds = self.model(x)
                raw_logit = preds['direction'].cpu().item()
                prob = self._calibrated_sigmoid(raw_logit)
                direction_probs.append(prob)

        self.model.eval()

        # === Phase 2: Aggregate MC samples ===
        direction_prob = float(np.mean(direction_probs))
        direction_std = float(np.std(direction_probs))

        # === Phase 3: Derive confidence (matching MLPredictor.py L12936-12939) ===
        direction_confidence = self._derive_direction_confidence(direction_prob)

        # === Phase 4: Determine signal (matching _generate_signal asymmetric thresholds) ===
        is_bullish = direction_prob > self.dir_threshold
        signal_type = "BUY" if is_bullish else "SELL"

        if direction_prob > self.buy_threshold:
            if direction_prob > self.buy_threshold + 0.10 and direction_confidence > 0.5:
                action = "Strong Buy"
            else:
                action = "Buy"
        elif direction_prob < self.sell_threshold:
            if direction_prob < self.sell_threshold - 0.10 and direction_confidence > 0.5:
                action = "Strong Sell"
            else:
                action = "Sell"
        else:
            action = "Hold"

        # === Phase 5: JEV execution quality (orthogonal confidence axis) ===
        jev_result = self.jev_evaluator.evaluate(
            direction_prob=direction_prob,
            direction_std=direction_std,
            signal_type=signal_type,
        )

        # === Phase 6: Composite confidence ===
        composite_confidence = self._compute_composite_confidence(
            direction_confidence=direction_confidence,
            jev_quality=jev_result.quality_score,
            direction_std=direction_std,
        )

        # === Phase 7: Actionability gate ===
        is_actionable = (
            composite_confidence >= self.confidence_threshold
            and action != "Hold"
            and jev_result.is_executable
        )

        return {
            "action": action,
            "confidence": composite_confidence,
            "direction_prob": direction_prob,
            "direction_confidence": direction_confidence,
            "direction_std": direction_std,
            "jev_quality": jev_result.quality_score,
            "jev_reasons": jev_result.reasons,
            "is_actionable": is_actionable,
            "trend_strength": direction_confidence,  # proxy for compatibility
        }

    async def predict(self, features: np.ndarray) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._predict_sync, features)


# =====================================================================
# 4. Execution Manager with Paper Trade Logging
# =====================================================================

class ArthaExecutionManager:
    """Risk filters, state reconciliation, and order dispatching with Paper Trade Logging."""

    def __init__(self, broker: PaperBrokerAdapter, log_file: str = "paper_trades_log.csv"):
        self.broker = broker
        self.open_positions: Dict[str, int] = {}
        self.daily_pnl = 0.0
        self.max_daily_loss = -10000.0
        self.log_file = log_file

        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "side", "quantity", "price",
                    "composite_confidence", "direction_prob", "direction_confidence",
                    "jev_quality", "action_label",
                ])

    def _log_trade(self, symbol: str, side: OrderSide, quantity: int,
                   price: float, signal: Dict[str, Any], is_square_off: bool = False) -> None:
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(), symbol, side.name, quantity, price,
                round(signal.get("confidence", 0.0), 4),
                round(signal.get("direction_prob", 0.0), 4),
                round(signal.get("direction_confidence", 0.0), 4),
                round(signal.get("jev_quality", 0.0), 2),
                "SQUARE_OFF" if is_square_off else signal.get("action", ""),
            ])

    async def process_signal(self, symbol: str, signal: Dict[str, Any],
                             current_price: float) -> None:
        if self.daily_pnl <= self.max_daily_loss:
            logger.warning("[RISK GATE] Max daily loss ceiling breached. Execution halted.")
            return

        action = signal["action"]
        side = OrderSide.BUY if "Buy" in action else OrderSide.SELL
        quantity = 50

        current_pos = self.open_positions.get(symbol, 0)
        if (side == OrderSide.BUY and current_pos > 0) or \
           (side == OrderSide.SELL and current_pos < 0):
            return

        if current_pos != 0:
            flatten_side = OrderSide.BUY if current_pos < 0 else OrderSide.SELL
            flatten_req = OrderRequest(symbol, abs(current_pos), flatten_side,
                                       BrokerOrderType.MARKET, ProductType.CNC)
            await self.broker.place_order(flatten_req)
            self.open_positions[symbol] = 0
            self._log_trade(symbol, flatten_side, abs(current_pos), current_price,
                           signal, is_square_off=True)
            logger.info(f"[EXECUTION] Squared off opposite position for {symbol}")

        req = OrderRequest(symbol, quantity, side, BrokerOrderType.MARKET, ProductType.CNC)
        await self.broker.place_order(req)
        self.open_positions[symbol] = quantity if side == OrderSide.BUY else -quantity
        self._log_trade(symbol, side, quantity, current_price, signal)
        logger.info(
            f"[EXECUTION] {side.name} {quantity} {symbol} | "
            f"Conf={signal['confidence']:.2f} "
            f"(Dir={signal['direction_confidence']:.2f}, "
            f"JEV={signal['jev_quality']:.1f}/10, "
            f"P(bull)={signal['direction_prob']:.3f})"
        )


# =====================================================================
# Main Event Loop Orchestrator
# =====================================================================

async def main() -> None:
    universe = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    api_key = "3fa6f770-e8d8-444f-8eb8-851c0169a887"

    streamer = KiteDataStreamer(universe, api_key)
    feature_engine = StreamingFeatureEngine(lookback=40)
    predictor = AsyncNeuralPredictor(
        confidence_threshold=0.50,
        dir_decision_threshold=0.50,
        temperature=1.0,
        mc_samples=5,
        buy_threshold=0.55,
        sell_threshold=0.35,
    )
    broker = PaperBrokerAdapter()
    execution_manager = ArthaExecutionManager(broker)

    await streamer.connect()
    logger.info("[SYSTEM] Pipeline v2 started (confidence fix applied).")

    while True:
        tick = await streamer.tick_queue.get()
        t0 = time.perf_counter()
        symbol = tick["symbol"]

        feature_engine.aggregate_tick(symbol, tick)
        features = feature_engine.get_model_inputs(symbol)

        if features is not None:
            signal = await predictor.predict(features)

            if signal["is_actionable"]:
                await execution_manager.process_signal(symbol, signal, tick["ltp"])

        latency_ms = (time.perf_counter() - t0) * 1000
        if latency_ms > 50:
            logger.warning(f"[TELEMETRY] High latency for {symbol}: {latency_ms:.2f}ms")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[SYSTEM] Shutting down execution pipeline.")
