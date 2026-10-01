"""
Jev (TypeSafe AI) pre-trade execution gate.
Evaluates market microstructure conditions before order placement.
Falls back to heuristic chain if Jev is unavailable.
"""
import os
import time
import logging
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Type definitions matching Jev's API contract
ExecutionDecision = Literal["EXECUTE", "WAIT", "SKIP"]


@dataclass
class MarketSnapshot:
    """Structured state blob sent to Jev for evaluation."""
    ticker: str
    ml_probability: float          # Calibrated P(bull) from ensemble
    ensemble_std: float            # Cross-seed disagreement
    bid_ask_spread_bps: float      # Current bid/ask spread in basis points
    volume_ratio_20d: float        # Today's volume / 20-day average
    psi_drift_score: float         # Population Stability Index
    regime_state: str              # "bull" | "bear" | "sideways"
    atr_pct: float                 # ATR as % of price (volatility proxy)
    signal_type: str               # "BUY" | "SELL"
    position_size_pct: float       # Proposed position as % of portfolio
    days_since_last_trade: int     # Cooldown period


@dataclass
class JevDecision:
    """Typed response from Jev."""
    action: ExecutionDecision
    action_confidence: float       # P(action) from Jev's probability distribution
    execution_quality_score: float # 0-10 score on execution conditions
    is_liquid: bool                # Noul (boolean) judgment
    is_liquid_confidence: float    # P(is_liquid)
    latency_ms: float
    fallback_used: bool = False


class JevPreTradeGate:
    """
    Pre-trade gate using Jev (TypeSafe AI) for execution decisions.
    Runs AFTER ML prediction, BEFORE order placement.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_ms: int = 500,
        min_execute_confidence: float = 0.70,
        min_liquidity_confidence: float = 0.80,
        enable_fallback: bool = True,
    ):
        self.api_key = api_key or os.getenv('TYPESAFE_API_KEY', '')
        self.timeout_s = timeout_ms / 1000.0
        self.min_execute_confidence = min_execute_confidence
        self.min_liquidity_confidence = min_liquidity_confidence
        self.enable_fallback = enable_fallback
        self._client = None
    
    def _get_client(self):
        """Lazy-init the TypeSafe SDK client."""
        if self._client is None:
            try:
                from typesafe_sdk import TypeSafe
                self._client = TypeSafe(api_key=self.api_key)
            except ImportError:
                logger.warning(
                    "typesafe-sdk not installed. Install with: pip install typesafe-sdk"
                )
                return None
        return self._client
    
    def _build_state_text(self, snapshot: MarketSnapshot) -> str:
        """Convert MarketSnapshot to a state description for Jev."""
        return (
            f"Stock: {snapshot.ticker}. "
            f"ML model predicts {snapshot.signal_type} with probability {snapshot.ml_probability:.3f} "
            f"(ensemble disagreement σ={snapshot.ensemble_std:.3f}). "
            f"Market regime: {snapshot.regime_state}. "
            f"Current bid-ask spread: {snapshot.bid_ask_spread_bps:.1f} bps. "
            f"Volume today is {snapshot.volume_ratio_20d:.1f}x the 20-day average. "
            f"Feature drift (PSI): {snapshot.psi_drift_score:.2f}. "
            f"ATR volatility: {snapshot.atr_pct:.1f}% of price. "
            f"Proposed position: {snapshot.position_size_pct:.1f}% of portfolio. "
            f"Days since last trade in this stock: {snapshot.days_since_last_trade}."
        )
    
    def evaluate(self, snapshot: MarketSnapshot) -> JevDecision:
        """
        Query Jev for a pre-trade execution decision.
        Falls back to heuristic chain if Jev is unavailable.
        """
        client = self._get_client()
        
        if client is None:
            if self.enable_fallback:
                return self._heuristic_fallback(snapshot)
            raise RuntimeError("Jev client unavailable and fallback disabled")
        
        state_text = self._build_state_text(snapshot)
        t0 = time.time()
        
        try:
            # All three questions in a single parallel pass (Jev processes
            # all questions about a state simultaneously)
            response = client.system_one(
                state=state_text,
                questions=[
                    {
                        "type": "choice",
                        "name": "execution_action",
                        "description": (
                            "Should this trade be executed right now, "
                            "delayed until conditions improve, or skipped entirely? "
                            "Consider the model confidence, spread, liquidity, and drift."
                        ),
                        "options": ["EXECUTE", "WAIT", "SKIP"],
                    },
                    {
                        "type": "score",
                        "name": "execution_quality",
                        "description": (
                            "Rate the overall execution quality of current market "
                            "conditions for this trade from 0 (terrible) to 10 (ideal). "
                            "Consider spread, volume, volatility, and model confidence."
                        ),
                        "min": 0,
                        "max": 10,
                    },
                    {
                        "type": "noul",  # Boolean judgment
                        "name": "is_liquid_enough",
                        "description": (
                            "Is this stock liquid enough for the proposed position size? "
                            "Consider the volume ratio and bid-ask spread."
                        ),
                    },
                ],
                timeout=self.timeout_s,
            )
            
            latency_ms = (time.time() - t0) * 1000
            
            # Parse typed response
            action_result = response.get_choice("execution_action")
            quality_result = response.get_score("execution_quality")
            liquidity_result = response.get_noul("is_liquid_enough")
            
            decision = JevDecision(
                action=action_result.selected,
                action_confidence=action_result.probability,
                execution_quality_score=quality_result.value,
                is_liquid=liquidity_result.answer,
                is_liquid_confidence=liquidity_result.probability,
                latency_ms=latency_ms,
            )
            
            logger.info(
                f"Jev decision for {snapshot.ticker}: "
                f"{decision.action} (p={decision.action_confidence:.2f}), "
                f"quality={decision.execution_quality_score:.1f}/10, "
                f"liquid={decision.is_liquid} (p={decision.is_liquid_confidence:.2f}), "
                f"latency={decision.latency_ms:.0f}ms"
            )
            
            return decision
            
        except Exception as e:
            logger.warning(f"Jev query failed ({e}), falling back to heuristics")
            if self.enable_fallback:
                return self._heuristic_fallback(snapshot)
            raise
    
    def _heuristic_fallback(self, snapshot: MarketSnapshot) -> JevDecision:
        """
        Deterministic fallback matching the existing safety guard chain.
        Maps the current if/else logic from _generate_signal() into a
        JevDecision-compatible response.
        """
        action: ExecutionDecision = "EXECUTE"
        quality = 7.0
        is_liquid = True
        reasons = []
        
        # PSI drift gate (from CONFIG at L988-990)
        if snapshot.psi_drift_score > 1.0:
            action = "SKIP"
            quality -= 5.0
            reasons.append(f"PSI={snapshot.psi_drift_score:.2f} > critical(1.0)")
        elif snapshot.psi_drift_score > 0.5:
            action = "WAIT"
            quality -= 3.0
            reasons.append(f"PSI={snapshot.psi_drift_score:.2f} > severe(0.5)")
        
        # Ensemble disagreement gate
        if snapshot.ensemble_std > 0.08:
            if action == "EXECUTE":
                action = "WAIT"
            quality -= 2.0
            reasons.append(f"ensemble_σ={snapshot.ensemble_std:.3f} > 0.08")
        
        # Liquidity gate
        if snapshot.volume_ratio_20d < 0.3:
            is_liquid = False
            if action == "EXECUTE":
                action = "WAIT"
            quality -= 2.0
            reasons.append(f"vol_ratio={snapshot.volume_ratio_20d:.1f} < 0.3")
        
        # Spread gate (>50 bps is expensive for NSE)
        if snapshot.bid_ask_spread_bps > 50:
            quality -= 2.0
            reasons.append(f"spread={snapshot.bid_ask_spread_bps:.0f}bps > 50")
        
        # ML confidence gate
        if snapshot.signal_type == "BUY" and snapshot.ml_probability < 0.55:
            if action == "EXECUTE":
                action = "WAIT"
            reasons.append(f"weak BUY prob={snapshot.ml_probability:.3f}")
        
        quality = max(0.0, min(10.0, quality))
        
        decision = JevDecision(
            action=action,
            action_confidence=0.95 if not reasons else 0.60,
            execution_quality_score=quality,
            is_liquid=is_liquid,
            is_liquid_confidence=0.90 if is_liquid else 0.80,
            latency_ms=0.1,
            fallback_used=True,
        )
        
        if reasons:
            logger.info(f"Heuristic fallback for {snapshot.ticker}: {action} ({'; '.join(reasons)})")
        
        return decision
    
    def should_execute(self, decision: JevDecision) -> bool:
        """
        Final binary gate: should the order be placed?
        Confidence-gated routing pattern.
        """
        if decision.action != "EXECUTE":
            return False
        if decision.action_confidence < self.min_execute_confidence:
            logger.info(
                f"EXECUTE confidence {decision.action_confidence:.2f} "
                f"< threshold {self.min_execute_confidence:.2f}, blocking"
            )
            return False
        if not decision.is_liquid and decision.is_liquid_confidence > self.min_liquidity_confidence:
            logger.info("Liquidity check failed with high confidence, blocking")
            return False
        return True
