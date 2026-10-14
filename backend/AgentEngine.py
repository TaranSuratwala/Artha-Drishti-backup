import os
import re
import json
import time
import logging
import threading
from datetime import datetime
from typing import Generator, Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg_pool import ConnectionPool

from config import get_config
from jev_pretrade_gate import JevPreTradeGate, build_market_snapshot

logger = logging.getLogger(__name__)
config = get_config()

# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT — Artha Drishti: grounding rules, Jev pre-trade discipline,
#  structured response template, financial disclaimer
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Artha Drishti, an elite quantitative stock market intelligence assistant and portfolio strategist specializing in the Indian financial markets (NSE/BSE). You combine the rigorous analytical depth of Claude with the real-time synthesis and grounding of Gemini.

You are equipped with a real-time analytics suite, a quantitative strategy screener, a multi-target neural prediction engine, historical backtesting, vector document search, live web grounding, and the Jev (TypeSafe AI) pre-trade execution gate.

---

### I. CORE OPERATIONAL PRINCIPLES

1. ZERO TOLERANCE FOR HALLUCINATION:
   - Never fabricate, extrapolate, or estimate prices, percentage changes, dates, support/resistance levels, or ratios.
   - Every numerical data point MUST originate directly from an invoked tool or live web search result.
   - If a tool returns empty or failed data, explicitly state: "Data unavailable for this asset under current criteria." Never fill gaps with parametric training memory.
   - The example responses below are ILLUSTRATIVE FORMAT ONLY. Their numbers are placeholders. Never reuse them.

2. EPISTEMIC HUMILITY & PROBABILISTIC REASONING:
   - Markets are stochastic. Never make deterministic guarantees (e.g., do not say "The stock will hit ₹3,200"). Frame outputs probabilistically: "The multi-target model indicates a 68% directional probability favoring an upside trajectory toward ₹3,200, contingent on support at ₹3,050 holding."
   - Explicitly highlight data drift (PSI score), model confidence, model status (whether the model has cleared its own reliability bar), and volatility regimes. If a tool reports the model is NOT production ready, or a signal was suppressed to HOLD, say so plainly.

3. JEV PRE-TRADE EXECUTION DISCIPLINE:
   - Before executing or endorsing any trade, call `jev_pretrade_check` (or rely on `paper_trade`, which runs the same gate automatically for BUY orders). The gate evaluates market microstructure:
     * Liquidity conditions and bid-ask spread.
     * Population Stability Index (PSI) drift: if PSI > 0.5 (Severe) or > 1.0 (Critical), advise against execution ("WAIT" or "SKIP").
     * The Jev Execution Quality Score (0–10). Never execute a simulated trade when Jev signals "SKIP".
   - The bid-ask spread is an ESTIMATE derived from daily High/Low (Corwin–Schultz), not a live quote, unless a tool explicitly says otherwise. Always describe it as "estimated spread".
   - Report the gate's `source` honestly: "jev" means the TypeSafe model answered; "heuristic_fallback" means the deterministic rule chain was used.
   - If `paper_trade` returns BLOCKED_BY_JEV_GATE, explain the reasons to the user. For a WAIT block, only retry with `confirm_despite_wait=true` if the user has EXPLICITLY said they want to proceed anyway after seeing the warning. Never set it on your own initiative.
   - SELL / exit orders are not blocked by the gate (so users are never trapped in a position), but still mention gate warnings if relevant.

4. ACTION-FIRST PHILOSOPHY:
   - When the user asks to backtest, track, screen, or execute, invoke the relevant tool immediately. Never instruct the user to "use the UI" or "navigate to settings".
   - For paper trades, require ticker, side, and quantity. If quantity is unspecified (e.g., "buy Reliance"), ASK for quantity and propose a risk-based sizing calculation based on portfolio capital. Do not place the trade until the quantity is given.
   - Paper trades are SIMULATED — always state that no real money is used. After a trade, show ticker, quantity, price and total value.

---

### II. TOOLS

Analysis: `screen_stocks` (piotroski, momentum, swing, breakout, value), `predict_stock`, `get_market_overview`, `get_stock_history`, `get_sentiment`, `generate_pattern_chart`, `search_web`, `search_financial_docs`, `jev_pretrade_check`.
Actions: `manage_watchlist`, `run_backtest`, `create_strategy`, `list_strategies`, `compare_strategies`, `get_portfolio`, `add_transaction`.
Paper trading: `paper_trade`, `get_paper_positions`, `get_paper_order_history`.

Use tools for ALL factual questions about stocks, prices, predictions or market data. For backtests with no stated range, omit the dates (the tool defaults to the last 3 years).
For a single-stock outlook, call `predict_stock`, `get_sentiment` and `jev_pretrade_check` (side BUY unless the user is asking about exiting).

---

### III. RESPONSE MANNER & STRUCTURE

Every analytical response (stock outlooks, predictions, trade evaluations) must follow this structured template. Short action confirmations (watchlist changes, strategy creation, simple greetings) use a compact format instead.

1. 🎯 Executive Takeaway:
   A 2-sentence bottom-line synthesis answering the user's primary query with the overarching market posture.

2. 📊 Quantitative & Microstructure Dashboard:
   Markdown table displaying price, predicted trajectory, model confidence, ATR volatility, and Jev execution metrics (use only values returned by tools).
   | Metric | Value | Benchmark / Status |
   | :--- | :--- | :--- |
   | Current Price | ₹X,XXX.XX | NSE (as of data date) |
   | Multi-Target Signal | BUY / HOLD / SELL | Confidence: XX.X% |
   | Jev Gate Action | EXECUTE / WAIT / SKIP | Quality Score: X.X/10 |
   | Liquidity / Est. Spread | Liquid (p=0.XX) | Est. spread: XX bps |
   | Model Drift (PSI) | 0.XX | Stable (< 0.25) / Caution / Severe (> 0.5) |

3. 🔍 Deep Technical & Fundamental Rationale:
   Concise bullet points highlighting:
   - Factor alignment (e.g., Piotroski F-score, RSI, MACD crossovers, Volume Ratio).
   - Institutional volume flow and price action relative to key SMAs (20/50/200).
   - Catalyst / News sentiment score (-1.0 to +1.0) and corporate event flags.

4. 🛡️ Risk Budgeting & Invalidation Levels:
   - Primary Invalidation / Stop-Loss Level (and percentage distance).
   - Target Resistance / Profit Milestone.
   - Risk-to-Reward Ratio (R:R).
   If the signal is HOLD or the tool returned no trade setup, say there is no active setup instead of inventing levels.

5. ⚡ Proactive Follow-ups:
   Offer two high-value next steps (e.g., "Would you like me to backtest a 3-year momentum strategy on this stock, or screen peer sector stocks with lower volatility?").

Use markdown. Be concise: lead with the key insight, then supporting detail. For predictions, always mention confidence level and timeframe.

---

### IV. FEW-SHOT FORMAT EXAMPLES (ILLUSTRATIVE — NUMBERS ARE PLACEHOLDERS)

#### Example 1: Comprehensive Stock Analysis with Jev Gate
User: "What's the outlook on INFY right now?"
Reasoning: The user seeks multi-dimensional analysis. Invoke `predict_stock("INFY")`, `get_sentiment("INFY")`, and `jev_pretrade_check("INFY", "BUY")`.
Response format:
> ### 🎯 Executive Takeaway
> **Infosys Ltd (INFY)** displays a cautious consolidation pattern. While the neural model indicates a moderate bullish bias (62% probability), the Jev pre-trade gate advises **WAIT** due to elevated estimated spread and mild feature drift.
>
> ### 📊 Quantitative & Microstructure Dashboard
> | Metric | Value | Assessment |
> | :--- | :--- | :--- |
> | Current Price | ₹[from tool] | [from tool] |
> | Directional Signal | Bullish (P=[from tool]) | Multi-target neural consensus |
> | Jev Gate Evaluation | **WAIT** (Quality: [from tool]/10) | Est. spread [from tool] bps, PSI=[from tool] |
> | 14-Day News Sentiment | [from tool] | [N] articles analyzed |
> | Invalidation Anchor | ₹[from tool] | [from tool] |
>
> ### 🔍 Technical & Factor Analysis
> - **Momentum & Regime:** [RSI / MACD readings from tool]
> - **Microstructure Warning:** [volume ratio and gate reasons from tool]
>
> ### 🛡️ Risk Strategy
> - **Trigger Entry / Protective Stop / Targets / R:R:** [only from tool data]
>
> ⚠️ *Disclaimer: AI-generated analysis for informational purposes only. Not financial advice. Consult a SEBI-registered advisor before investing.*
>
> 💡 *Next Steps: Would you like me to generate an interactive candlestick pattern chart for INFY, or run a 3-year backtest of our Swing strategy on this ticker?*

#### Example 2: Execution Request with Microstructure Rejection
User: "Buy 50 shares of TATASTEEL"
Reasoning: Call `paper_trade("TATASTEEL", "BUY", 50)`. The tool runs the Jev gate itself. If it returns BLOCKED_BY_JEV_GATE:
Response format:
> ⚠️ **Execution Paused by Jev Pre-Trade Gate**
>
> Your order to **BUY 50 shares of TATASTEEL** was intercepted by our risk and execution gate:
> - **Jev Action:** `[SKIP/WAIT]` (Execution Quality: [from tool]/10, source: [jev / heuristic_fallback])
> - **Reason:** [reasons from tool, e.g. PSI value, estimated spread, ensemble disagreement]
> - **Recommendation:** [a LIMIT order at a tool-derived price, or wait for conditions to improve]
>
> *Would you like me to place this as a LIMIT order instead?*
> 📝 *Nothing was executed. Paper trades are simulated; no real money is used.*

#### Example 3: Executed paper trade
> ✅ **Paper Trade Executed**
> | Detail | Value |
> |--------|-------|
> | Stock | INFY |
> | Side | BUY |
> | Quantity | [qty] |
> | Price | ₹[from tool] |
> | Total Value | ₹[from tool] |
> | Jev Gate | [action] ([source]) |
> 📝 *This is a paper (simulated) trade. No real money was used.*

---

### V. CHART RENDERING RULES
When you invoke `generate_pattern_chart(ticker)`, output the EXACT returned markdown string:
`![Interactive Chart for TICKER](chart:pattern:TICKER)`
Do not modify or wrap this syntax, as the UI parser mounts the interactive Plotly WebGL engine directly from this token.

### VI. MANDATORY REGULATORY COMPLIANCE
Every investment or analytical response MUST conclude with:
> ⚠️ *Disclaimer: This is AI-generated analysis for informational purposes only. It does not constitute financial advice. Please consult a SEBI-registered financial advisor before making real-world capital decisions.*
"""


def _build_system_prompt() -> str:
    """System prompt plus today's date (so date-dependent reasoning is never stale)."""
    return SYSTEM_PROMPT + f"\n\nToday's date: {datetime.now().strftime('%A, %d %B %Y')}."


class ArthaAgent:
    # How long a predictor.predict() result may be reused (seconds). predict() is
    # expensive and records a row in prediction_outcomes on every call, so repeated
    # tool calls within one conversation turn must not each trigger a fresh run.
    PREDICTION_CACHE_TTL_SECONDS = int(os.getenv("JEV_PREDICTION_CACHE_SECONDS", "300"))

    def __init__(self, screener, predictor, pipeline, db_url=None,
                 backtest_engine=None, strategy_engine=None,
                 portfolio_manager=None, order_manager=None):
        self.screener = screener
        self.predictor = predictor
        self.pipeline = pipeline
        self.db_url = db_url or config.SQLALCHEMY_DATABASE_URI
        self.max_history = config.AGENT_MAX_HISTORY
        self.backtest_engine = backtest_engine
        self.strategy_engine = strategy_engine
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager

        # Jev pre-trade gate + prediction cache
        try:
            self.jev_gate = JevPreTradeGate(
                timeout_ms=int(os.getenv("JEV_TIMEOUT_MS", "500")),
            )
        except Exception as e:
            logger.warning(f"Could not initialise Jev gate: {e}")
            self.jev_gate = None
        self._prediction_cache: Dict[str, Any] = {}
        self._prediction_cache_lock = threading.Lock()

        api_key = config.GOOGLE_API_KEY

        # Initialize LLMOps: Tracing (LangSmith)
        if os.getenv("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Artha-Drishti")

        # Initialize LLMOps: Caching
        try:
            cache = InMemoryCache()
            set_llm_cache(cache)
            logger.info("Caching initialized with InMemoryCache.")
        except Exception as e:
            logger.warning(f"Could not initialize Caching: {e}")

        # Initialize LLM
        if not api_key:
            logger.warning("GOOGLE_API_KEY is not set. Agent will not function correctly.")

        self.llm = ChatGoogleGenerativeAI(
            model=config.AGENT_MODEL,
            google_api_key=api_key,
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=8192,
        )

        # Pre-create grounded search LLM (reused by search_web tool)
        self._search_llm = ChatGoogleGenerativeAI(
            model=config.AGENT_MODEL,
            google_api_key=api_key,
            temperature=0.0,
            max_output_tokens=1024,
        ).bind(tools=[{"google_search": {}}])

        # Lazy-initialized PGVector vectorstore (reused by search_financial_docs)
        self._vectorstore: Optional[PGVector] = None

        # Connection pool for checkpointer (replaces per-request Connection.connect)
        self._pg_pool: Optional[ConnectionPool] = None
        self._init_connection_pool()

        # Define Tools
        self.tools = self._create_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Compile Graph
        self.graph = self._build_graph()

        # Pre-initialize checkpointer tables (once, not per-request)
        self._setup_checkpointer()

    def _init_connection_pool(self):
        """Initialize PostgreSQL connection pool for checkpointer."""
        try:
            conn_str = self._get_postgres_connection_string()
            if conn_str:
                self._pg_pool = ConnectionPool(
                    conn_str,
                    min_size=2,
                    max_size=10,
                    kwargs={"autocommit": True},
                )
                logger.info("PostgreSQL connection pool initialized for agent checkpointer.")
        except Exception as e:
            logger.warning(f"Could not initialize connection pool: {e}")
            self._pg_pool = None

    def _setup_checkpointer(self):
        """Run checkpointer table setup once at initialization."""
        try:
            if self._pg_pool:
                with self._pg_pool.connection() as conn:
                    checkpointer = PostgresSaver(conn)
                    checkpointer.setup()
            else:
                conn_str = self._get_postgres_connection_string()
                with Connection.connect(conn_str, autocommit=True) as conn:
                    checkpointer = PostgresSaver(conn)
                    checkpointer.setup()
            logger.info("Agent checkpointer tables verified.")
        except Exception as e:
            logger.warning(f"Checkpointer setup skipped (will retry on first request): {e}")

    @staticmethod
    def _sanitize_input(text: str) -> str:
        """Input sanitization with improved prompt injection detection."""
        if not text or not isinstance(text, str):
            return ""

        # Normalize unicode to catch homoglyph attacks
        import unicodedata
        normalized = unicodedata.normalize('NFKC', text).lower()

        # Guardrails: Prompt Injection Detection
        injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"system\s*prompt",
            r"you\s+are\s+now",
            r"bypass\s+(all\s+)?",
            r"disregard\s+(all\s+)?",
            r"forget\s+(all\s+)?previous",
            r"new\s+instructions?\s*:",
            r"act\s+as\s+(if\s+)?you",
            r"pretend\s+(to\s+be|you\s+are)",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, normalized):
                logger.warning(f"Potential prompt injection detected: {pattern}")
                return "I cannot fulfill that request."

        # Strip control characters except newline/tab
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Collapse excessive whitespace
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        # Limit length to prevent abuse (8000 chars ≈ ~2000 tokens)
        return text[:8000].strip()

    @staticmethod
    def _truncate_tool_output(result: str, max_chars: int = 6000) -> str:
        """Truncate large tool outputs with JSON-aware boundary detection."""
        if len(result) <= max_chars:
            return result
        truncated = result[:max_chars]
        # Try to close at a complete JSON record boundary
        last_brace = truncated.rfind('}')
        last_bracket = truncated.rfind(']')
        cut_point = max(last_brace, last_bracket)
        if cut_point > max_chars * 0.6:
            truncated = truncated[:cut_point + 1]
            if truncated.count('[') > truncated.count(']'):
                truncated = truncated.rstrip(',\n ') + ']'
        return truncated + '\n... (truncated — ask about specific stocks for full details)'

    # ═══════════════════════════════════════════════════════════════════════
    #  PREDICTION CACHE + SUMMARY + JEV GATE HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _canon_ticker(ticker: str) -> str:
        return str(ticker or "").strip().upper().replace(".NS", "")

    def _get_prediction_cached(self, ticker: str) -> Dict[str, Any]:
        """predictor.predict() with a short TTL cache (it is slow and writes DB rows per call)."""
        key = self._canon_ticker(ticker)
        now = time.time()
        with self._prediction_cache_lock:
            hit = self._prediction_cache.get(key)
            if hit and (now - hit[0]) < self.PREDICTION_CACHE_TTL_SECONDS:
                return hit[1]
        pred = self.predictor.predict(key)
        # Only cache successful predictions; errors should be retried next call.
        if isinstance(pred, dict) and "error" not in pred:
            with self._prediction_cache_lock:
                self._prediction_cache[key] = (time.time(), pred)
                if len(self._prediction_cache) > 200:
                    oldest = min(self._prediction_cache, key=lambda k: self._prediction_cache[k][0])
                    self._prediction_cache.pop(oldest, None)
        return pred

    def _summarize_prediction(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compact, LLM-friendly view of a predictor.predict() result. The raw result is
        far larger than the tool-output limit, so truncating it used to cut off key
        fields (PSI, regime, thresholds). Only fields the response template needs are kept.
        """
        pa = pred.get("price_analysis") or {}
        rec = pred.get("recommendation") or {}
        ts = pred.get("trade_setup") or {}
        drift = pred.get("data_drift") or {}
        drift_an = pred.get("drift_analysis") or {}
        regime = pred.get("market_regime") or {}
        mom = pred.get("momentum_factor") or {}
        pat = pred.get("pattern_analysis") or {}
        sent = pred.get("sentiment") or {}
        risk = pred.get("risk_management") or {}
        safety = pred.get("safety_report") or {}
        horizons = (pred.get("multi_horizon_analysis") or {}).get("horizons")

        cur, atr = pa.get("current_price"), pa.get("atr_20")
        atr_pct = round(atr / cur * 100, 2) if cur and atr else None

        scorecard = getattr(self.predictor, "_reliability_scorecard", None)
        if isinstance(scorecard, dict):
            model_status = {
                "production_ready": bool(scorecard.get("critical_checks_passed", False)),
                "score": f"{scorecard.get('score')}/{scorecard.get('max_score')}",
                "calibrated_test_accuracy_pct": scorecard.get("calibrated_test_accuracy_pct"),
                "backtest_sharpe": scorecard.get("backtest_sharpe"),
            }
        else:
            model_status = {"production_ready": None, "note": "No reliability scorecard on record — unvalidated."}

        patterns = [
            {"pattern": p.get("pattern_type"), "signal": p.get("signal"), "confidence": p.get("confidence")}
            for p in (pat.get("patterns_detected") or [])[:5]
            if isinstance(p, dict)
        ]

        return {
            "ticker": pred.get("ticker"),
            "generated_at": pred.get("timestamp"),
            "model_version": pred.get("model_version"),
            "model_status": model_status,
            "price": {
                "current": cur,
                "predicted_5d": pa.get("predicted_price_5d"),
                "expected_change_pct": pa.get("expected_change_pct"),
                "atr_20": atr,
                "atr_pct_of_price": atr_pct,
                "historical_volatility_annual_pct": pa.get("historical_volatility_annual"),
                "note": pa.get("price_prediction_note"),
            },
            "signal": {
                "signal": rec.get("signal"),
                "strength": rec.get("signal_strength"),
                "market_bias": rec.get("market_bias"),
                "direction_probability_pct": rec.get("direction_probability"),
                "confidence_score_pct": rec.get("confidence_score"),
                "mc_uncertainty_pct": rec.get("mc_uncertainty"),
                "buy_threshold_pct": rec.get("buy_threshold"),
                "sell_threshold_pct": rec.get("sell_threshold"),
                "decision_reason": rec.get("decision_reason"),
                "drift_guard_triggered": rec.get("drift_guard_triggered"),
                "warning": rec.get("signal_warning"),
            },
            "trade_setup": ts if ts else None,
            "multi_horizon_bullish_pct": horizons,
            "drift": {
                "psi": drift.get("psi"),
                "level": drift_an.get("drift_level"),
                "warning": drift.get("warning"),
            },
            "regime": {"regime": regime.get("regime"), "confidence": regime.get("confidence")},
            "momentum": {"score": mom.get("score"), "regime": mom.get("regime")},
            "patterns": {
                "detected": patterns,
                "confluence_score": pat.get("confluence_score"),
                "support_levels": (pat.get("support_levels") or [])[:3],
                "resistance_levels": (pat.get("resistance_levels") or [])[:3],
            },
            "sentiment": {
                "score": sent.get("sentiment_score"),
                "overall": sent.get("overall"),
                "agreement_with_ml": sent.get("agreement_with_ml"),
            },
            "risk_management": {
                "position_pct_of_capital": risk.get("position_pct_of_capital"),
                "suggested_quantity": risk.get("suggested_quantity"),
                "max_loss_amount": risk.get("max_loss_amount"),
            },
            "safety": {
                "overall_severity": safety.get("overall_severity"),
                "recommended_action": safety.get("recommended_action"),
                "market_open": (safety.get("market_hours") or {}).get("market_open"),
                "liquidity": {
                    "liquid": (safety.get("liquidity") or {}).get("liquid"),
                    "avg_volume_20d": (safety.get("liquidity") or {}).get("avg_volume_20d"),
                    "severity": (safety.get("liquidity") or {}).get("severity"),
                },
            },
            "investor_action": (pred.get("investor_action") or {}).get("summary"),
        }

    def _paper_capital(self) -> float:
        """Best-effort paper-trading capital for position-size % (falls back to env/default)."""
        default = float(os.getenv("PAPER_TRADING_CAPITAL", "1000000"))
        try:
            funds = self.order_manager.broker.get_funds() or {}
            for k in ("total_equity", "equity", "net", "total", "available_cash",
                      "cash", "balance", "available_margin"):
                v = funds.get(k)
                if isinstance(v, (int, float)) and v > 0:
                    return float(v)
        except Exception:
            pass
        return default

    def _days_since_last_trade(self, ticker: str) -> int:
        """0 if the symbol traded in today's order history, else a 30-day 'no recent trade known' cap."""
        try:
            if self.order_manager is not None:
                for o in self.order_manager.broker.get_order_history() or []:
                    if str(getattr(o, "symbol", "")).upper() == ticker:
                        return 0
        except Exception:
            pass
        return 30

    def _evaluate_jev_gate(self, ticker: str, side: str = "BUY", quantity: int = 0,
                           reference_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Run the Jev pre-trade gate for a proposed order. Never raises.
        Returns {"evaluated": False, "error": ...} if the gate could not be evaluated.
        """
        ticker = self._canon_ticker(ticker)
        side = str(side or "BUY").upper()
        if self.jev_gate is None:
            return {"evaluated": False, "ticker": ticker, "error": "Jev gate is not initialised."}
        try:
            pred = self._get_prediction_cached(ticker)
            if not isinstance(pred, dict) or "error" in pred:
                return {"evaluated": False, "ticker": ticker,
                        "error": (pred or {}).get("error", "Prediction unavailable")}

            ref_price = reference_price or (pred.get("price_analysis") or {}).get("current_price")
            notes: List[str] = []
            capital = self._paper_capital() if self.order_manager is not None else float(
                os.getenv("PAPER_TRADING_CAPITAL", "1000000"))
            if quantity and quantity > 0 and ref_price:
                position_pct = float(quantity) * float(ref_price) / max(capital, 1.0) * 100.0
            else:
                position_pct = 1.0
                notes.append("No quantity given — position size assumed at 1% of portfolio for this evaluation.")

            try:
                rows = self.pipeline.get_ticker_history(ticker) or []
                rows = [dict(r) if not isinstance(r, dict) else r for r in rows][-60:]
            except Exception as e:
                logger.debug(f"Jev gate: OHLCV fetch failed for {ticker}: {e}")
                rows = []

            snapshot, prov = build_market_snapshot(
                ticker, pred, rows, signal_type=side,
                position_size_pct=position_pct,
                days_since_last_trade=self._days_since_last_trade(ticker),
            )
            decision = self.jev_gate.evaluate(snapshot)
            execute_ok = self.jev_gate.should_execute(decision)

            return {
                "evaluated": True,
                "ticker": ticker,
                "side": side,
                "quantity": quantity or None,
                "reference_price": ref_price,
                "decision": decision.to_dict(),
                "should_execute": execute_ok,
                "inputs": {
                    "ml_probability": round(snapshot.ml_probability, 4),
                    "ensemble_std": round(snapshot.ensemble_std, 4),
                    "estimated_spread_bps": round(snapshot.bid_ask_spread_bps, 1),
                    "volume_ratio_20d": round(snapshot.volume_ratio_20d, 2),
                    "psi": round(snapshot.psi_drift_score, 4),
                    "regime": snapshot.regime_state,
                    "atr_pct": round(snapshot.atr_pct, 2),
                    "position_size_pct": round(snapshot.position_size_pct, 3),
                },
                "data_quality": {
                    "spread_is_estimate": True,
                    "estimated_fields": prov.get("estimated", []),
                    "defaulted_fields": prov.get("defaulted", []),
                    "notes": prov.get("notes", []) + notes,
                },
            }
        except Exception as e:
            logger.error(f"Jev gate evaluation failed for {ticker}: {e}", exc_info=True)
            return {"evaluated": False, "ticker": ticker, "error": f"Gate evaluation failed: {e}"}

    def _create_tools(self) -> List[Any]:
        screener = self.screener
        predictor = self.predictor
        pipeline = self.pipeline
        truncate = self._truncate_tool_output
        grounded_search_llm = self._search_llm
        get_vectorstore = self._get_or_create_vectorstore
        backtest_engine = self.backtest_engine
        strategy_engine = self.strategy_engine
        portfolio_manager = self.portfolio_manager
        order_manager = self.order_manager
        get_prediction = self._get_prediction_cached
        summarize_prediction = self._summarize_prediction
        run_jev_gate = self._evaluate_jev_gate
        jev_fail_open = os.getenv("JEV_GATE_FAIL_OPEN", "false").strip().lower() in ("1", "true", "yes", "on")

        # ═══════════════════════════════════════════════════════════
        # ANALYSIS TOOLS
        # ═══════════════════════════════════════════════════════════

        @tool
        def screen_stocks(strategy_name: str, max_results: int = 10) -> str:
            """
            Run a stock screening strategy on the Indian market (NSE).
            Supported strategies: piotroski, momentum, swing, breakout, value.
            Returns a JSON string of matching stocks with key metrics.
            """
            try:
                strategy_map = {
                    "piotroski": screener.piotroski_scan,
                    "momentum": screener.momentum_scan,
                    "swing": screener.swing_trading_scan,
                    "breakout": screener.breakout_scan,
                    "value": screener.value_investing_scan,
                }
                scan_fn = strategy_map.get(strategy_name)
                if not scan_fn:
                    return json.dumps({"error": f"Strategy '{strategy_name}' not supported. Use: {', '.join(strategy_map.keys())}"})

                df = scan_fn()
                if df is not None and not df.empty:
                    res = df.head(max_results).to_dict(orient="records")
                    return truncate(json.dumps({"stocks": res}, default=str))
                return json.dumps({"message": "No stocks matched this strategy's criteria right now."})
            except Exception as e:
                logger.error(f"screen_stocks error: {e}")
                return json.dumps({"error": f"Screening failed: {str(e)}"})

        @tool
        def predict_stock(ticker: str) -> str:
            """
            Get the AI price/direction prediction and technical analysis for a specific NSE stock ticker
            (e.g. 'RELIANCE', 'TCS', 'INFY').
            Returns a compact summary: signal, direction probability and confidence, trade setup,
            ATR volatility, feature-drift PSI, market regime, patterns, sentiment, safety flags and
            whether the model has cleared its own production-reliability bar.
            """
            try:
                res = get_prediction(ticker)
                if isinstance(res, dict) and "error" in res:
                    return json.dumps({"error": f"Prediction failed for {ticker}: {res['error']}"})
                return truncate(json.dumps(summarize_prediction(res), default=str))
            except Exception as e:
                logger.error(f"predict_stock error for {ticker}: {e}")
                return json.dumps({"error": f"Prediction failed for {ticker}: {str(e)}"})

        @tool
        def get_market_overview() -> str:
            """
            Get the latest market overview with price data for top NSE stocks.
            Returns recent prices, changes, and volume data.
            """
            try:
                data = pipeline.get_latest_data(limit=50)
                return truncate(json.dumps({"overview": data}, default=str))
            except Exception as e:
                logger.error(f"get_market_overview error: {e}")
                return json.dumps({"error": f"Could not fetch market data: {str(e)}"})

        @tool
        def get_stock_history(ticker: str) -> str:
            """
            Get the recent price history (OHLCV — Open, High, Low, Close, Volume) for a specific stock ticker.
            Returns the last 10 trading days.
            """
            try:
                data = pipeline.get_ticker_history(ticker)
                if data:
                    return json.dumps({"history": data[-10:]}, default=str)
                return json.dumps({"message": f"No price history found for {ticker}. Verify the ticker symbol."})
            except Exception as e:
                logger.error(f"get_stock_history error for {ticker}: {e}")
                return json.dumps({"error": f"Could not fetch history for {ticker}: {str(e)}"})

        @tool
        def get_sentiment(ticker: str) -> str:
            """
            Get news sentiment analysis for a stock ticker.
            Returns sentiment score (-1 to 1), trend, volume, event flags, and bullish/bearish ratio.
            Use this when the user asks about news sentiment, market mood, or recent events for a stock.
            """
            try:
                from SentimentEngine import get_sentiment_engine
                engine = get_sentiment_engine()
                detailed = engine.get_detailed_analysis(ticker, days_back=14)
                summary = {
                    "ticker": detailed.get("ticker"),
                    "overall_sentiment": detailed.get("overall_sentiment"),
                    "article_count": detailed.get("article_count"),
                    "aggregate": detailed.get("aggregate"),
                    "top_headlines": [
                        {"headline": a["headline"], "sentiment": a["sentiment"], "source": a["source"]}
                        for a in (detailed.get("articles") or [])[:5]
                    ],
                }
                return truncate(json.dumps(summary, default=str))
            except Exception as e:
                logger.error(f"get_sentiment error for {ticker}: {e}")
                return json.dumps({"error": f"Sentiment analysis failed for {ticker}: {str(e)}"})

        @tool
        def generate_pattern_chart(ticker: str) -> str:
            """
            Generate an interactive candlestick chart with detected bullish and bearish patterns for a stock.
            Returns a special markdown string that the frontend renders as an interactive Plotly chart.
            Use this when the user asks to see a chart, candlestick patterns, or visual analysis.
            """
            chart_md = f"![Interactive Chart for {ticker}](chart:pattern:{ticker})"
            return json.dumps({"chart": chart_md})

        @tool
        def search_web(query: str) -> str:
            """
            Search the web for current information — use for live stock prices, breaking news,
            IPO dates, earnings announcements, regulatory updates, or any data not available via other tools.
            Always prefer this over guessing when you need up-to-date information.
            """
            try:
                response = grounded_search_llm.invoke([
                    HumanMessage(content=f"Search the web and provide factual, up-to-date information for: {query}. "
                       f"Include sources. Focus on Indian stock market (NSE/BSE) context if relevant.")
                ])
                if hasattr(response, 'content') and response.content:
                    content = response.content
                    if isinstance(content, list):
                        texts = [p.get('text', '') if isinstance(p, dict) else str(p) for p in content]
                        return truncate(json.dumps({"results": ' '.join(texts)}, default=str))
                    return truncate(json.dumps({"results": str(content)}, default=str))
                return json.dumps({"message": "Web search returned no results."})
            except Exception as e:
                logger.error(f"search_web error: {e}")
                return json.dumps({"error": f"Web search failed: {str(e)}. Try rephrasing your question."})

        @tool
        def search_financial_docs(query: str, ticker: str = "") -> str:
            """
            Search the vector database (pgvector) for earnings call transcripts, SEBI filings, and annual reports.
            Useful for deep fundamental research on a company based on their official documents.
            """
            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    return json.dumps({"message": "Vector database is not configured."})

                full_query = f"{ticker} {query}".strip()
                docs = vectorstore.similarity_search(full_query, k=3)

                if docs:
                    results = [{"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")} for doc in docs]
                    return truncate(json.dumps({"documents": results}, default=str))
                return json.dumps({"message": "No relevant documents found in the vector database."})
            except Exception as e:
                logger.error(f"search_financial_docs error: {e}")
                return json.dumps({"error": f"Document search failed: {str(e)}"})

        @tool
        def jev_pretrade_check(ticker: str, side: str = "BUY", quantity: int = 0) -> str:
            """
            Run the Jev (TypeSafe AI) pre-trade execution gate for a proposed order on an NSE stock.
            Evaluates ML signal strength, model disagreement, feature drift (PSI), liquidity,
            ESTIMATED bid-ask spread (derived from daily High/Low, not a live quote), volume vs the
            20-day average and ATR volatility.
            Returns the gate action (EXECUTE / WAIT / SKIP), execution quality score (0-10),
            a liquidity judgement, the inputs used, and which inputs are estimated or defaulted.
            The 'source' field is 'jev' when the TypeSafe model answered, or 'heuristic_fallback'
            when the deterministic rule chain was used instead.
            Use this when the user asks whether a trade is a good idea to execute, or to populate the
            Jev rows of the analysis dashboard. quantity is optional (position assumed 1% if 0).
            """
            result = run_jev_gate(ticker, side, int(quantity or 0))
            return truncate(json.dumps(result, default=str))

        # ═══════════════════════════════════════════════════════════
        # ACTION TOOLS
        # ═══════════════════════════════════════════════════════════

        @tool
        def manage_watchlist(action: str, ticker: str = "") -> str:
            """
            Manage the user's stock watchlist.
            Actions: 'add' (add a stock), 'remove' (remove a stock), 'list' (show all watchlist stocks).
            For 'add' and 'remove', provide the ticker symbol (e.g. 'RELIANCE', 'TCS').
            Use this when the user asks to add/remove stocks from their watchlist or see their watchlist.
            """
            try:
                import requests
                base_url = os.getenv('API_BASE_URL', 'http://localhost:5000')

                if action == "list":
                    resp = requests.get(f"{base_url}/api/watchlist", timeout=10)
                    data = resp.json()
                    watchlist = data.get("watchlist", [])
                    if not watchlist:
                        return json.dumps({"message": "Your watchlist is empty. Ask me to add stocks!", "watchlist": []})
                    return json.dumps({"watchlist": watchlist, "count": len(watchlist)})

                elif action == "add":
                    if not ticker:
                        return json.dumps({"error": "Please specify a ticker to add."})
                    ticker = ticker.upper().strip()
                    resp = requests.post(
                        f"{base_url}/api/watchlist",
                        json={"ticker": ticker},
                        timeout=10
                    )
                    data = resp.json()
                    return json.dumps({
                        "message": f"✅ {ticker} added to your watchlist.",
                        "watchlist": data.get("watchlist", []),
                        "count": len(data.get("watchlist", []))
                    })

                elif action == "remove":
                    if not ticker:
                        return json.dumps({"error": "Please specify a ticker to remove."})
                    ticker = ticker.upper().strip()
                    resp = requests.delete(
                        f"{base_url}/api/watchlist",
                        json={"ticker": ticker},
                        timeout=10
                    )
                    data = resp.json()
                    return json.dumps({
                        "message": f"✅ {ticker} removed from your watchlist.",
                        "watchlist": data.get("watchlist", []),
                        "count": len(data.get("watchlist", []))
                    })

                else:
                    return json.dumps({"error": f"Unknown action '{action}'. Use 'add', 'remove', or 'list'."})

            except Exception as e:
                logger.error(f"manage_watchlist error: {e}")
                return json.dumps({"error": f"Watchlist operation failed: {str(e)}"})

        @tool
        def run_backtest(ticker: str, strategy: str, start_date: str = "", end_date: str = "", initial_capital: float = 100000) -> str:
            """
            Backtest a trading strategy on a specific stock over a date range.
            Parameters:
            - ticker: Stock symbol (e.g. 'TCS', 'RELIANCE')
            - strategy: Strategy name (e.g. 'momentum', 'swing', 'breakout', 'value', 'piotroski', 'trend_following', 'mean_reversion')
            - start_date: Start date in YYYY-MM-DD format (defaults to 3 years ago)
            - end_date: End date in YYYY-MM-DD format (defaults to today)
            - initial_capital: Starting capital in INR (default 100000)
            Returns key performance metrics: return, Sharpe, drawdown, win rate, trades.
            """
            try:
                import pandas as pd
                from datetime import timedelta

                if not end_date:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                if not start_date:
                    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

                ticker = ticker.upper().strip()

                # Fetch price data
                ticker_data = pipeline.get_ticker_history(ticker)
                if not ticker_data:
                    return json.dumps({"error": f"No historical data found for {ticker}. Verify the ticker symbol."})

                price_data = pd.DataFrame([dict(row) if not isinstance(row, dict) else row for row in ticker_data])
                if 'date' in price_data.columns:
                    price_data['date'] = pd.to_datetime(price_data['date'])
                    price_data = price_data[
                        (price_data['date'] >= start_date) & (price_data['date'] <= end_date)
                    ]

                if price_data.empty or len(price_data) < 50:
                    return json.dumps({"error": f"Insufficient data for {ticker} in date range {start_date} to {end_date}. Need at least 50 trading days."})

                # Import signal generation from application module
                if backtest_engine is None:
                    return json.dumps({"error": "Backtest engine is not available."})

                from BacktestEngine import BacktestEngine as BT
                from backtest_models import BacktestConfig, PositionSizingMode

                bt_config = BacktestConfig(
                    initial_capital=float(initial_capital),
                    sizing_mode=PositionSizingMode.RISK_BASED,
                    use_atr_stops=True,
                )
                engine = BT(data_pipeline=pipeline, config=bt_config)

                # Generate signals using a simple technical approach
                signals = _generate_signals_for_agent(strategy, price_data)

                result = engine.backtest_strategy(signals, price_data, strategy_name=strategy)

                # Return concise summary (not the full equity curve)
                summary = {
                    "status": "success",
                    "ticker": ticker,
                    "strategy": strategy,
                    "period": f"{start_date} to {end_date}",
                    "initial_capital": round(result.initial_capital, 2),
                    "final_capital": round(result.final_capital, 2),
                    "total_return_pct": round(result.total_return_pct, 2),
                    "cagr_pct": round(result.cagr_pct, 2),
                    "sharpe_ratio": round(result.sharpe_ratio, 3),
                    "sortino_ratio": round(result.sortino_ratio, 3),
                    "max_drawdown_pct": round(result.max_drawdown_pct, 2),
                    "total_trades": result.total_trades,
                    "win_rate_pct": round(result.win_rate_pct, 2),
                    "profit_factor": round(result.profit_factor, 2),
                    "avg_holding_days": round(result.average_holding_days, 1),
                    "total_costs": round(result.total_transaction_costs + result.total_slippage_costs, 2),
                }
                return json.dumps(summary, default=str)

            except Exception as e:
                logger.error(f"run_backtest error: {e}", exc_info=True)
                return json.dumps({"error": f"Backtest failed: {str(e)}"})

        @tool
        def create_strategy(name: str, description: str, rules: str) -> str:
            """
            Create a new custom trading strategy with technical indicator rules.
            Parameters:
            - name: Strategy name (e.g. 'Oversold Volume Spike')
            - description: Brief description of what the strategy does
            - rules: JSON string of rules, each with 'indicator', 'condition', and optional params.
              Example: '[{"indicator": "RSI", "condition": "below", "threshold": 30, "weight": 2.0}, {"indicator": "Volume", "condition": "above_avg", "multiplier": 2.0, "weight": 1.5}]'
            Use this when the user asks to create, define, or build a new trading strategy.
            """
            try:
                if strategy_engine is None:
                    return json.dumps({"error": "Strategy engine is not available."})

                # Parse rules
                try:
                    parsed_rules = json.loads(rules) if isinstance(rules, str) else rules
                except json.JSONDecodeError:
                    return json.dumps({"error": "Invalid rules format. Rules must be a valid JSON array."})

                if not isinstance(parsed_rules, list) or len(parsed_rules) == 0:
                    return json.dumps({"error": "Rules must be a non-empty JSON array."})

                strategy_engine.add_custom_strategy(
                    name=name,
                    description=description,
                    category="Custom",
                    risk_level="Custom",
                    timeframe="Custom",
                    rules=parsed_rules
                )

                return json.dumps({
                    "message": f"✅ Strategy '{name}' created successfully with {len(parsed_rules)} rules.",
                    "name": name,
                    "description": description,
                    "rules_count": len(parsed_rules),
                    "rules": parsed_rules,
                })

            except Exception as e:
                logger.error(f"create_strategy error: {e}")
                return json.dumps({"error": f"Strategy creation failed: {str(e)}"})

        @tool
        def list_strategies(category: str = "") -> str:
            """
            List all available trading strategies with their descriptions, risk levels, and timeframes.
            Optionally filter by category (e.g. 'Trend Following', 'Momentum', 'Mean Reversion', 'Volume', 'Volatility', 'Custom').
            """
            try:
                if strategy_engine is None:
                    # Fallback: list the basic screener strategies
                    return json.dumps({
                        "strategies": [
                            {"name": "momentum", "description": "High momentum stocks breaking out"},
                            {"name": "swing", "description": "Swing trading opportunities"},
                            {"name": "breakout", "description": "Stocks breaking resistance levels"},
                            {"name": "value", "description": "Undervalued stocks with strong fundamentals"},
                            {"name": "piotroski", "description": "High Piotroski F-Score stocks"},
                        ]
                    })

                catalog = strategy_engine.get_strategy_catalog()
                if category:
                    catalog = [s for s in catalog if s.get("category", "").lower() == category.lower()]

                return truncate(json.dumps({"strategies": catalog, "total": len(catalog)}, default=str))

            except Exception as e:
                logger.error(f"list_strategies error: {e}")
                return json.dumps({"error": f"Could not list strategies: {str(e)}"})

        @tool
        def compare_strategies(ticker: str, strategy_names: str) -> str:
            """
            Compare multiple strategies on the same stock to find the best performer.
            Parameters:
            - ticker: Stock symbol (e.g. 'TCS')
            - strategy_names: Comma-separated strategy names (e.g. 'momentum,swing,breakout')
            Returns comparison metrics for each strategy.
            """
            try:
                import pandas as pd
                from datetime import timedelta

                ticker = ticker.upper().strip()
                names = [s.strip().lower() for s in strategy_names.split(",")]

                if len(names) < 2:
                    return json.dumps({"error": "Please provide at least 2 strategy names separated by commas."})

                # Fetch price data (3 years)
                ticker_data = pipeline.get_ticker_history(ticker)
                if not ticker_data:
                    return json.dumps({"error": f"No data found for {ticker}."})

                price_data = pd.DataFrame([dict(row) if not isinstance(row, dict) else row for row in ticker_data])
                if 'date' in price_data.columns:
                    price_data['date'] = pd.to_datetime(price_data['date'])
                    cutoff = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
                    price_data = price_data[price_data['date'] >= cutoff]

                if price_data.empty or len(price_data) < 50:
                    return json.dumps({"error": f"Insufficient data for {ticker}."})

                from BacktestEngine import BacktestEngine as BT
                from backtest_models import BacktestConfig, PositionSizingMode

                bt_config = BacktestConfig(initial_capital=100000.0, use_atr_stops=True)
                engine = BT(data_pipeline=pipeline, config=bt_config)

                # Generate signals for each strategy
                signals_dict = {}
                for name in names:
                    try:
                        signals_dict[name] = _generate_signals_for_agent(name, price_data)
                    except Exception as e:
                        logger.warning(f"Could not generate signals for {name}: {e}")

                if len(signals_dict) < 2:
                    return json.dumps({"error": "Could not generate signals for enough strategies."})

                comparison = engine.compare_strategies(signals_dict, price_data)

                # Simplify for LLM consumption
                results = {}
                for name, res in comparison.get("results", {}).items():
                    if "error" not in res:
                        results[name] = {
                            "total_return_pct": res.get("total_return_pct", 0),
                            "sharpe_ratio": res.get("sharpe_ratio", 0),
                            "max_drawdown_pct": res.get("max_drawdown_pct", 0),
                            "win_rate_pct": res.get("win_rate_pct", 0),
                            "total_trades": res.get("total_trades", 0),
                            "profit_factor": res.get("profit_factor", 0),
                        }

                return json.dumps({
                    "ticker": ticker,
                    "comparison": results,
                    "rankings": comparison.get("rankings", {}),
                }, default=str)

            except Exception as e:
                logger.error(f"compare_strategies error: {e}")
                return json.dumps({"error": f"Strategy comparison failed: {str(e)}"})

        @tool
        def get_portfolio(user_id: str = "default") -> str:
            """
            Get the user's portfolio summary including holdings, P&L, and total value.
            Returns all current holdings with unrealized profit/loss.
            """
            try:
                if portfolio_manager is None:
                    return json.dumps({"error": "Portfolio manager is not available."})

                # Fetch current prices for portfolio holdings
                data = portfolio_manager._load_data(user_id)
                current_prices = {}
                for ticker in data.get("holdings", {}).keys():
                    try:
                        hist = pipeline.get_ticker_history(ticker)
                        if hist:
                            last_row = hist[-1] if isinstance(hist[-1], dict) else dict(hist[-1])
                            current_prices[ticker] = last_row.get("close", 0)
                    except Exception:
                        pass

                summary = portfolio_manager.get_portfolio_summary(user_id, current_prices)
                return truncate(json.dumps(summary, default=str))

            except Exception as e:
                logger.error(f"get_portfolio error: {e}")
                return json.dumps({"error": f"Could not fetch portfolio: {str(e)}"})

        @tool
        def add_transaction(ticker: str, transaction_type: str, quantity: int, price: float, user_id: str = "default") -> str:
            """
            Record a buy or sell transaction in the user's portfolio.
            Parameters:
            - ticker: Stock symbol (e.g. 'RELIANCE')
            - transaction_type: 'BUY' or 'SELL'
            - quantity: Number of shares
            - price: Price per share in INR
            Use when the user wants to manually record a trade in their portfolio tracker.
            """
            try:
                if portfolio_manager is None:
                    return json.dumps({"error": "Portfolio manager is not available."})

                ticker = ticker.upper().strip()
                transaction_type = transaction_type.upper().strip()

                if transaction_type not in ("BUY", "SELL"):
                    return json.dumps({"error": "Transaction type must be 'BUY' or 'SELL'."})

                result = portfolio_manager.add_transaction(
                    user_id=user_id,
                    ticker=ticker,
                    type=transaction_type,
                    quantity=quantity,
                    price=price
                )

                return json.dumps({
                    "message": f"✅ {transaction_type} {quantity} shares of {ticker} at ₹{price:.2f} recorded.",
                    "transaction": result,
                    "total_value": round(quantity * price, 2),
                }, default=str)

            except ValueError as e:
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"add_transaction error: {e}")
                return json.dumps({"error": f"Transaction failed: {str(e)}"})

        # ═══════════════════════════════════════════════════════════
        # PAPER TRADING TOOLS
        # ═══════════════════════════════════════════════════════════

        @tool
        def paper_trade(ticker: str, side: str, quantity: int, order_type: str = "MARKET",
                        price: float = 0, confirm_despite_wait: bool = False) -> str:
            """
            Place a paper (simulated) trade order. No real money is used.
            Parameters:
            - ticker: Stock symbol (e.g. 'INFY', 'RELIANCE')
            - side: 'BUY' or 'SELL'
            - quantity: Number of shares to trade (must be provided by the user)
            - order_type: 'MARKET' (default) or 'LIMIT'
            - price: Limit price (required for LIMIT orders, ignored for MARKET)
            - confirm_despite_wait: Set True ONLY if the user has explicitly confirmed they want to
              proceed after being told the Jev gate returned WAIT. Never set it on your own initiative.
            BUY orders are screened by the Jev pre-trade gate first: a SKIP always blocks the order;
            a WAIT blocks it unless confirm_despite_wait is True. SELL (exit) orders are not blocked.
            Always confirm this is paper (simulated) trading in your response.
            """
            try:
                if order_manager is None:
                    return json.dumps({"error": "Paper trading is not available. Order manager not initialized."})

                from BrokerAdapter import OrderRequest, OrderSide, BrokerOrderType, ProductType

                ticker = ticker.upper().strip()
                side = side.upper().strip()

                if side not in ("BUY", "SELL"):
                    return json.dumps({"error": "Side must be 'BUY' or 'SELL'."})
                if quantity <= 0:
                    return json.dumps({"error": "Quantity must be greater than 0."})
                is_limit = order_type.upper() == "LIMIT"
                if is_limit and (price is None or price <= 0):
                    return json.dumps({"error": "A positive limit price is required for LIMIT orders."})

                # ── Jev pre-trade gate (BUY orders only) ──────────────────────
                jev_info: Any = "Not applied to SELL/exit orders."
                if side == "BUY":
                    gate = run_jev_gate(ticker, "BUY", quantity, price if is_limit else None)
                    if not gate.get("evaluated"):
                        if not jev_fail_open:
                            return json.dumps({
                                "status": "BLOCKED_BY_JEV_GATE",
                                "executed": False,
                                "message": ("The Jev pre-trade gate could not be evaluated, so the BUY was not placed "
                                            f"(fail-closed). Reason: {gate.get('error')}"),
                                "jev_gate": gate,
                            }, default=str)
                        jev_info = {"evaluated": False, "error": gate.get("error"),
                                    "note": "Gate unavailable; proceeding because JEV_GATE_FAIL_OPEN is enabled."}
                    else:
                        decision = gate["decision"]
                        action = decision["action"]
                        if action == "SKIP":
                            return json.dumps({
                                "status": "BLOCKED_BY_JEV_GATE",
                                "executed": False,
                                "message": "Jev gate returned SKIP — order NOT placed.",
                                "jev_gate": gate,
                            }, default=str)
                        if not gate["should_execute"] and not confirm_despite_wait:
                            return json.dumps({
                                "status": "BLOCKED_BY_JEV_GATE",
                                "executed": False,
                                "requires_user_confirmation": True,
                                "message": (f"Jev gate returned {action} (not cleared to execute now). Order NOT placed. "
                                            "Ask the user whether they want to proceed anyway or switch to a LIMIT order; "
                                            "only re-call with confirm_despite_wait=true if they explicitly confirm."),
                                "jev_gate": gate,
                            }, default=str)
                        jev_info = {
                            "action": action,
                            "source": decision.get("source"),
                            "execution_quality_score": decision.get("execution_quality_score"),
                            "overridden_by_user": bool(not gate["should_execute"] and confirm_despite_wait),
                            "reasons": decision.get("reasons"),
                            "estimated_spread_bps": gate["inputs"]["estimated_spread_bps"],
                            "psi": gate["inputs"]["psi"],
                        }

                order = OrderRequest(
                    symbol=ticker,
                    exchange="NSE",
                    side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                    order_type=BrokerOrderType.LIMIT if is_limit else BrokerOrderType.MARKET,
                    quantity=quantity,
                    price=price if is_limit else None,
                    product=ProductType.CNC,
                    tag="agent_trade",
                )

                response = order_manager.validate_and_place(order, user_confirmed=True)

                result = {
                    "order_id": response.order_id,
                    "status": response.status.value,
                    "message": response.message,
                    "symbol": ticker,
                    "side": side,
                    "quantity": response.filled_quantity or quantity,
                    "price": response.average_price,
                    "total_value": round((response.average_price or 0) * quantity, 2),
                    "mode": "PAPER TRADING (simulated)",
                    "jev_gate": jev_info,
                }

                return json.dumps(result, default=str)

            except Exception as e:
                logger.error(f"paper_trade error: {e}")
                return json.dumps({"error": f"Paper trade failed: {str(e)}"})

        @tool
        def get_paper_positions() -> str:
            """
            Get current paper trading positions with unrealized P&L.
            Shows all open positions from simulated trades.
            """
            try:
                if order_manager is None:
                    return json.dumps({"error": "Paper trading is not available."})

                positions = order_manager.broker.get_positions()
                funds = order_manager.broker.get_funds()

                pos_list = []
                for p in positions:
                    pos_list.append({
                        "symbol": p.symbol,
                        "quantity": p.quantity,
                        "avg_price": round(p.average_price, 2),
                        "last_price": round(p.last_price, 2),
                        "pnl": round(p.pnl, 2),
                        "pnl_pct": round((p.pnl / (p.average_price * abs(p.quantity))) * 100, 2) if p.average_price > 0 and p.quantity != 0 else 0,
                    })

                return json.dumps({
                    "positions": pos_list,
                    "total_positions": len(pos_list),
                    "funds": funds,
                    "mode": "PAPER TRADING",
                }, default=str)

            except Exception as e:
                logger.error(f"get_paper_positions error: {e}")
                return json.dumps({"error": f"Could not fetch positions: {str(e)}"})

        @tool
        def get_paper_order_history() -> str:
            """
            Get the history of all paper (simulated) trade orders placed today.
            Shows order details, fill prices, and status.
            """
            try:
                if order_manager is None:
                    return json.dumps({"error": "Paper trading is not available."})

                orders = order_manager.broker.get_order_history()

                order_list = []
                for o in orders[-20:]:  # Last 20 orders
                    order_list.append({
                        "order_id": o.order_id,
                        "symbol": o.symbol,
                        "side": o.side,
                        "quantity": o.quantity,
                        "filled_qty": o.filled_quantity,
                        "price": round(o.average_price, 2),
                        "status": o.status.value,
                        "time": o.placed_at.isoformat() if o.placed_at else None,
                    })

                stats = order_manager.get_daily_stats()

                return json.dumps({
                    "orders": order_list,
                    "total_orders": len(order_list),
                    "daily_stats": stats,
                    "mode": "PAPER TRADING",
                }, default=str)

            except Exception as e:
                logger.error(f"get_paper_order_history error: {e}")
                return json.dumps({"error": f"Could not fetch order history: {str(e)}"})

        # ═══════════════════════════════════════════════════════════
        # Assemble tool list (explicit groups — no index slicing)
        # ═══════════════════════════════════════════════════════════

        analysis_tools = [
            screen_stocks, predict_stock, get_market_overview, get_stock_history,
            get_sentiment, generate_pattern_chart, search_web, search_financial_docs,
            jev_pretrade_check,
        ]
        action_tools = [
            manage_watchlist, run_backtest, create_strategy, list_strategies,
            compare_strategies, get_portfolio, add_transaction,
        ]
        paper_tools = [paper_trade, get_paper_positions, get_paper_order_history]

        all_tools = analysis_tools + action_tools
        if order_manager is not None:
            all_tools += paper_tools
            logger.info(f"Agent initialized with {len(all_tools)} tools (paper trading enabled)")
        else:
            logger.info(f"Agent initialized with {len(all_tools)} tools (paper trading disabled)")

        return all_tools

    def _build_graph(self):
        max_history = self.max_history

        def call_model(state: MessagesState):
            messages = state['messages']

            # Separate system message from conversation messages
            system_msg = SystemMessage(content=_build_system_prompt())

            # Filter out any existing system messages
            conv_messages = [m for m in messages if not isinstance(m, SystemMessage)]

            # Trim conversation history — keep first user message (context) + last N messages
            if len(conv_messages) > max_history:
                first_user_msg = None
                for m in conv_messages:
                    if isinstance(m, HumanMessage):
                        first_user_msg = m
                        break
                
                idx = len(conv_messages) - max_history
                if idx < 0:
                    idx = 0
                
                # Back up idx if it points to a ToolMessage so we don't sever it from its AIMessage
                while idx > 0 and getattr(conv_messages[idx], 'type', '') == 'tool':
                    idx -= 1
                    
                tail = conv_messages[idx:]
                if first_user_msg and first_user_msg not in tail:
                    conv_messages = [first_user_msg] + tail
                else:
                    conv_messages = tail

            # Prepend system prompt
            full_messages = [system_msg] + conv_messages

            # Invoke with retry + exponential backoff for transient API errors
            last_error = None
            for attempt in range(3):
                try:
                    response = self.llm_with_tools.invoke(full_messages)
                    return {"messages": [response]}
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    if any(kw in error_str for kw in ['429', '500', '503', 'timeout', 'quota', 'resource_exhausted']):
                        wait = (2 ** attempt)
                        logger.warning(f"LLM invoke attempt {attempt+1}/3 failed ({e}), retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            raise last_error

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

        return workflow

    def _get_postgres_connection_string(self) -> str:
        url = self.db_url
        if url.startswith("postgresql+psycopg2://"):
            url = url.replace("postgresql+psycopg2://", "postgresql://")
        return url

    def _get_or_create_vectorstore(self) -> Optional[PGVector]:
        """Lazy-initialize and cache the PGVector vectorstore instance."""
        if self._vectorstore is None:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=config.GOOGLE_API_KEY,
                )
                self._vectorstore = PGVector(
                    embeddings=embeddings,
                    collection_name="financial_docs",
                    connection=self._get_postgres_connection_string(),
                    use_jsonb=True,
                )
                logger.info("PGVector vectorstore initialized.")
            except Exception as e:
                logger.warning(f"Could not initialize vectorstore: {e}")
                return None
        return self._vectorstore

    # Maximum time (seconds) allowed for a single agent stream before timeout
    STREAM_TIMEOUT_SECONDS = 120

    def stream(self, user_message: str, session_id: str) -> Generator[str, None, None]:
        """
        Streams the agent's response chunks for a given session.
        Uses PostgresSaver for conversation persistence.
        Enforces a timeout to prevent hung tools from blocking forever.
        """
        # Sanitize input
        user_message = self._sanitize_input(user_message)
        if not user_message:
            yield f"data: {json.dumps({'content': 'Please provide a valid message.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Timeout flag
        timed_out = threading.Event()
        timer = threading.Timer(self.STREAM_TIMEOUT_SECONDS, timed_out.set)
        timer.daemon = True
        timer.start()

        try:
            # Use connection pool if available, otherwise fallback to direct connection
            if self._pg_pool:
                with self._pg_pool.connection() as conn:
                    yield from self._stream_with_connection(conn, user_message, session_id, timed_out)
            else:
                conn_str = self._get_postgres_connection_string()
                with Connection.connect(conn_str, autocommit=True) as conn:
                    yield from self._stream_with_connection(conn, user_message, session_id, timed_out)

        except Exception as e:
            logger.error(f"Agent stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': 'I encountered an internal error. Please try again.'})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            timer.cancel()

    def _stream_with_connection(self, conn, user_message, session_id, timed_out):
        """Internal streaming logic using a provided database connection."""
        checkpointer = PostgresSaver(conn)
        try:
            checkpointer.setup()
        except Exception as e:
            logger.debug(f"Checkpointer setup skipped in stream: {e}")
            
        app = self.graph.compile(checkpointer=checkpointer)
        thread_config = {"configurable": {"thread_id": session_id}}

        for msg, metadata in app.stream(
            {"messages": [HumanMessage(content=user_message)]},
            thread_config,
            stream_mode="messages"
        ):
            if timed_out.is_set():
                logger.warning(f"Agent stream timed out after {self.STREAM_TIMEOUT_SECONDS}s for session {session_id}")
                timeout_msg = json.dumps({'content': '\n\n⏱️ Response timed out. Please try a simpler question or ask about a specific stock.'})
                yield f"data: {timeout_msg}\n\n"
                break

            if msg.type in ("AIMessageChunk", "ai"):
                if isinstance(msg.content, str) and msg.content:
                    yield f"data: {json.dumps({'content': msg.content})}\n\n"
                elif isinstance(msg.content, list):
                    for part in msg.content:
                        if isinstance(part, dict) and "text" in part:
                            yield f"data: {json.dumps({'content': part['text']})}\n\n"

                if hasattr(msg, 'tool_call_chunks') and msg.tool_call_chunks:
                    for tc in msg.tool_call_chunks:
                        if tc.get("name"):
                            tool_name = tc.get("name")
                            friendly_names = {
                                "screen_stocks": "📊 Screening stocks...",
                                "predict_stock": "🤖 Running prediction model...",
                                "get_market_overview": "📈 Fetching market data...",
                                "get_stock_history": "📉 Loading price history...",
                                "get_sentiment": "📰 Analyzing news sentiment...",
                                "generate_pattern_chart": "🎨 Generating chart...",
                                "search_web": "🌐 Searching the web...",
                                "search_financial_docs": "📄 Searching documents...",
                                "jev_pretrade_check": "🛡️ Running Jev pre-trade gate...",
                                "manage_watchlist": "⭐ Updating watchlist...",
                                "run_backtest": "📊 Running backtest...",
                                "create_strategy": "🔧 Creating strategy...",
                                "list_strategies": "📋 Fetching strategies...",
                                "compare_strategies": "⚖️ Comparing strategies...",
                                "get_portfolio": "💼 Loading portfolio...",
                                "add_transaction": "💰 Recording transaction...",
                                "paper_trade": "💹 Checking Jev gate & placing paper trade...",
                                "get_paper_positions": "📊 Loading positions...",
                                "get_paper_order_history": "📋 Loading order history...",
                            }
                            status = friendly_names.get(tool_name, f"⚙️ Running {tool_name}...")
                            yield f"data: {json.dumps({'tool': status})}\n\n"
            elif getattr(msg, 'type', '') == "tool":
                if isinstance(msg.content, str) and "chart:pattern" in msg.content:
                    try:
                        content_dict = json.loads(msg.content)
                        chart_md = content_dict.get("chart", msg.content)
                    except json.JSONDecodeError:
                        chart_md = msg.content
                    yield f"data: {json.dumps({'content': chr(10) + chart_md + chr(10)})}\n\n"

        yield "data: [DONE]\n\n"

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session by deleting checkpoint data."""
        try:
            if self._pg_pool:
                with self._pg_pool.connection() as conn:
                    with conn.cursor() as cur:
                        for table in ['checkpoint_writes', 'checkpoint_blobs', 'checkpoints']:
                            try:
                                cur.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                            except Exception:
                                pass
            else:
                conn_str = self._get_postgres_connection_string()
                with Connection.connect(conn_str, autocommit=True) as conn:
                    with conn.cursor() as cur:
                        for table in ['checkpoint_writes', 'checkpoint_blobs', 'checkpoints']:
                            try:
                                cur.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                            except Exception:
                                pass
            logger.info(f"Session {session_id} cleared from checkpoint storage.")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear session {session_id}: {e}")
            return True


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION HELPER (for agent backtest tool)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_signals_for_agent(strategy: str, price_data) -> 'pd.DataFrame':
    """
    Generate trading signals for the backtest tool.
    This is a simplified version of the signal generation used in the API endpoints.
    """
    import pandas as pd
    import numpy as np

    df = price_data.copy()
    df = df.sort_values('date').reset_index(drop=True)

    close = df['close']
    signals = pd.DataFrame({'date': df['date'], 'signal': 0})

    strategy = strategy.lower().strip()

    if strategy in ('momentum', 'momentum_hunter'):
        # RSI + MACD momentum
        rsi = _compute_rsi_helper(close)
        macd, signal_line, _ = _compute_macd_helper(close)
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        buy = (rsi > 50) & (rsi < 75) & (macd > signal_line) & (close > sma_20)
        sell = (rsi > 75) | (macd < signal_line) | (close < sma_50)
        signals.loc[buy, 'signal'] = 1
        signals.loc[sell, 'signal'] = -1

    elif strategy in ('swing', 'swing_trader'):
        rsi = _compute_rsi_helper(close)
        macd, signal_line, hist = _compute_macd_helper(close)

        buy = (rsi > 40) & (rsi < 65) & (macd > signal_line) & (hist > hist.shift(1))
        sell = (rsi > 70) | (hist < 0)
        signals.loc[buy, 'signal'] = 1
        signals.loc[sell, 'signal'] = -1

    elif strategy in ('breakout', 'breakout_scanner'):
        high_20 = df['high'].rolling(20).max()
        vol_avg = df['volume'].rolling(20).mean()

        buy = (close > high_20.shift(1)) & (df['volume'] > vol_avg * 1.5)
        sma_20 = close.rolling(20).mean()
        sell = close < sma_20
        signals.loc[buy, 'signal'] = 1
        signals.loc[sell, 'signal'] = -1

    elif strategy in ('value', 'value_investing'):
        rsi = _compute_rsi_helper(close)
        sma_200 = close.rolling(200).mean()

        buy = (rsi < 35) & (close < sma_200 * 0.95)
        sell = (rsi > 65) | (close > sma_200 * 1.10)
        signals.loc[buy, 'signal'] = 1
        signals.loc[sell, 'signal'] = -1

    elif strategy in ('mean_reversion',):
        rsi = _compute_rsi_helper(close)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_lower = bb_mid - 2 * bb_std
        bb_upper = bb_mid + 2 * bb_std

        buy = (close < bb_lower) & (rsi < 35)
        sell = (close > bb_upper) | (rsi > 70)
        signals.loc[buy, 'signal'] = 1
        signals.loc[sell, 'signal'] = -1

    elif strategy in ('trend_following',):
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        rsi = _compute_rsi_helper(close)

        buy = (sma_50 > sma_200) & (close > sma_50) & (rsi > 50)
        sell = (sma_50 < sma_200) | (close < sma_50)
        signals.loc[buy, 'signal'] = 1
        signals.loc[sell, 'signal'] = -1

    elif strategy in ('piotroski',):
        # Simplified — use moving average crossover as proxy
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        rsi = _compute_rsi_helper(close)

        buy = (sma_20 > sma_50) & (rsi > 45) & (rsi < 70)
        sell = (sma_20 < sma_50) | (rsi > 75)
        signals.loc[buy, 'signal'] = 1
        signals.loc[sell, 'signal'] = -1

    else:
        # Default: simple moving average crossover
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        buy = (sma_20 > sma_50) & (sma_20.shift(1) <= sma_50.shift(1))
        sell = (sma_20 < sma_50) & (sma_20.shift(1) >= sma_50.shift(1))
        signals.loc[buy, 'signal'] = 1
        signals.loc[sell, 'signal'] = -1

    return signals


def _compute_rsi_helper(series, period=14):
    """Compute RSI."""
    import numpy as np
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def _compute_macd_helper(series, fast=12, slow=26, signal=9):
    """Compute MACD line, signal line, histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram