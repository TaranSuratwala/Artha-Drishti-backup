"""
News Provider Service
======================
Multi-source news aggregation for stock market intelligence.
Bridges to SentimentEngine for analysis.

Data sources (priority order):
  1. Finnhub API (via SentimentEngine)
  2. yfinance .news attribute (free fallback)

Important:
- This module must not crash if optional ML/NLP dependencies (pandas, scipy,
  vaderSentiment, textblob) are missing.
- It should gracefully fall back so the API can still return latest news.
"""

import os
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime

import requests


logger = logging.getLogger(__name__)

# ---- Critical defaults ----
# SentimentEngine uses FINNHUB_KEY (Finnhub). We keep a safe default so
# imports never crash; however, Finnhub calls may still fail (401/403).
_DEFAULT_FINNHUB_KEY = "O122CS7T5G9DLCIM"

# AlphaVantage key for NEWS_SENTIMENT / NEWS
# Your project uses an AlphaVantage key stored in code (as per your instruction).
_DEFAULT_ALPHAVANTAGE_KEY = "d7ovmj9r01qr68pb6oegd7ovmj9r01qr68pb6of0"


# ---- yfinance fallback (free, no API key) ----
try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    yf = None
    _HAS_YFINANCE = False


class YFinanceNewsProvider:
    """Free fallback news provider using yfinance Ticker.news."""

    _cache: Dict[str, tuple] = {}
    _cache_ttl = 600  # 10 min
    _last_req = 0.0

    @classmethod
    def _rate_limit(cls):
        elapsed = time.time() - cls._last_req
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        cls._last_req = time.time()

    @classmethod
    def get_company_news(cls, symbol: str, limit: int = 15) -> List[Dict]:
        cache_key = f"yf_{symbol}"
        if cache_key in cls._cache:
            ts, data = cls._cache[cache_key]
            if time.time() - ts < cls._cache_ttl:
                return data[:limit]

        cls._rate_limit()
        try:
            if yf is None:
                return []
            ticker = yf.Ticker(f"{symbol}.NS")
            raw = ticker.news or []

            articles: List[Dict] = []
            for item in raw:
                content = item.get("content", item)
                articles.append(
                    {
                        "headline": content.get("title", item.get("title", "")),
                        "summary": content.get(
                            "summary", content.get("description", "")
                        ),
                        "datetime": content.get(
                            "pubDate",
                            datetime.fromtimestamp(
                                item.get("providerPublishTime", 0)
                            ).isoformat()
                            if item.get("providerPublishTime")
                            else None,
                        ),
                        "source": (
                            content.get("provider", {}).get(
                                "displayName", "Yahoo Finance"
                            )
                            if isinstance(content.get("provider"), dict)
                            else str(content.get("provider", "Yahoo Finance"))
                        ),
                        "url": (
                            content.get("canonicalUrl", {}).get(
                                "url", item.get("link", "")
                            )
                            if isinstance(content.get("canonicalUrl"), dict)
                            else str(
                                content.get("canonicalUrl", item.get("link", ""))
                            )
                        ),
                        "category": "company_news",
                        "related": symbol,
                    }
                )

            cls._cache[cache_key] = (time.time(), articles)
            return articles[:limit]
        except Exception as exc:
            logger.warning(f"yfinance news fetch failed for {symbol}: {exc}")
            return []

    @classmethod
    def get_market_news(cls, limit: int = 20) -> List[Dict]:
        """Fetch market-level news via broad index tickers."""

        seen_titles: set[str] = set()
        articles: List[Dict] = []

        for idx_sym in ["^NSEI", "^BSESN"]:
            cache_key = f"yf_mkt_{idx_sym}"
            if cache_key in cls._cache:
                ts, data = cls._cache[cache_key]
                if time.time() - ts < cls._cache_ttl:
                    for a in data:
                        if a.get("headline") not in seen_titles:
                            seen_titles.add(a.get("headline"))
                            articles.append(a)
                    continue

            cls._rate_limit()
            try:
                raw = yf.Ticker(idx_sym).news or []
                batch: List[Dict] = []

                for item in raw:
                    content = item.get("content", item)
                    art = {
                        "headline": content.get("title", item.get("title", "")),
                        "summary": content.get(
                            "summary", content.get("description", "")
                        ),
                        "datetime": content.get(
                            "pubDate",
                            datetime.fromtimestamp(
                                item.get("providerPublishTime", 0)
                            ).isoformat()
                            if item.get("providerPublishTime")
                            else None,
                        ),
                        "source": (
                            content.get("provider", {}).get(
                                "displayName", "Yahoo Finance"
                            )
                            if isinstance(content.get("provider"), dict)
                            else str(content.get("provider", "Yahoo Finance"))
                        ),
                        "url": (
                            content.get("canonicalUrl", {}).get(
                                "url", item.get("link", "")
                            )
                            if isinstance(content.get("canonicalUrl"), dict)
                            else str(
                                content.get("canonicalUrl", item.get("link", ""))
                            )
                        ),
                        "category": "market",
                    }

                    batch.append(art)
                    if art["headline"] not in seen_titles:
                        seen_titles.add(art["headline"])
                        articles.append(art)

                cls._cache[cache_key] = (time.time(), batch)
            except Exception as exc:
                logger.warning(f"yfinance market news failed for {idx_sym}: {exc}")

        return articles[:limit]


def _safe_create_sentiment_engine(finnhub_key: str):
    """Create SentimentEngine only if it can be imported without missing deps."""
    try:
        from SentimentEngine import SentimentEngine

        return SentimentEngine(finnhub_key=finnhub_key)
    except Exception as exc:
        logger.warning(
            "SentimentEngine unavailable; returning news with neutral sentiment. "
            f"Reason: {type(exc).__name__}: {exc}"
        )
        return None


class NewsAggregator:
    """Unified news aggregation interface."""

    def __init__(self, finnhub_key: Optional[str] = None):
        self.finnhub_key = finnhub_key or os.getenv("FINNHUB_KEY", "") or _DEFAULT_FINNHUB_KEY
        self._sentiment_engine = _safe_create_sentiment_engine(self.finnhub_key)
        self._yf_fallback = _HAS_YFINANCE

    def get_providers(self) -> List[Dict]:
        return [
            {
                "name": "SentimentEngine/Finnhub",
                "active": bool(self._sentiment_engine),
                "type": "company_news (sentiment)",
            },
            {
                "name": "Yahoo Finance (fallback)",
                "active": self._yf_fallback,
                "type": "company_news (sentiment placeholder)",
            },
        ]

    def get_market_news(self, limit: int = 20) -> Dict:
        if not self._yf_fallback:
            return {
                "articles": [],
                "count": 0,
                "source": "none",
                "timestamp": datetime.now().isoformat(),
            }

        articles = YFinanceNewsProvider.get_market_news(limit=limit)
        return {
            "articles": articles,
            "count": len(articles),
            "source": "yfinance",
            "timestamp": datetime.now().isoformat(),
        }

    def get_stock_news(self, symbol: str, limit: int = 20, days_back: int = 14) -> Dict:
        # 0) AlphaVantage (NEWS_SENTIMENT) — primary for “latest news”

        # since Finnhub may be blocked (401/403) in some environments.
        av_articles = []
        try:
            av_articles = self._get_alphavantage_stock_news(symbol=symbol, limit=limit)
        except Exception as exc:
            logger.warning(f"AlphaVantage news failed for {symbol}: {exc}")
            av_articles = []

        if av_articles:
            # Aggregate from AlphaVantage sentiment_score fields
            scores = [a.get("sentiment", 0.0) for a in av_articles if isinstance(a.get("sentiment"), (int, float))]
            if scores:
                avg_score = sum(scores) / len(scores)
                overall = "BULLISH" if avg_score > 0.15 else "BEARISH" if avg_score < -0.15 else "NEUTRAL"
                bullish_ratio = sum(1 for s in scores if s > 0.05) / len(scores)
                bearish_ratio = sum(1 for s in scores if s < -0.05) / len(scores)
            else:
                avg_score = 0.0
                overall = "NEUTRAL"
                bullish_ratio = 0.0
                bearish_ratio = 0.0

            return {
                "ticker": symbol,
                "articles": av_articles,
                "article_count": len(av_articles),
                "aggregate": {
                    "sentiment_score": round(avg_score, 4),
                    "bullish_ratio": round(bullish_ratio, 4),
                    "bearish_ratio": round(bearish_ratio, 4),
                },
                "overall_sentiment": overall,
            }

        # 1) If SentimentEngine is available, use its structured article-level output.
        if self._sentiment_engine is not None:
            try:
                detailed = self._sentiment_engine.get_detailed_analysis(symbol, days_back=days_back)
                if detailed.get("articles"):
                    detailed["articles"] = detailed["articles"][:limit]
                    return detailed
            except Exception as exc:
                logger.warning(f"SentimentEngine failed for {symbol}: {exc}")

        # 2) Fallback: yfinance latest news + neutral placeholders for sentiment fields.
        if not self._yf_fallback:
            return {
                "ticker": symbol,
                "articles": [],
                "article_count": 0,
                "aggregate": {"sentiment_score": 0.0, "bullish_ratio": 0.0, "bearish_ratio": 0.0},
                "overall_sentiment": "NEUTRAL",
            }

        articles_raw = YFinanceNewsProvider.get_company_news(symbol, limit=limit)

        analyzed: List[Dict] = []
        for art in articles_raw:
            analyzed.append(
                {
                    "headline": art.get("headline", ""),
                    "source": art.get("source", ""),
                    "datetime": art.get("datetime", ""),
                    "url": art.get("url", ""),
                    "sentiment": 0.0,
                    "magnitude": 0.0,
                    "event_category": "general",
                    "positive_keywords": 0,
                    "negative_keywords": 0,
                }
            )

        return {
            "ticker": symbol,
            "articles": analyzed,
            "article_count": len(analyzed),
            "aggregate": {"sentiment_score": 0.0, "bullish_ratio": 0.0, "bearish_ratio": 0.0},
            "overall_sentiment": "NEUTRAL",
        }

    def _get_alphavantage_stock_news(self, symbol: str, limit: int = 20) -> List[Dict]:
        """Fetch AlphaVantage NEWS_SENTIMENT and normalize to this module’s schema."""
        # AlphaVantage expects US tickers for NEWS_SENTIMENT; if your UI uses NSE tickers,
        # you’ll still get graceful fallback to SentimentEngine/yfinance.
        if not _DEFAULT_ALPHAVANTAGE_KEY:
            return []

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": _DEFAULT_ALPHAVANTAGE_KEY,
        }

        resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=20)
        if resp.status_code != 200:
            return []

        data = resp.json()
        if not isinstance(data, dict):
            return []

        if "feed" not in data:
            # often contains "Note" or "Error Message"
            return []

        feed = data.get("feed") or []
        # feed items are newest first sometimes; keep top-N
        out: List[Dict] = []
        for item in feed[:limit]:
            # AlphaVantage fields: title, url, time_published, authors, summary, sentiment_score
            try:
                score = float(item.get("sentiment_score", 0.0))
            except (TypeError, ValueError):
                score = 0.0

            out.append(
                {
                    "headline": item.get("title", ""),
                    "source": item.get("source", ""),
                    "datetime": item.get("time_published", ""),
                    "url": item.get("url", ""),
                    "sentiment": score,
                    "magnitude": abs(score),
                    "event_category": "general",
                    "positive_keywords": 0,
                    "negative_keywords": 0,
                }
            )

        return out


    def get_portfolio_news(self, symbols: List[str], limit_per_stock: int = 5) -> Dict:
        return {
            "portfolio": {
                s: self.get_stock_news(s, limit=limit_per_stock, days_back=7) for s in symbols
            },
            "stocks_count": len(symbols),
            "timestamp": datetime.now().isoformat(),
        }


# ---- Singleton factory ----
_aggregator: Optional[NewsAggregator] = None


def get_news_aggregator(finnhub_key: Optional[str] = None) -> NewsAggregator:
    """Get or create a singleton NewsAggregator instance."""
    global _aggregator
    if _aggregator is None:
        _aggregator = NewsAggregator(finnhub_key=finnhub_key)
    return _aggregator

