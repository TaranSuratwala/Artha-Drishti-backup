"""
NSE Index Universe Provider (Feature #3)
========================================
Serves constituent lists for NSE indices (NIFTY50, NIFTY100, NIFTY200,
NIFTY500, NIFTYBANK and sectoral indices) for universe-level
backtesting and screening.

- Downloads official constituent CSVs from NSE index archives
- Weekly TTL cache (constituents change rarely) + disk persistence
- Static fallback lists for NIFTY50 / NIFTYBANK so core universes work
  even when NSE is unreachable on first run
"""

import csv
import io
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_BASE_URL = os.getenv(
    "NSE_INDEX_ARCHIVE_BASE",
    "https://nsearchives.nseindia.com/content/indices",
)

# Index name -> official NSE constituents CSV filename
INDEX_SOURCES: Dict[str, str] = {
    "NIFTY50": "ind_nifty50list.csv",
    "NIFTY100": "ind_nifty100list.csv",
    "NIFTY200": "ind_nifty200list.csv",
    "NIFTY500": "ind_nifty500list.csv",
    "NIFTYBANK": "ind_niftybanklist.csv",
    "NIFTYIT": "ind_niftyitlist.csv",
    "NIFTYFMCG": "ind_niftyfmcglist.csv",
    "NIFTYPHARMA": "ind_niftypharmalist.csv",
    "NIFTYAUTO": "ind_niftyautolist.csv",
    "NIFTYMIDCAP150": "ind_niftymidcap150list.csv",
    "NIFTYSMALLCAP250": "ind_niftysmallcap250list.csv",
}

# Static fallbacks (used only when network + disk cache are unavailable).
# These are refreshed automatically once a successful download occurs.
_STATIC_FALLBACKS: Dict[str, List[str]] = {
    "NIFTY50": [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
        "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
        "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
        "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
        "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
        "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT",
        "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
        "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
        "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS",
        "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
    ],
    "NIFTYBANK": [
        "AUBANK", "AXISBANK", "BANDHANBNK", "BANKBARODA", "CANBK",
        "FEDERALBNK", "HDFCBANK", "ICICIBANK", "IDFCFIRSTB", "INDUSINDBK",
        "KOTAKBANK", "PNB", "SBIN",
    ],
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
}

_TTL_SECONDS = float(os.getenv("INDEX_UNIVERSE_TTL_HOURS", "168")) * 3600  # 7 days


def normalize_index_name(name: str) -> str:
    """Normalize user input like 'nifty 50' / 'NIFTY-BANK' to 'NIFTY50'/'NIFTYBANK'."""
    cleaned = "".join(ch for ch in str(name or "").upper() if ch.isalnum())
    return cleaned


class IndexUniverseProvider:
    """Fetch and cache NSE index constituent lists."""

    def __init__(self, cache_dir: Optional[str] = None):
        base = Path(cache_dir) if cache_dir else Path(__file__).resolve().parent / "data"
        base.mkdir(exist_ok=True)
        self.cache_dir = base
        self._lock = Lock()
        self._cache: Dict[str, Dict] = {}  # name -> {symbols, loaded_at, source}

    def list_supported(self) -> List[Dict[str, object]]:
        out = []
        for name in sorted(INDEX_SOURCES):
            entry = {"name": name}
            cached = self._cache.get(name)
            if cached:
                entry["count"] = len(cached["symbols"])
                entry["source"] = cached["source"]
            out.append(entry)
        return out

    def get_constituents(self, index_name: str, force_refresh: bool = False) -> List[str]:
        name = normalize_index_name(index_name)
        if name not in INDEX_SOURCES:
            raise ValueError(
                f"Unsupported index '{index_name}'. Supported: {', '.join(sorted(INDEX_SOURCES))}"
            )
        with self._lock:
            cached = self._cache.get(name)
            if cached and not force_refresh and (time.time() - cached["loaded_at"]) < _TTL_SECONDS:
                return list(cached["symbols"])

            symbols = self._download(name)
            source = "nse_archive"
            if not symbols and not force_refresh:
                symbols = self._load_disk(name)
                source = "disk_cache"
            if not symbols:
                symbols = _STATIC_FALLBACKS.get(name, [])
                source = "static_fallback"

            if symbols:
                self._cache[name] = {
                    "symbols": symbols,
                    "loaded_at": time.time(),
                    "source": source,
                }
                if source == "nse_archive":
                    self._persist_disk(name, symbols)
            else:
                raise RuntimeError(f"No constituents available for {name} from any source")
            return list(symbols)

    def get_info(self, index_name: str) -> Dict[str, object]:
        name = normalize_index_name(index_name)
        symbols = self.get_constituents(name)
        cached = self._cache.get(name, {})
        return {
            "index": name,
            "count": len(symbols),
            "source": cached.get("source"),
            "loaded_at": (
                datetime.fromtimestamp(cached["loaded_at"]).isoformat()
                if cached.get("loaded_at") else None
            ),
            "symbols": symbols,
        }

    # ── internals ──────────────────────────────────────────────────

    def _download(self, name: str) -> List[str]:
        url = f"{_BASE_URL}/{INDEX_SOURCES[name]}"
        try:
            session = requests.Session()
            session.headers.update(_HEADERS)
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            return self._parse_csv(resp.text)
        except Exception as e:
            logger.warning(f"Could not download constituents for {name}: {e}")
            return []

    @staticmethod
    def _parse_csv(text: str) -> List[str]:
        symbols = set()
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            return []
        field_map = {f.strip().upper(): f for f in reader.fieldnames}
        sym_field = field_map.get("SYMBOL")
        if not sym_field:
            return []
        for row in reader:
            symbol = (row.get(sym_field) or "").strip().upper()
            if symbol:
                symbols.add(symbol)
        return sorted(symbols)

    def _disk_path(self, name: str) -> Path:
        return self.cache_dir / f"index_{name.lower()}.csv"

    def _persist_disk(self, name: str, symbols: List[str]):
        try:
            with open(self._disk_path(name), "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["SYMBOL"])
                for s in symbols:
                    writer.writerow([s])
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not persist {name} constituents: {e}")

    def _load_disk(self, name: str) -> List[str]:
        path = self._disk_path(name)
        try:
            if not path.exists():
                return []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                return sorted({row[0].strip().upper() for row in reader if row and row[0].strip()})
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not load {name} constituents from disk: {e}")
            return []


_provider: Optional[IndexUniverseProvider] = None
_provider_lock = Lock()


def get_index_universe() -> IndexUniverseProvider:
    global _provider
    with _provider_lock:
        if _provider is None:
            _provider = IndexUniverseProvider()
        return _provider
