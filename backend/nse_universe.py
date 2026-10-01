"""
NSE Universe Provider
=====================
Sources the complete list of NSE-listed equities from the official NSE
equity master file (EQUITY_L.csv), with:

- In-memory caching with a 24h TTL (configurable via NSE_UNIVERSE_TTL_HOURS)
- Disk persistence (backend/data/nse_universe.csv) so restarts do not
  require a network call
- Graceful fallback to the database pipeline's ticker list when the
  NSE endpoint is unreachable

Usage:
    from nse_universe import get_nse_universe
    universe = get_nse_universe(fallback_tickers_fn=lambda: [...])
    symbols = universe.get_symbols()
"""

import csv
import io
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

NSE_EQUITY_MASTER_URL = os.getenv(
    "NSE_EQUITY_MASTER_URL",
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
)

# Equity series considered part of the tradeable cash-market universe
_TRADEABLE_SERIES = {"EQ", "BE", "BZ", "SM", "ST"}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


class NSEUniverseProvider:
    """Fetch, cache and serve the full NSE equity symbol universe."""

    def __init__(
        self,
        cache_path: Optional[str] = None,
        ttl_hours: Optional[float] = None,
        fallback_tickers_fn: Optional[Callable[[], List[str]]] = None,
    ):
        data_dir = Path(__file__).resolve().parent / "data"
        data_dir.mkdir(exist_ok=True)
        self.cache_path = Path(cache_path) if cache_path else data_dir / "nse_universe.csv"
        ttl = ttl_hours if ttl_hours is not None else float(os.getenv("NSE_UNIVERSE_TTL_HOURS", "24"))
        self.ttl_seconds = max(ttl, 0.25) * 3600
        self.fallback_tickers_fn = fallback_tickers_fn

        self._lock = Lock()
        self._symbols: List[str] = []
        self._meta: Dict[str, object] = {}
        self._loaded_at: float = 0.0
        self._source: str = "none"

    # ── public API ──────────────────────────────────────────────────

    def get_symbols(self, force_refresh: bool = False) -> List[str]:
        """Return the full NSE symbol universe, refreshing if stale."""
        with self._lock:
            if force_refresh or self._is_stale():
                self._refresh_locked()
            return list(self._symbols)

    def get_info(self) -> Dict[str, object]:
        """Return metadata about the current universe snapshot."""
        with self._lock:
            if self._is_stale():
                self._refresh_locked()
            return {
                "count": len(self._symbols),
                "source": self._source,
                "loaded_at": (
                    datetime.fromtimestamp(self._loaded_at).isoformat()
                    if self._loaded_at
                    else None
                ),
                "ttl_seconds": int(self.ttl_seconds),
                "cache_file": str(self.cache_path),
            }

    def refresh(self) -> Dict[str, object]:
        """Force a refresh from the NSE master list."""
        with self._lock:
            self._refresh_locked(force_network=True)
        return self.get_info()

    # ── internals ───────────────────────────────────────────────────

    def _is_stale(self) -> bool:
        return not self._symbols or (time.time() - self._loaded_at) > self.ttl_seconds

    def _refresh_locked(self, force_network: bool = False):
        """Refresh symbols. Order: network → disk cache → DB fallback."""
        symbols = self._fetch_from_nse()
        if symbols:
            self._symbols = symbols
            self._source = "nse_master"
            self._loaded_at = time.time()
            self._persist_to_disk(symbols)
            logger.info(f"NSE universe refreshed from NSE master: {len(symbols)} symbols")
            return

        if not force_network:
            disk_symbols = self._load_from_disk()
            if disk_symbols:
                self._symbols = disk_symbols
                self._source = "disk_cache"
                self._loaded_at = time.time()
                logger.warning(
                    f"NSE master unreachable — using disk cache ({len(disk_symbols)} symbols)"
                )
                return

        if self.fallback_tickers_fn is not None:
            try:
                fallback = sorted({s.strip().upper() for s in self.fallback_tickers_fn() if s})
                if fallback:
                    self._symbols = fallback
                    self._source = "db_pipeline"
                    self._loaded_at = time.time()
                    logger.warning(
                        f"NSE master unreachable — using DB pipeline universe ({len(fallback)} symbols)"
                    )
                    return
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Fallback ticker source failed: {e}")

        if not self._symbols:
            logger.error("No NSE universe available from any source")

    def _fetch_from_nse(self) -> List[str]:
        """Download and parse the official NSE equity master CSV."""
        try:
            session = requests.Session()
            session.headers.update(_HEADERS)
            resp = session.get(NSE_EQUITY_MASTER_URL, timeout=20)
            resp.raise_for_status()
            return self._parse_master_csv(resp.text)
        except Exception as e:
            logger.warning(f"Failed to download NSE equity master: {e}")
            return []

    @staticmethod
    def _parse_master_csv(text: str) -> List[str]:
        symbols = set()
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            return []
        # Normalise header names (NSE files often pad with spaces)
        field_map = {f.strip().upper(): f for f in reader.fieldnames}
        sym_field = field_map.get("SYMBOL")
        series_field = field_map.get("SERIES")
        if not sym_field:
            return []
        for row in reader:
            symbol = (row.get(sym_field) or "").strip().upper()
            if not symbol:
                continue
            if series_field:
                series = (row.get(series_field) or "").strip().upper()
                if series and series not in _TRADEABLE_SERIES:
                    continue
            symbols.add(symbol)
        return sorted(symbols)

    def _persist_to_disk(self, symbols: List[str]):
        try:
            with open(self.cache_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["SYMBOL"])
                for s in symbols:
                    writer.writerow([s])
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not persist NSE universe to disk: {e}")

    def _load_from_disk(self) -> List[str]:
        try:
            if not self.cache_path.exists():
                return []
            with open(self.cache_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # header
                return sorted({row[0].strip().upper() for row in reader if row and row[0].strip()})
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not load NSE universe from disk: {e}")
            return []


_provider: Optional[NSEUniverseProvider] = None
_provider_lock = Lock()


def get_nse_universe(
    fallback_tickers_fn: Optional[Callable[[], List[str]]] = None,
) -> NSEUniverseProvider:
    """Singleton accessor for the NSE universe provider."""
    global _provider
    with _provider_lock:
        if _provider is None:
            _provider = NSEUniverseProvider(fallback_tickers_fn=fallback_tickers_fn)
        elif fallback_tickers_fn is not None and _provider.fallback_tickers_fn is None:
            _provider.fallback_tickers_fn = fallback_tickers_fn
        return _provider
