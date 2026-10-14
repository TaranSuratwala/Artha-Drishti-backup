"""
Detect upper/lower circuit conditions from OHLCV data.
NSE circuit limits: 2%, 5%, 10%, 20% based on stock category.
"""

def detect_circuit(row: dict, prev_close: float,
                   circuit_pcts: list = [0.02, 0.05, 0.10, 0.20]) -> dict:
    """
    Detect if a bar shows circuit limit characteristics.
    
    Returns:
        {"at_uc": bool, "at_lc": bool, "circuit_pct": float}
    """
    if prev_close <= 0:
        return {"at_uc": False, "at_lc": False, "circuit_pct": None}

    open_p = row.get("open", 0)
    high = row.get("high", 0)
    low = row.get("low", 0)
    close = row.get("close", 0)

    # UC indicators: high ≈ close ≈ open, near-zero volume, change near circuit %
    for pct in sorted(circuit_pcts):
        uc_price = prev_close * (1 + pct)
        lc_price = prev_close * (1 - pct)

        # Upper circuit
        if (abs(close - uc_price) / uc_price < 0.001 and
            abs(high - close) / close < 0.001):
            return {"at_uc": True, "at_lc": False, "circuit_pct": pct}

        # Lower circuit
        if (abs(close - lc_price) / lc_price < 0.001 and
            abs(low - close) / close < 0.001):
            return {"at_uc": False, "at_lc": True, "circuit_pct": pct}

    return {"at_uc": False, "at_lc": False, "circuit_pct": None}
