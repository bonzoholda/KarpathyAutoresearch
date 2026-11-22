"""Pure signal computation: OKX funding rate percentile -> SHORT candidates.

No network calls at import time, no writes, no dependency on the main
autoresearch modules. All functions are deterministic and independently
testable.
"""
import json
import math
from datetime import datetime, timezone

from . import config


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def fetch_funding(exchange, symbol, days=None):
    """Return [(timestamp_ms, rate), ...] with explicit pagination."""
    days = days or config.LOOKBACK_DAYS
    end = exchange.milliseconds()
    since = end - days * 86400000
    rows, guard = [], 0
    while since < end and guard < 40:
        guard += 1
        batch = exchange.fetch_funding_rate_history(symbol.replace("/", "-") + "-SWAP",
                                                    since=since, limit=100)
        if not batch:
            break
        rows.extend(batch)
        ts = batch[-1].get("timestamp")
        if ts is None or ts <= since:
            break
        since = ts + 1
    points = sorted({int(r["timestamp"]): float(r["fundingRate"])
                     for r in rows if r.get("timestamp") is not None}.items())
    return points


def quantile(values, q):
    """Linear-interpolated quantile; values must be a non-empty sequence."""
    arr = sorted(float(v) for v in values)
    if not arr:
        raise ValueError("empty values")
    if len(arr) == 1:
        return arr[0]
    pos = (len(arr) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    return arr[lo] + (arr[hi] - arr[lo]) * (pos - lo)


def evaluate(points, q_lo=None, min_history=None, now_ms=None):
    """Evaluate the LATEST funding observation against its trailing history.

    ``points`` is [(ts_ms, rate), ...]. The threshold uses only observations
    strictly before the latest one, so there is no self-leakage.
    """
    q_lo = config.Q_LO if q_lo is None else q_lo
    min_history = config.MIN_HISTORY if min_history is None else min_history
    if len(points) < min_history + 1:
        return {"fire": False, "reason": "insufficient_history",
                "n": len(points), "required": min_history + 1}
    ts, rate = points[-1]
    history = [v for _, v in points[:-1]]
    threshold = quantile(history, q_lo)
    fire = rate <= threshold
    return {
        "fire": bool(fire),
        "reason": "funding_at_or_below_percentile" if fire else "funding_above_percentile",
        "symbol_ts": ts,
        "funding_rate": rate,
        "threshold": threshold,
        "percentile": q_lo,
        "history_n": len(history),
        "direction": "SHORT" if fire else None,
        "evaluated_at": utcnow(),
        "now_ms": now_ms,
    }


def executor_symbol(symbol):
    """Mirror main.executor_symbol: OKX perpetual compact form."""
    return symbol.replace("/", "").replace("-", "").upper()


def build_research(result, symbol):
    """Diagnostic payload. Explicitly shadow - never a validated winner."""
    return {
        "status": "shadow",
        "gate_profile": "funding_percentile_observation",
        "reasons": [
            "funding-percentile SHORT observation lane",
            result["reason"],
            "paper only: RSI thresholds are unreachable by design",
            config.EVIDENCE["caveat"],
        ],
        "signal": result,
        "evidence": {**config.EVIDENCE, "symbol": symbol},
        "scanned_at": utcnow(),
    }
