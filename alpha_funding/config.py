"""Configuration for the funding-percentile observation lane.

Every value has a safe default. Nothing here modifies the main autoresearch
config: this module is standalone and imported only from its own entry point.
"""
import os


def _f(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _i(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Universe. Kept to the symbols that had enough aligned history in the study.
SYMBOLS = [s.strip() for s in os.getenv(
    "FUNDING_SYMBOLS",
    "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,ADA/USDT,AVAX/USDT,"
    "LINK/USDT,DOT/USDT,LTC/USDT,ATOM/USDT,UNI/USDT,OP/USDT",
).split(",") if s.strip()]

# Signal: SHORT when the latest funding rate is at or below this percentile
# of the trailing distribution. The LONG leg was loss-making in validation,
# so it is deliberately not emitted.
Q_LO = _f("FUNDING_Q_LO", 0.15)
LOOKBACK_DAYS = _i("FUNDING_LOOKBACK_DAYS", 90)
MIN_HISTORY = _i("FUNDING_MIN_HISTORY", 20)
HOLD_DAYS = _i("FUNDING_HOLD_DAYS", 3)
LEVERAGE = _f("FUNDING_LEVERAGE", 3.0)

# Publishing. Same single door as the main scanner.
WEBHOOK_URL = (os.getenv("EXECUTOR_WEBHOOK_URL", "").strip()
               or os.getenv("TETHGARD_WEBHOOK_URL", "").strip()
               or "https://tethgard-multi-go-production.up.railway.app/webhook/strategy")
WEBHOOK_TIMEOUT = _f("FUNDING_WEBHOOK_TIMEOUT", 20.0)
STATE_PATH = os.getenv("FUNDING_STATE_PATH", "alpha_funding/state.json")
AUDIT_PATH = os.getenv("FUNDING_AUDIT_PATH", "alpha_funding/signals.jsonl")

# Evidence snapshot from research/validate_funding.py, carried in diagnostics.
EVIDENCE = {
    "family": "funding_percentile_short",
    "holdout_t": 2.994,
    "holdout_mean_gross_pct": 1.491,
    "holdout_trades": 601,
    "holdout_profit_factor": 1.401,
    "dev_t": 0.067,
    "holdout_days": 23,
    "caveat": "underpowered: short holdout, dev flat, 4/5 positive folds",
}
