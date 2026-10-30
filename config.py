# --------------------------------------------------------------
# config.py - single source of truth for the autoresearch feeder
# --------------------------------------------------------------
#
# Every constant in this file mirrors something the Go executor
# (tethgard-executor-go) actually does at runtime. If a number here
# is not in the executor, it does not belong in the search.
#
# The rule that governs this file: the feeder may only search over
# knobs that survive the trip to production. The executor stores a
# strategy as {symbol, direction, leverage, rsi_lower, rsi_upper} and
# discards everything else, so anything else the optimiser tunes is a
# parameter that is validated in the backtest and silently ignored in
# production. That mismatch was the single worst defect of the old
# feeder: a "RSI_MOMENTUM_BREAKOUT" winner was executed by the Go
# engine as an RSI mean-reversion, i.e. the exact opposite trade,
# while still reporting its own backtest Sharpe as proof of quality.
# --------------------------------------------------------------
import os


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ------------------------------------------------------------------ data
TIMEFRAME = "15m"
BAR_MINUTES = 15
BARS_PER_DAY = 96  # 24h * 4 bars per hour

# The old feeder used 1000 bars, i.e. 10.4 days. On a 10-day window the
# standard error of an annualised Sharpe is sqrt(35040/166) ~= 14.5 on the
# test fold alone, so the acceptance gate of "oos_sharpe > 0.1" was a
# fraction of one standard error: it accepted noise. 90 days is the
# minimum that makes a walk-forward fold worth having (see research/README).
DATA_DAYS = _int("AUTORESEARCH_DATA_DAYS", 180)
DATA_BARS = _int("AUTORESEARCH_DATA_BARS", DATA_DAYS * BARS_PER_DAY)
EXCHANGE_ID = os.getenv("AUTORESEARCH_EXCHANGE", "okx")

# ------------------------------------------------------------------ entry filter
# The executor applies its own entry filter to EVERY entry, before it
# commits money (internal/strategy/entry_filter.go). A backtest that
# ignores it scores a strategy that trades far more often than the one
# that runs. These are the executor's defaults.
ENTRY_FILTER_ENABLED = _bool("AUTORESEARCH_ENTRY_FILTER", True)
ENTRY_FILTER_MAX_ADX = _float("AUTORESEARCH_ENTRY_FILTER_MAX_ADX", 32.0)
ENTRY_FILTER_MIN_DIST_ATR = _float("AUTORESEARCH_ENTRY_FILTER_MIN_DIST_ATR", 1.0)

# ------------------------------------------------------------------ costs
LEVERAGE = _float("AUTORESEARCH_LEVERAGE", 3.0)
# OKX taker fee on USDT swaps is 0.05% of notional per side, so a round trip
# costs 0.10% of notional. Measured in gross margin percent (price move x
# leverage x 100) that is 0.10 x leverage, i.e. 0.30 at 3x. This is the
# convention used by the executor's own research harness, and it is the
# number the old backtest simply did not have: evaluator.py modelled ZERO
# transaction costs, so every "winner" it found was free money.
TAKER_FEE_NOTIONAL = _float("AUTORESEARCH_TAKER_FEE", 0.0005)
FEE_GROSS = TAKER_FEE_NOTIONAL * 2.0 * LEVERAGE * 100.0
# Fill slippage against the stop, matching the research harness.
SLIPPAGE = _float("AUTORESEARCH_SLIPPAGE", 0.0002)

# ------------------------------------------------------------------ signal
# Go: calculateRSI(candles, 14)  -> the period is hardcoded.
RSI_PERIOD = 14
# Go: RSIReversalThreshold = 2.0 -> the confirmation size is hardcoded.
RSI_REVERSAL = 2.0
# Go: isStrongTrend := adx14 >= 25.0 -> the inversion trigger is hardcoded.
INVERSION_ADX = 25.0
# Go applies the inversion; the backtest reproduces it so that what is
# scored is what is traded. Turn off only to measure its contribution.
INVERSION_ENABLED = _bool("AUTORESEARCH_INVERSION", True)

# The only free knob. For a LONG the executor arms on rsi_lower
# (liveRSI <= rsi_lower); for a SHORT it arms on rsi_upper
# (liveRSI >= rsi_upper). The other bound is inert for that direction.
RSI_LOWER_RANGE = (20, 45)
RSI_UPPER_RANGE = (55, 80)

# ------------------------------------------------------------------ exit
# Mirror of Go internal/strategy/exit_policy.go DefaultExitPolicy().
# The old backtest used its own stop_loss_pct (2-4% of PRICE, i.e. 6-12%
# gross at 3x) while production risks 3% gross. The geometries differed by
# 2-4x, so the backtest was measuring a robot nobody runs.
EXIT_POLICY = {
    "initial_sl_gross_pct": 3.0,
    "take_profit_gross_pct": 0.0,
    "be_arm_gross_pct": 2.0,
    "be_lock_gross_pct": 0.6,
    "trail_arm_gross_pct": 3.0,
    "trail_giveback_gross_pct": 0.8,
    "trail_widen_peak_gross_pct": 12.0,
    "trail_widen_giveback_gross_pct": 2.0,
    "min_lock_gross_pct": 0.4,
    "max_hold_minutes": 480,
}
MAX_HOLD_BARS = EXIT_POLICY["max_hold_minutes"] // BAR_MINUTES  # 32 bars

# ------------------------------------------------------------------ search
N_FOLDS = _int("AUTORESEARCH_FOLDS", 5)
N_TRIALS = _int("AUTORESEARCH_TRIALS", 120)
# Fraction of the sample reserved as a final holdout that is never used for
# selection. The old feeder evaluated the winning parameters on the same
# window that selected them.
HOLDOUT_FRACTION = _float("AUTORESEARCH_HOLDOUT_FRACTION", 0.25)

# ------------------------------------------------------------------ gates
# A fold only counts if it produced enough trades to mean anything. One
# trade is not evidence, but 60 trades in an 18-day window is not
# reachable at 15m, so the per-fold bar is deliberately modest and the
# pooled bar carries the statistical weight.
MIN_FOLD_TRADES = _int("AUTORESEARCH_MIN_FOLD_TRADES", 12)
MIN_TRAIN_TRADES = _int("AUTORESEARCH_MIN_TRAIN_TRADES", 10)
MIN_POOLED_TRADES = _int("AUTORESEARCH_MIN_POOLED_TRADES", 40)
MIN_VALIDATION_TRADES = _int("AUTORESEARCH_MIN_VALIDATION_TRADES", 10)
# Majority of the walk-forward folds must be individually profitable.
REQUIRED_FOLD_PASS = _int("AUTORESEARCH_REQUIRED_FOLDS", 4)
# t-statistic of the holdout trades. The metric is per-trade expectancy,
# not Sharpe: with 10-40 trades a Sharpe ratio is dominated by the
# standardisation and tells you nothing a t-statistic does not.
MIN_T = _float("AUTORESEARCH_MIN_T", 1.96)
# Multiple testing. The search evaluates N_TRIALS x N_FOLDS candidates.
# Even though the final holdout is scored once for a single chosen
# strategy, the same data is re-scored every 30 minutes, so the honest
# bar is reported and can be enforced.
ALPHA = _float("AUTORESEARCH_ALPHA", 0.05)
REQUIRE_DEFLATED = _bool("AUTORESEARCH_REQUIRE_DEFLATED", False)

# ------------------------------------------------------------------ delivery
WEBHOOK_URL = os.getenv("TETHGARD_WEBHOOK_URL", "").strip()
WEBHOOK_TIMEOUT = _float("AUTORESEARCH_WEBHOOK_TIMEOUT", 20.0)
# Empty means "use the default pair list in main.py". The old feeder
# scanned ten pairs; the override exists so a cycle can be narrowed
# without editing code.
SYMBOLS = [
    s.strip()
    for s in os.getenv("AUTORESEARCH_SYMBOLS", "").split(",")
    if s.strip()
]
# When a symbol fails the gate, push a neutral row so the executor stops
# trading a stale strategy. The executor has no expiry of its own
# (GetActiveStrategies returns every row), so a symbol pushed once would
# otherwise keep trading forever.
RETIRE_FAILED_SYMBOLS = _bool("AUTORESEARCH_RETIRE_FAILED", True)
