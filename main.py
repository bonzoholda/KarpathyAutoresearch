# --------------------------------------------------------------
# main.py - the 30-minute cycle
# --------------------------------------------------------------
#
# What this does, in order:
#
#   1. Fetch 90 days of 15m candles from OKX for every symbol.
#      The exchange is the SAME one the executor trades on, and the
#      instrument is the SWAP (BTC/USDT:USDT), not the spot market.
#      The old feeder pulled from Binance spot while the executor
#      traded OKX swaps.
#
#   2. Run the walk-forward search on each symbol.
#
#   3. Push a strategy to the executor ONLY if the search survived
#      its own gates. There is no fallback to a hardcoded strategy:
#      the old feeder fell back to MA_CROSSOVER, a strategy the
#      executor cannot even execute, and shipped it anyway.
#
#   4. Push a NEUTRAL row for every symbol that failed, so a stale
#      strategy stops trading. The executor never expires a strategy
#      of its own accord (GetActiveStrategies returns every row), so a
#      symbol pushed once would otherwise keep trading forever.
#
# Nothing here invents a winner. A cycle that finds nothing says so.
# --------------------------------------------------------------
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

import config

STATE_PATH = os.path.join("config", "strategy_config.json")
BAR_MS = 15 * 60 * 1000

# Production endpoint of the Go executor. Kept as a default so the
# service keeps working without extra configuration, but the
# environment variable always wins.
DEFAULT_WEBHOOK_URL = (
    "https://tethgard-multi-go-production.up.railway.app/webhook/strategy"
)

# The pairs the old feeder scanned.
DEFAULT_PAIRS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "XAUT/USDT",
    "NEAR/USDT",
]


# ------------------------------------------------------------------ utils
def executor_symbol(symbol: str) -> str:
    """The compact form the executor stores: 'BTC/USDT' -> 'BTCUSDT'.

    The executor's cleanSymbolKey() turns this back into BTC-USDT-SWAP,
    so the compact form is what it expects. It is used here on purpose:
    the symbol column is UNIQUE, and the rows already in the database
    were written in this form. Pushing 'BTC-USDT-SWAP' would create a
    SECOND row for the same instrument and double the trading.
    """
    cleaned = symbol.strip().upper()
    if ":" in cleaned:
        # 'BTC/USDT:USDT' -> 'BTC/USDT': drop the settlement currency.
        cleaned = cleaned.split(":", 1)[0]
    if cleaned.endswith("-SWAP"):
        cleaned = cleaned[: -len("-SWAP")]
    return cleaned.replace("/", "").replace("-", "")


def to_ccxt_swap(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC/USDT:USDT' (the OKX perpetual)."""
    if ":" in symbol:
        return symbol
    base, _, quote = symbol.partition("/")
    return f"{base}/{quote}:{quote}"


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] {message}", flush=True)


# ------------------------------------------------------------------- data
def fetch_ohlcv(symbol: str, total_bars: int = None):
    """Fetch `total_bars` of 15m history, paginating forward from the oldest bar."""
    import ccxt

    total_bars = total_bars or config.DATA_BARS
    exchange = ccxt.okx({"enableRateLimit": True})
    market = to_ccxt_swap(symbol)

    now = exchange.milliseconds()
    since = now - int(total_bars * 1.2) * BAR_MS

    rows = []
    guard = 0
    while since < now and guard < 400:
        guard += 1
        try:
            batch = exchange.fetch_ohlcv(
                market, timeframe=config.TIMEFRAME, since=since, limit=300
            )
        except Exception as exc:  # network, rate limit, bad symbol
            log(f"  !! fetch error for {market}: {type(exc).__name__}: {exc}")
            break
        if not batch:
            break
        rows.extend(batch)
        newest = batch[-1][0]
        if newest <= since:
            break  # no forward progress, stop
        since = newest + BAR_MS

    if not rows:
        return None

    frame = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    )
    frame = frame.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    frame["datetime"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    frame = frame.set_index("datetime")
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = frame[col].astype(float)

    if len(frame) < 500:
        log(f"  !! {market}: only {len(frame)} bars, not enough")
        return None
    return frame


# --------------------------------------------------------------- delivery
def webhook_url() -> str:
    for name in ("EXECUTOR_WEBHOOK_URL", "TETHGARD_WEBHOOK_URL"):
        value = os.getenv(name, "").strip()
        if value:
            return value.rstrip("/")
    return config.WEBHOOK_URL or DEFAULT_WEBHOOK_URL


def push_strategy(symbol: str, direction: str, rsi_lower: float, rsi_upper: float,
                  leverage: str = "3x", dry_run: bool = False,
                  retired: bool = False) -> bool:
    """Register (or replace) the strategy the executor will trade.

    The payload carries exactly the fields the executor's
    database.ActiveStrategy understands. The old feeder also sent
    strategy_type and oos_sharpe, which the Go handler silently
    discarded -- which is how a strategy validated as a momentum
    breakout could be executed as a mean reversion.

    `retired` is the explicit retirement flag. The executor stores it and
    refuses to arm a row that carries it, so a retired symbol is blocked
    by state rather than by an unreachable threshold.
    """
    url = webhook_url()
    payload = {
        "symbol": executor_symbol(symbol),
        "direction": direction,
        "leverage": leverage,
        "rsi_lower": float(rsi_lower),
        "rsi_upper": float(rsi_upper),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "retired": bool(retired),
    }

    if dry_run:
        log(f"  [dry-run] would push {json.dumps(payload)} -> {url}")
        return True

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=config.WEBHOOK_TIMEOUT,
        )
        if response.status_code >= 300:
            log(f"  !! push rejected: HTTP {response.status_code} {response.text[:200]}")
            return False
        log(f"  -> pushed {json.dumps(payload)}")
        return True
    except Exception as exc:
        log(f"  !! push failed: {type(exc).__name__}: {exc}")
        return False


def push_neutral(symbol: str, dry_run: bool = False) -> bool:
    """Retire a strategy so the executor stops trading that symbol.

    Two things are pushed together, and both matter:

    1. `retired = true` -- the explicit flag. The executor stores it and its
       arming gate refuses any row that carries it.
    2. The neutral thresholds LONG / rsi_lower = 0 -- the legacy shape. The
       row still exists so the symbol stays visible in the dashboard, and
       the shape keeps older executors (which predate the flag) from
       trading it too.

    The previous docstring here claimed "the RSI is never 0". That was
    wrong: the executor's own calculateRSI returns exactly 0.0 when the
    average gain is zero, which happens on a pure downtrend series. A LONG
    with rsi_lower = 0 arms on liveRSI <= 0, so the neutral shape alone
    could never be relied on to block a position. The explicit flag is
    what makes the retirement safe.
    """
    return push_strategy(symbol, "LONG", 0.0, 100.0, dry_run=dry_run, retired=True)


# ------------------------------------------------------------------ state
def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}
    return {}


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def load_active_config():
    """The most recently published decisions, or None."""
    state = load_state()
    return state.get("active") or None


# ------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="TethGard autoresearch feeder")
    parser.add_argument("--dry-run", action="store_true", help="do not call the webhook")
    parser.add_argument("--symbols", help="comma separated list, overrides config")
    args = parser.parse_args()

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else (config.SYMBOLS or DEFAULT_PAIRS)
    )

    log("=" * 70)
    log(f"autoresearch cycle | symbols={len(symbols)} | "
        f"{config.DATA_DAYS} days of {config.TIMEFRAME} ({config.DATA_BARS} bars)")
    log(f"candidates per fold = {26 + 26} | min holdout trades = "
        f"{config.MIN_VALIDATION_TRADES} | required t = {config.MIN_T}")
    log("=" * 70)

    from strategy_engine import BayesianStrategyEngine  # imported late: heavy

    decisions = {}
    pushed = 0

    for symbol in symbols:
        log(f"- {symbol}")
        frame = fetch_ohlcv(symbol)
        if frame is None:
            log(f"  !! {symbol}: no data, skipping (this is NOT a decision)")
            continue

        last_bar = int(frame["ts"].iloc[-1])

        engine = BayesianStrategyEngine(frame)
        started = time.time()
        winner, report = engine.find_winner(pair=symbol)
        elapsed = time.time() - started

        params = report.get("params") or {}
        holdout = report.get("holdout") or {}
        folds = report.get("folds", [])
        log(f"  searched {len(frame)} bars in {elapsed:.1f}s | "
            f"candidates={report.get('candidates')} evaluations={report.get('evaluations')}")
        if params:
            log(f"  development best: {params['direction']} "
                f"rsi_lower={params['rsi_lower']:.0f} rsi_upper={params['rsi_upper']:.0f}")
        if holdout:
            log(f"  HOLDOUT (production ladder): n={holdout.get('n')} "
                f"mean={holdout.get('mean', 0):.2f} gross "
                f"t={holdout.get('t_stat', 0):.2f} "
                f"(required {report.get('required_t', 0):.2f}, "
                f"multiplicity-critical {report.get('critical_t', 0):.2f})")
            fwd = report.get("holdout_forward") or {}
            if fwd:
                log(f"  HOLDOUT (forward, no exit rule): n={fwd.get('n')} "
                    f"mean={fwd.get('mean', 0):.2f} gross "
                    f"t={fwd.get('t_stat', 0):.2f}   <- the entry's own edge")
            fb = report.get("forward_best") or {}
            if fb:
                p = fb["params"]
                fh = fb["holdout_forward"]
                fl = fb["holdout_ladder"]
                log(f"  BEST-FORWARD candidate ({p['direction']} "
                    f"rsi_lower={p['rsi_lower']:.0f}): holdout forward "
                    f"{fh.get('mean', 0):.2f} gross t={fh.get('t_stat', 0):.2f} "
                    f"-> with the production ladder {fl.get('mean', 0):.2f} gross "
                    f"(the ladder gives up "
                    f"{fh.get('mean', 0) - fl.get('mean', 0):.2f} per trade)")
            log(f"  WALK-FORWARD: {report.get('folds_pass')}/{len(folds)} folds passed | "
                f"pooled {report.get('pooled_trades')} trades "
                f"mean {report.get('pooled_mean', 0):.2f} gross")

        if winner is None:
            reasons = report.get("reasons") or [report.get("detail", "unknown")]
            log(f"  REJECTED: {'; '.join(str(r) for r in reasons)}")
            if config.RETIRE_FAILED_SYMBOLS:
                if push_neutral(symbol, dry_run=args.dry_run):
                    pushed += 1
            decisions[symbol] = {
                "status": "rejected",
                "reasons": [str(r) for r in reasons],
                "last_bar": last_bar,
                "params": params,
                "holdout": holdout,
            }
            continue

        log(f"  ACCEPTED: {winner['direction']} rsi_lower={winner['rsi_lower']:.0f} "
            f"rsi_upper={winner['rsi_upper']:.0f}")
        ok = push_strategy(
            symbol,
            winner["direction"],
            winner["rsi_lower"],
            winner["rsi_upper"],
            dry_run=args.dry_run,
        )
        if ok:
            pushed += 1
        decisions[symbol] = {
            "status": "accepted" if ok else "push_failed",
            "params": winner,
            "holdout": holdout,
            "last_bar": last_bar,
            "folds_pass": report.get("folds_pass"),
            "pooled_trades": report.get("pooled_trades"),
        }

    save_state(
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "decisions": decisions,
            "active": {
                s: d["params"]
                for s, d in decisions.items()
                if d.get("status") == "accepted"
            },
        }
    )

    accepted = sum(1 for d in decisions.values() if d.get("status") == "accepted")
    log("=" * 70)
    log(f"done | {accepted} accepted, {pushed} pushed, {len(decisions)} symbols scanned")
    if accepted == 0:
        log("NO STRATEGY PASSED THE GATES THIS CYCLE. Nothing was published.")
        log("That is the honest answer for this data, not a failure to run.")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # the cron job must fail loudly, not silently
        log(f"CRITICAL ERROR in cron execution: {type(exc).__name__}: {exc}")
        sys.exit(1)
