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
#      strategy stops trading, AND attach the research verdict (holdout
#      t, fold count, rejection reasons) so the dashboard can explain
#      WHY instead of showing a silent "Inactive" card.
#
#   5. In the opt-in shadow lane, publish the best available rejected
#      candidate as an explicitly labelled, paper-only experiment.
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
    # Gold PERP on OKX is XAU-USDT-SWAP. There is no XAUT-USDT-SWAP
    # (the API answers 51001), so the old "XAUT/USDT" pair never fetched
    # a single candle and this symbol was silently skipped every cycle.
    # XAUT-USDT is a SPOT instrument, not the perpetual the executor
    # trades. Keep the perp spelling here so to_ccxt_swap() resolves to
    # the contract that actually exists.
    "XAU/USDT",
    "NEAR/USDT",
]

# Verified against https://www.okx.com/api/v5/public/instruments?instType=SWAP
# on 2026-11-13: 543 SWAP instruments, the ONLY gold perpetual is
# XAU-USDT-SWAP, and there is no XAUT-* SWAP at all. XAUT-USDT is a SPOT
# instrument. XAUTUSDT is therefore not a tradable pair on this executor and
# must never be reintroduced through --symbols or AUTORESEARCH_SYMBOLS.
def reject_untradable_pairs(symbols) -> list:
    """Fail loudly instead of silently scanning a pair OKX cannot trade."""
    out = []
    for symbol in symbols:
        compact = (
            symbol.strip().upper()
            .replace("/", "").replace("-", "").replace(":USDT", "")
        )
        if compact.startswith("XAUT"):
            raise ValueError(
                f"{symbol!r} is not an OKX futures pair. OKX lists only "
                "XAU-USDT-SWAP for gold; use 'XAU/USDT' instead."
            )
        out.append(symbol)
    return out


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
    cleaned = cleaned.replace("/", "").replace("-", "")
    # LEGACY DATA GUARD ONLY. Historic rows were written with both gold
    # spellings; collapsing them here stops a duplicate row for one contract.
    # It is NOT a licence to trade XAUT — reject_untradable_pairs() rejects
    # that spelling before any cycle starts.
    if cleaned.startswith("XAUT"):
        cleaned = "XAU" + cleaned[len("XAUT"):]
    return cleaned


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


def build_research(status: str, report: dict = None, reasons=None) -> dict:
    """The compact, machine-readable verdict the executor persists.

    Before this existed the engine printed a detailed rejection to Railway
    stdout and then published only a silent neutral tombstone. Two weeks of
    correct "no edge" decisions were indistinguishable from a dead cron job.
    Now every row carries the binding gate numbers with it, and /stats can
    explain exactly why a symbol is inactive.
    """
    report = report or {}
    holdout = report.get("holdout") or {}
    params = report.get("params") or {}
    if reasons is None:
        reasons = report.get("reasons") or report.get("detail") or []
    if isinstance(reasons, str):
        reasons = [reasons]

    best = ""
    if params:
        direction = str(params.get("direction", "?")).upper()
        if direction == "LONG":
            best = f"LONG rsi<={params.get('rsi_lower', 0):.0f}"
        else:
            best = f"SHORT rsi>={params.get('rsi_upper', 0):.0f}"

    folds = report.get("folds") or []
    return {
        "status": status,
        "mode": "shadow" if status == "shadow" else "strict",
        "reasons": [str(r) for r in reasons][:6],
        "holdout_n": int(holdout.get("n", 0) or 0),
        "holdout_mean": round(float(holdout.get("mean", 0) or 0), 4),
        "holdout_t": round(float(holdout.get("t_stat", 0) or 0), 4),
        "required_t": round(float(report.get("required_t") or config.MIN_T), 4),
        "folds_pass": int(report.get("folds_pass", 0) or 0),
        "folds_total": len(folds) or config.N_FOLDS,
        "pooled_trades": int(report.get("pooled_trades", 0) or 0),
        "best_candidate": best,
        "exit_is_the_problem": bool(report.get("exit_is_the_problem")),
        "data_bars": int(report.get("bars", 0) or 0),
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }


def push_strategy(symbol: str, direction: str, rsi_lower: float, rsi_upper: float,
                  leverage: str = "3x", dry_run: bool = False,
                  retired: bool = False, research: dict = None,
                  mode: str = None) -> bool:
    """Register (or replace) the strategy the executor will trade.

    The payload carries exactly the fields the executor's
    database.ActiveStrategy understands. The old feeder also sent
    strategy_type and oos_sharpe, which the Go handler silently
    discarded -- which is how a strategy validated as a momentum
    breakout could be executed as a mean reversion.

    `retired` is the explicit retirement flag. The executor stores it and
    refuses to arm a row that carries it, so a retired symbol is blocked
    by state rather than by an unreachable threshold.

    Re-arming works through THIS function, not by weakening the executor's
    guard: when a later cycle finds a winner it calls push_strategy with
    retired=False and reachable thresholds. The executor's DeriveRetired
    then stores retired=false and the row passes CanArm again. The guard
    itself stays absolute, so a retirement marker can never be talked
    back into trading.
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
        # The lane is explicit on the wire. "strict" is a validated winner;
        # "shadow" is a labelled paper-only experiment. The executor will
        # downgrade a shadow row unless its own switch is also on.
        "mode": mode or ("shadow" if (research or {}).get("status") == "shadow" else "strict"),
    }
    # Diagnostic only — it never changes an arming decision. Omit the key
    # entirely when there is no verdict so the executor keeps the row's
    # previous research instead of overwriting it with an empty object.
    if research is not None:
        payload["research"] = research

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


def push_neutral(symbol: str, dry_run: bool = False,
                 research: dict = None) -> bool:
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
    return push_strategy(symbol, "LONG", 0.0, 100.0, dry_run=dry_run,
                         retired=True, research=research)


def push_shadow(symbol: str, params: dict, research: dict,
                dry_run: bool = False) -> bool:
    """Publish the best available rejected candidate as a paper experiment.

    This is the answer to "the bot did nothing for two weeks": on a paper
    deployment, refusing to publish anything also refuses to collect the
    forward data needed to improve. A shadow row is NOT a winner and is
    labelled as such; the executor keeps it paper-only and refuses to arm it
    at all unless SHADOW_TRADING_ENABLED=true is set there too.

    The candidate is still structurally validated: a direction we do not
    execute, or an arm threshold outside (0, 100), is never published.
    """
    if not params:
        return False
    direction = str(params.get("direction", "")).upper()
    lower = float(params.get("rsi_lower", 0) or 0)
    upper = float(params.get("rsi_upper", 0) or 0)
    arm = lower if direction == "LONG" else upper
    if direction not in ("LONG", "SHORT") or not (0 < arm < 100):
        return False
    return push_strategy(symbol, direction, lower, upper,
                         dry_run=dry_run, retired=False,
                         research=research, mode="shadow")


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

    symbols = reject_untradable_pairs(
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else (config.SYMBOLS or DEFAULT_PAIRS)
    )

    log("=" * 70)
    log(f"autoresearch cycle | mode={config.PUBLISH_MODE} | symbols={len(symbols)} | "
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
            # An unavailable feed is NOT a licence to keep trading the
            # previous row. The executor's freshness gate expires rows it
            # has not heard about in 48h, but until then an unvalidated
            # symbol would still look armed. Retiring it explicitly is the
            # conservative choice; the next successful cycle re-arms it by
            # pushing a fresh winner payload with retired=False.
            log(f"  !! {symbol}: no data — cannot validate, retiring (not a winner)")
            if config.RETIRE_FAILED_SYMBOLS:
                if push_neutral(
                    symbol, dry_run=args.dry_run,
                    research=build_research(
                        "no_data",
                        reasons=[f"{symbol}: OKX returned no usable 15m candles this cycle"],
                    ),
                ):
                    pushed += 1
            # Record the outcome either way: `len(decisions)` is reported as
            # "symbols scanned" and a failed feed must not silently vanish.
            decisions[symbol] = {"status": "no_data", "last_bar": None}
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
            published_status = "rejected"

            # On a paper deployment, publishing nothing at all means the
            # system never gathers forward evidence and can never improve.
            # The shadow lane publishes the best available candidate with an
            # explicit label. It changes NO statistical gate; a shadow row is
            # still a rejected strategy, just one we are willing to observe.
            if config.PUBLISH_MODE == "shadow" and params:
                if push_shadow(symbol, params, build_research("shadow", report),
                               dry_run=args.dry_run):
                    pushed += 1
                    published_status = "shadow"
                    log(f"  SHADOW: paper-only candidate {params['direction']} "
                        f"rsi_lower={params['rsi_lower']:.0f} "
                        f"rsi_upper={params['rsi_upper']:.0f} "
                        f"(gates NOT met — labelled as an experiment)")
                else:
                    log("  !! shadow candidate was not publishable, retiring instead")

            if published_status == "rejected" and config.RETIRE_FAILED_SYMBOLS:
                if push_neutral(symbol, dry_run=args.dry_run,
                                research=build_research("rejected", report)):
                    pushed += 1
            decisions[symbol] = {
                "status": published_status,
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
            research=build_research("winner", report),
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
            "mode": config.PUBLISH_MODE,
            "gates": {
                "min_t": config.MIN_T,
                "required_fold_pass": config.REQUIRED_FOLD_PASS,
                "n_folds": config.N_FOLDS,
                "min_validation_trades": config.MIN_VALIDATION_TRADES,
                "min_pooled_trades": config.MIN_POOLED_TRADES,
                "fee_gross": config.FEE_GROSS,
            },
            "decisions": decisions,
            "active": {
                s: d["params"]
                for s, d in decisions.items()
                if d.get("status") == "accepted"
            },
        }
    )

    accepted = sum(1 for d in decisions.values() if d.get("status") == "accepted")
    shadows = sum(1 for d in decisions.values() if d.get("status") == "shadow")
    log("=" * 70)
    log(f"done | mode={config.PUBLISH_MODE} | {accepted} accepted, {shadows} shadow, "
        f"{pushed} pushed, {len(decisions)} symbols scanned")
    if accepted == 0 and shadows == 0:
        log("NO STRATEGY PASSED THE GATES THIS CYCLE. Nothing was published.")
        log("That is the honest answer for this data, not a failure to run.")
        log("Diagnostic telemetry went with every neutral row: see /api/research/status.")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # the cron job must fail loudly, not silently
        log(f"CRITICAL ERROR in cron execution: {type(exc).__name__}: {exc}")
        sys.exit(1)
