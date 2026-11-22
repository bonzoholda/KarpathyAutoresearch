#!/usr/bin/env python3
"""Cron entry point for the funding-percentile observation lane.

Usage:
  python -m alpha_funding.run --dry-run          # compute + show, no POST
  python -m alpha_funding.run                    # compute + publish shadow rows

Cron example (every 8h, just after OKX funding settlement):
  0 */8 * * * cd /path/to/KarpathyAutoresearch && \
    /path/to/python -m alpha_funding.run >> alpha_funding/cron.log 2>&1

Safe by construction: published rows can never arm because their RSI
thresholds are 0/100, which the executor's CanArm() rejects.
"""
import argparse
import json
import sys
from datetime import datetime, timezone

from . import config, publisher, signal


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="compute only, do not POST")
    ap.add_argument("--symbols", default=None,
                    help="comma-separated override of FUNDING_SYMBOLS")
    ap.add_argument("--force", action="store_true",
                    help="publish even if the funding timestamp is unchanged")
    args = ap.parse_args(argv)

    symbols = ([s.strip() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else config.SYMBOLS)
    try:
        import ccxt
    except ImportError:
        log("ccxt is required"); return 2
    exchange = ccxt.okx({"enableRateLimit": True})
    state = publisher.load_state()
    rows = []

    for sym in symbols:
        try:
            points = signal.fetch_funding(exchange, sym)
            result = signal.evaluate(points)
            result["symbol"] = sym
            key = signal.executor_symbol(sym)
            last_ts = state.get(key, {}).get("symbol_ts")
            fresh = args.force or last_ts != result.get("symbol_ts")
            if result["fire"] and fresh:
                payload = publisher.payload_for(sym, result)
                resp = publisher.post(payload, dry_run=args.dry_run)
                log(f"{sym}: FIRE funding={result['funding_rate']:.6f} "
                    f"<= q{result['percentile']:.2f}={result['threshold']:.6f} "
                    f"posted={resp.get('ok')} dry={args.dry_run}")
                if not args.dry_run:
                    state[key] = {"symbol_ts": result["symbol_ts"],
                                  "fired_at": result["evaluated_at"],
                                  "result": result, "post": resp}
            else:
                log(f"{sym}: no fire ({result['reason']}, fresh={fresh}, "
                    f"funding={result.get('funding_rate')})")
            rows.append({"ts": result["evaluated_at"], "symbol": sym,
                         "fired": bool(result["fire"] and fresh), "result": result})
        except Exception as exc:
            log(f"{sym}: ERROR {type(exc).__name__}: {exc}")
            rows.append({"ts": signal.utcnow(), "symbol": sym,
                         "error": f"{type(exc).__name__}: {exc}"})

    publisher.append_audit(rows)
    if not args.dry_run:
        publisher.save_state(state)
    log(f"done: {len(rows)} symbols, {sum(1 for r in rows if r.get('fired'))} fired")
    return 0


if __name__ == "__main__":
    sys.exit(main())
