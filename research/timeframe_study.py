#!/usr/bin/env python3
"""Empirically test whether another timeframe finds a winning strategy.

This does NOT weaken any gate. It runs the exact same BayesianStrategyEngine,
the exact same entry filters, exit policy and holdout gates, but varies the
candle timeframe. The executor must be changed to match before a winner on
another timeframe can be traded, so this is a research result, not a payload.

Run: python research/timeframe_study.py
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ccxt
import numpy as np
import pandas as pd

import config
import strategy_engine
from main import fetch_ohlcv


SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
           "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "NEAR/USDT", "SUI/USDT", "XAU/USDT"]
TIMEFRAMES = ["15m", "1h", "4h", "1d"]


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def study(timeframe):
    # Override the two constants the engine derives its horizons from.
    config.TIMEFRAME = timeframe
    bar_minutes = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}[timeframe]
    config.BAR_MINUTES = bar_minutes
    config.BARS_PER_DAY = 1440 // bar_minutes
    config.MAX_HOLD_BARS = max(2, config.EXIT_POLICY["max_hold_minutes"] // bar_minutes)
    total_bars = config.DATA_DAYS * config.BARS_PER_DAY
    # fetch_ohlcv paginates with the module-level BAR_MS, which is hardcoded
    # to 15m. Patch it for this timeframe or longer bars return only a couple
    # of pages and the study silently tests the wrong sample.
    import main
    main.BAR_MS = bar_minutes * 60 * 1000

    log(f"=== {timeframe}: {config.DATA_DAYS}d x {config.BARS_PER_DAY}/d = {total_bars} bars ===")
    results = []
    for symbol in SYMBOLS:
        try:
            frame = fetch_ohlcv(symbol, total_bars=total_bars)
        except Exception as exc:
            log(f"  {symbol}: fetch error {type(exc).__name__}")
            continue
        if frame is None or len(frame) < 300:
            log(f"  {symbol}: insufficient data")
            continue
        try:
            engine = strategy_engine.BayesianStrategyEngine(frame)
            winner, report = engine.find_winner(pair=symbol)
        except Exception as exc:
            log(f"  {symbol}: engine error {type(exc).__name__}: {exc}")
            continue
        holdout = report.get("holdout") or {}
        status = "WINNER" if winner else "reject"
        results.append({
            "symbol": symbol,
            "status": status,
            "t": round(float(holdout.get("t_stat", 0) or 0), 3),
            "mean": round(float(holdout.get("mean", 0) or 0), 3),
            "n": int(holdout.get("n", 0) or 0),
            "folds": f"{report.get('folds_pass', 0)}/{len(report.get('folds') or [])}",
            "pooled": report.get("pooled_trades", 0),
            "best": (report.get("params") or {}),
            "reasons": (report.get("reasons") or [])[:2],
        })
        log(f"  {symbol:10s} {status:6s} t={results[-1]['t']:>6} "
            f"mean={results[-1]['mean']:>6} n={results[-1]['n']:>4} "
            f"folds={results[-1]['folds']} pooled={results[-1]['pooled']}")
    wins = [r for r in results if r["status"] == "WINNER"]
    best = sorted(results, key=lambda r: r["t"], reverse=True)[:3]
    log(f"RESULT {timeframe}: {len(wins)} winner(s) / {len(results)} symbols; "
        f"best t: {[(r['symbol'], r['t']) for r in best]}")
    return {"timeframe": timeframe, "wins": wins, "results": results}


def main():
    all_results = []
    for tf in TIMEFRAMES:
        try:
            all_results.append(study(tf))
        except Exception as exc:
            log(f"FATAL {tf}: {type(exc).__name__}: {exc}")
    log("=== SUMMARY ===")
    for res in all_results:
        print(f"{res['timeframe']:>4}: {len(res['wins'])} winner(s) | "
              f"top t = {max([r['t'] for r in res['results']] or [0]):.3f}")
    out = os.path.join(os.path.dirname(__file__), "timeframe_study_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        import json
        json.dump(all_results, fh, indent=2, default=str)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
