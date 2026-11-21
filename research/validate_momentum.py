#!/usr/bin/env python3
"""Rigorous validation of the single signal that survived the broad sweep.

The broad signal_family_study.py measured every family on the FULL sample.
A t-stat from a full-sample scan is in-sample and is inflated by multiple
testing: 8 param sets x 11 symbols x 2 directions = 176 comparisons.

This script fixes both problems:
  1. Indicators are computed ONCE on the full series, then sliced (no
     recomputation on short windows which changes their meaning).
  2. Development / walk-forward folds / final holdout are time-ordered.
  3. Bonferroni critical t is reported for 176 comparisons.

Run: python research/validate_momentum.py
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
import exit_policy
import indicators
from main import fetch_ohlcv

SYMBOL = "NEAR/USDT"
LOOKBACK = 20
PCT = 4.0
N_COMPARISONS = 8 * 11 * 2  # families x symbols x directions from the sweep


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def build_arrays(frame):
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    vol = frame["volume"].to_numpy(dtype=float)
    adx = indicators.wilder_adx(high, low, close, 14)
    atr = indicators.atr_series(high, low, close, 14)
    rsi = indicators.wilder_rsi(close, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.abs(close - indicators.ema_seeded(close, 200)) / atr
    filt = (adx < config.ENTRY_FILTER_MAX_ADX) & (dist >= config.ENTRY_FILTER_MIN_DIST_ATR) & np.isfinite(dist)
    return {"close": close, "high": high, "low": low, "vol": vol,
            "adx": adx, "atr": atr, "rsi": rsi, "filter": filt}


def entries_for(d, direction, lo, hi):
    """Momentum entry on slice, using arrays that were precomputed on all data."""
    close = d["close"]
    e = np.zeros(hi - lo, dtype=bool)
    for i in range(max(lo, LOOKBACK), hi):
        roc = (close[i] - close[i - LOOKBACK]) / close[i - LOOKBACK] * 100.0
        e[i - lo] = roc >= PCT if direction == "LONG" else roc <= -PCT
    if config.ENTRY_FILTER_ENABLED:
        e &= d["filter"][lo:hi]
    return e


def stats_for(d, direction, lo, hi):
    e = entries_for(d, direction, lo, hi)
    res = exit_policy.simulate_trades(
        d["high"][lo:hi], d["low"][lo:hi], d["close"][lo:hi], e, direction == "LONG")
    s = exit_policy.trade_stats(res)
    s["entries"] = int(e.sum())
    return s


def critical_t_two_sided(n, alpha=0.05):
    """Normal approximation; exact t critical ~2.0-2.6 for n in 10..1000."""
    if n < 2:
        return float("inf")
    # z for alpha/2 with a small-sample correction
    z = 1.959963984540054
    return z * math.sqrt(1.0 + 3.0 / (8.0 * n))


def main():
    log(f"Fetching {SYMBOL} 15m, {config.DATA_DAYS}d")
    frame = fetch_ohlcv(SYMBOL)
    if frame is None:
        log("fetch failed")
        return
    d = build_arrays(frame)
    n = len(d["close"])
    log(f"bars={n}")

    holdout_start = int(n * (1 - config.HOLDOUT_FRACTION))
    fold_size = holdout_start // config.N_FOLDS

    dev_results = []
    for f in range(config.N_FOLDS):
        lo, hi = f * fold_size, (f + 1) * fold_size
        if f == config.N_FOLDS - 1:
            hi = holdout_start
        for direction in ("LONG", "SHORT"):
            s = stats_for(d, direction, lo, hi)
            dev_results.append({"fold": f, "direction": direction, **s})
            log(f"fold {f} {direction:5s}: n={s['n']:4d} mean={s['mean']:7.3f} t={s['t_stat']:6.3f}")

    hold = {}
    for direction in ("LONG", "SHORT"):
        s = stats_for(d, direction, holdout_start, n)
        hold[direction] = s
        log(f"HOLDOUT {direction:5s}: n={s['n']:4d} mean={s['mean']:7.3f} t={s['t_stat']:6.3f} "
            f"winrate={s['win_rate']:.3f} PF={s['profit_factor']:.3f}")

    naive = critical_t_two_sided(min(s["n"] for s in hold.values()))
    bonf_alpha = config.ALPHA / N_COMPARISONS
    # z for alpha/(2*comparisons): 0.000142 -> z ~3.79
    bonf_z = 3.790
    log(f"critical t: naive={naive:.3f}, Bonferroni({N_COMPARISONS} tests)={bonf_z:.3f}")

    short = hold["SHORT"]
    verdict = {
        "symbol": SYMBOL,
        "family": "momentum",
        "params": {"lookback": LOOKBACK, "pct": PCT},
        "holdout": short,
        "dev_folds": dev_results,
        "naive_critical_t": naive,
        "bonferroni_critical_t": bonf_z,
        "passes_naive": short["n"] >= config.MIN_VALIDATION_TRADES and short["t_stat"] >= naive and short["mean"] > 0,
        "passes_bonferroni": short["n"] >= config.MIN_VALIDATION_TRADES and short["t_stat"] >= bonf_z and short["mean"] > 0,
        "folds_profitable": sum(1 for r in dev_results if r["direction"] == "SHORT" and r["mean"] > 0 and r["n"] >= 2),
        "folds_total": config.N_FOLDS,
    }
    log(f"VERDICT: naive={verdict['passes_naive']}, bonferroni={verdict['passes_bonferroni']}, "
        f"profitable folds={verdict['folds_profitable']}/{verdict['folds_total']}")

    out = os.path.join(os.path.dirname(__file__), "validate_momentum_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2, default=str)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
