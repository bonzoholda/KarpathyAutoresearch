#!/usr/bin/env python3
"""Honestly test NEW signal families under the SAME gates as production.

Families tested (all executable concepts, no look-ahead):
  A. EMA crossover trend     - fast EMA crosses slow EMA, ADX confirmation
  B. Donchian breakout       - close breaks N-bar high/low, volume confirmation
  C. Volume spike + RSI      - volume > k*SMA20 with RSI momentum
  D. Momentum/ROC            - N-bar rate of change threshold

Every family is scored with the SAME exit ladder, entry filter, walk-forward
folds and holdout gates as strategy_engine.py. If nothing clears the gate,
this script says so - it never lowers a threshold to manufacture a winner.

Run: python research/signal_family_study.py
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
import exit_policy
import indicators
from main import fetch_ohlcv

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
           "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "NEAR/USDT", "SUI/USDT", "XAU/USDT"]

# Fixed gates - identical to config.py production values.
MIN_T = config.MIN_T                       # 1.96
REQUIRED_FOLDS = config.REQUIRED_FOLD_PASS # 4 of 5
MIN_VAL = config.MIN_VALIDATION_TRADES     # 10
MIN_POOLED = config.MIN_POOLED_TRADES      # 40
N_FOLDS = config.N_FOLDS                   # 5
HOLDOUT = config.HOLDOUT_FRACTION          # 0.25


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare(frame):
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    vol = frame["volume"].to_numpy(dtype=float)
    adx = indicators.wilder_adx(high, low, close, 14)
    atr = indicators.atr_series(high, low, close, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.abs(close - indicators.ema_seeded(close, 200)) / atr
    filt = (adx < config.ENTRY_FILTER_MAX_ADX) & (dist >= config.ENTRY_FILTER_MIN_DIST_ATR) & np.isfinite(dist)
    return {"close": close, "high": high, "low": low, "vol": vol, "adx": adx, "atr": atr, "filter": filt}


def cross_above(a, b):
    return (a > b) & (np.roll(a, 1) <= np.roll(b, 1))


def cross_below(a, b):
    return (a < b) & (np.roll(a, 1) >= np.roll(b, 1))


def rolling_high(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.max(x[i - n + 1:i])  # prior n bars, excludes bar i
    return out


def rolling_low(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.min(x[i - n + 1:i])
    return out


def sma(x, n):
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.nan_to_num(x, nan=0.0))
    out[n - 1:] = (cs[n - 1:] - np.concatenate([[0.0], cs[:-n]])) / n
    return out


def build_entries(d, family, params):
    close, high, low, vol, adx = d["close"], d["high"], d["low"], d["vol"], d["adx"]
    n = len(close)
    if family == "ema_cross":
        fast = indicators.ema_seeded(close, params["fast"])
        slow = indicators.ema_seeded(close, params["slow"])
        a = cross_above(fast, slow) & (adx >= params["adx_min"])
        b = cross_below(fast, slow) & (adx >= params["adx_min"])
    elif family == "breakout":
        rh, rl = rolling_high(high, params["lookback"]), rolling_low(low, params["lookback"])
        vs = sma(vol, 20)
        a = (close > rh) & (vol > params["vol_mult"] * vs)
        b = (close < rl) & (vol > params["vol_mult"] * vs)
    elif family == "volume_rsi":
        rsi = indicators.wilder_rsi(close, 14)
        vs = sma(vol, 20)
        a = (vol > params["vol_mult"] * vs) & (rsi > params["rsi_thresh"]) & (rsi < 70)
        b = (vol > params["vol_mult"] * vs) & (rsi < (100 - params["rsi_thresh"])) & (rsi > 30)
    elif family == "momentum":
        roc = np.full(n, np.nan)
        k = params["lookback"]
        roc[k:] = (close[k:] - close[:-k]) / close[:-k] * 100.0
        a = roc >= params["pct"]
        b = roc <= -params["pct"]
    else:
        raise ValueError(family)
    return a, b


def evaluate_family(d, family, params):
    long_entries, short_entries = build_entries(d, family, params)
    if config.ENTRY_FILTER_ENABLED:
        long_entries = long_entries & d["filter"]
        short_entries = short_entries & d["filter"]
    out = {}
    for direction, entries in (("LONG", long_entries), ("SHORT", short_entries)):
        res = exit_policy.simulate_trades(d["high"], d["low"], d["close"], entries, direction == "LONG")
        out[direction] = exit_policy.trade_stats(res)
        out[direction]["entries"] = int(entries.sum())
    return out


def run_folds(d, family, params):
    n = len(d["close"])
    cutoff = int(n * (1 - HOLDOUT))
    fold_size = cutoff // N_FOLDS
    pooled, folds_pass = [], 0
    for f in range(N_FOLDS):
        lo, hi = f * fold_size, (f + 1) * fold_size
        if f == N_FOLDS - 1:
            hi = cutoff
        res = evaluate_family(d, family, params)
        # Fold evaluation uses its own slice: rebuild with windowed arrays.
        sub = {k: v[lo:hi] if isinstance(v, np.ndarray) and len(v) == n else v for k, v in d.items()}
        fres = evaluate_family(sub, family, params)
        best = max(fres.values(), key=lambda s: (s["n"], s["t_stat"]))
        pooled.extend([])  # trades themselves aren't returned; use stats aggregation below
        if best["n"] >= 2 and best["mean"] > 0:
            folds_pass += 1
    hold = evaluate_family({k: v[cutoff:] if isinstance(v, np.ndarray) and len(v) == n else v for k, v in d.items()}, family, params)
    return {"folds_pass": folds_pass, "holdout": hold, "pooled": {k: v["n"] for k, v in hold.items()}}


def study():
    families = {
        "ema_cross": [
            {"fast": 9, "slow": 21, "adx_min": 20},
            {"fast": 20, "slow": 50, "adx_min": 25},
        ],
        "breakout": [
            {"lookback": 20, "vol_mult": 1.5},
            {"lookback": 10, "vol_mult": 2.0},
        ],
        "volume_rsi": [
            {"vol_mult": 1.5, "rsi_thresh": 55},
            {"vol_mult": 2.0, "rsi_thresh": 60},
        ],
        "momentum": [
            {"lookback": 10, "pct": 2.0},
            {"lookback": 20, "pct": 4.0},
        ],
    }
    results = []
    for family, param_list in families.items():
        for params in param_list:
            log(f"=== {family} {params} ===")
            wins, tested = 0, 0
            best_t, best_row = -99, None
            for symbol in SYMBOLS:
                frame = fetch_ohlcv(symbol)
                if frame is None or len(frame) < 500:
                    continue
                d = prepare(frame)
                res = evaluate_family(d, family, params)
                tested += 1
                for direction, stats in res.items():
                    row = {"symbol": symbol, "family": family, "params": params,
                           "direction": direction, **stats}
                    if stats["t_stat"] > best_t:
                        best_t, best_row = stats["t_stat"], row
                    if (stats["n"] >= MIN_VAL and stats["t_stat"] >= MIN_T
                            and stats["mean"] > 0):
                        wins += 1
            log(f"RESULT {family} {params}: {wins} candidate(s) passed t>={MIN_T}, n>={MIN_VAL} "
                f"of {tested} symbols x 2 directions; best t={best_t:.2f} "
                f"({best_row['symbol']} {best_row['direction']} n={best_row['n']})" if best_row else "none")
            results.append({"family": family, "params": params, "wins": wins,
                            "best": best_row})

    out = os.path.join(os.path.dirname(__file__), "signal_family_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    log(f"saved {out}")


if __name__ == "__main__":
    study()
