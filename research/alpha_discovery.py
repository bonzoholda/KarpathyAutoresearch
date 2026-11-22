#!/usr/bin/env python3
"""Broad, honest search for a signal with real out-of-sample edge.

Design principles (learned from the failed single-symbol sweeps):
  - Portfolio-level evaluation pools trades across symbols. One hypothesis
    produces ONE t-stat, so Bonferroni is over hypotheses (not 176 symbol
    variants). A cross-sectional anomaly is a portfolio phenomenon anyway.
  - Hypotheses are PRE-REGISTERED below. No result is allowed to generate a
    new hypothesis for the same holdout.
  - Indicators are computed once on the full series, then sliced.
  - 75% development / 25% time-ordered holdout. The holdout is touched
    exactly once, after the dev winner is frozen.
  - Every family is scored with an explicit, realistic exit and the same
    fee/slippage/leverage units as production.

Run: python research/alpha_discovery.py
"""
import json
import math
import os
import pickle
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
import exit_policy
from main import fetch_ohlcv

# Broader liquid OKX USDT-margined swaps: original 11 + higher-volatility names.
UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "NEAR/USDT", "SUI/USDT", "XAU/USDT",
    "LINK/USDT", "DOT/USDT", "LTC/USDT", "ATOM/USDT", "UNI/USDT",
    "APT/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT", "TAO/USDT",
]
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpha_1h_cache.pkl")
TIMEFRAME_HOURS = 1
HOLD_BARS = 48            # 48h maximum hold for the 1h-native strategies
FEE = config.FEE_GROSS
SLIP = config.SLIPPAGE
LEV = config.LEVERAGE


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def ema(x, n):
    a = 2.0 / (n + 1.0)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def atr(high, low, close, n=14):
    pc = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - pc), np.abs(low - pc)])
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return out
    out[n] = np.nanmean(tr[1:n + 1])
    for i in range(n + 1, len(close)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def rsi(close, n=14):
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = np.full(len(close), np.nan)
    ad = np.full(len(close), np.nan)
    if len(close) <= n:
        return au
    au[n] = np.mean(up[1:n + 1]); ad[n] = np.mean(dn[1:n + 1])
    for i in range(n + 1, len(close)):
        au[i] = (au[i - 1] * (n - 1) + up[i]) / n
        ad[i] = (ad[i - 1] * (n - 1) + dn[i]) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = au / ad
        out = 100 - 100 / (1 + rs)
    out[ad == 0] = 100.0
    out[:n] = np.nan
    return out


def simulate(entries, close, high, low, is_long, stop_mult=None, hold=HOLD_BARS, atr_arr=None):
    """Enter at signal-bar close; exit by fixed hold or ATR trailing stop."""
    out = []
    n = len(close)
    for i in np.nonzero(entries)[0]:
        k = int(i)
        if k + 1 >= n:
            break
        entry = close[k]
        if entry <= 0:
            continue
        stop_px = None
        peak = entry
        risk = stop_mult * atr_arr[k] if (stop_mult is not None and atr_arr is not None) else 0.0
        if stop_mult is not None and (atr_arr is None or not np.isfinite(risk) or risk <= 0):
            continue
        if stop_mult is not None:
            stop_px = entry - risk if is_long else entry + risk
        exit_px = None
        for bar in range(k + 1, min(n, k + 1 + hold)):
            if stop_px is not None:
                if is_long and low[bar] <= stop_px:
                    exit_px = stop_px * (1 - SLIP); break
                if (not is_long) and high[bar] >= stop_px:
                    exit_px = stop_px * (1 + SLIP); break
                if is_long and high[bar] > peak:
                    peak = high[bar]; stop_px = max(stop_px, peak - risk)
                if (not is_long) and low[bar] < peak:
                    peak = low[bar]; stop_px = min(stop_px, peak + risk)
        if exit_px is None:
            j = min(n - 1, k + hold)
            exit_px = close[j]
        gross = ((exit_px - entry) if is_long else (entry - exit_px)) / entry
        out.append(gross * LEV * 100.0 - FEE)
    return np.asarray(out, dtype=float)


def stats(arr):
    return exit_policy.trade_stats(arr)


def fetch_all():
    frames = {}
    for sym in UNIVERSE:
        try:
            f = fetch_ohlcv(sym)
        except Exception as exc:
            log(f"  fetch error {sym}: {type(exc).__name__}: {exc}")
            continue
        if f is None or len(f) < 500:
            log(f"  skip {sym}: insufficient data")
            continue
        frames[sym] = f
        log(f"  {sym}: {len(f)} bars")
    return frames


def build(frames):
    data = {}
    for sym, f in frames.items():
        c = f["close"].to_numpy(float); h = f["high"].to_numpy(float); l = f["low"].to_numpy(float)
        data[sym] = {"close": c, "high": h, "low": l,
                     "rsi": rsi(c, 14), "ema50": ema(c, 50), "ema200": ema(c, 200),
                     "roc": {k: np.concatenate([np.full(k, np.nan), c[k:] / c[:-k] - 1]) for k in (4, 8, 24)},
                     "atr": atr(h, l, c, 14)}
    return data


def zscore(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        w = x[i - n:i]
        s = w.std()
        if s > 0:
            out[i] = (x[i] - w.mean()) / s
    return out


# ---------------- pre-registered hypotheses ----------------
def h_cross_sectional(data, lookback, topk, rebalance, lo, hi, direction="BOTH"):
    """Rank by past return; long strongest / short weakest at each rebalance."""
    syms = list(data)
    n = len(data[syms[0]]["close"])
    trades = []
    for start in range(lo + lookback, hi - 1, rebalance):
        scores = {}
        for s in syms:
            r = data[s]["roc"][lookback][start]
            if np.isfinite(r):
                scores[s] = r
        if len(scores) < topk * 2 + 2:
            continue
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        picks = []
        if direction in ("BOTH", "LONG"):
            picks += [(s, True) for s, _ in ranked[:topk]]
        if direction in ("BOTH", "SHORT"):
            picks += [(s, False) for s, _ in ranked[-topk:]]
        for s, is_long in picks:
            entry = data[s]["close"][start]
            exit_i = min(start + rebalance, n - 1)
            gross = ((data[s]["close"][exit_i] - entry) if is_long else (entry - data[s]["close"][exit_i])) / entry
            trades.append(gross * LEV * 100.0 - FEE - SLIP * LEV * 100.0)
    return np.asarray(trades)


def h_breakout(data, look, mult, lo, hi):
    """Prior-bar Donchian breakout with ATR trailing stop."""
    trades = []
    for s, d in data.items():
        c, h, l, a = d["close"], d["high"], d["low"], d["atr"]
        e_long = np.zeros(hi - lo, bool); e_short = np.zeros(hi - lo, bool)
        for i in range(lo + look, hi):
            ph = np.max(h[i - look:i]); pl = np.min(l[i - look:i])
            e_long[i - lo] = c[i] > ph; e_short[i - lo] = c[i] < pl
        for e, is_long in ((e_long, True), (e_short, False)):
            trades.extend(simulate(e, c[lo:hi], h[lo:hi], l[lo:hi], is_long,
                                   stop_mult=mult, atr_arr=a[lo:hi]))
    return np.asarray(trades)


def h_rsi_reversal(data, lower, upper, lo, hi):
    """Executable today with zero Go changes: one threshold pair, all symbols."""
    trades = []
    for s, d in data.items():
        c, h, l, r = d["close"], d["high"], d["low"], d["rsi"]
        el = np.zeros(hi - lo, bool); es = np.zeros(hi - lo, bool)
        for i in range(lo + 1, hi):
            if r[i - 1] <= lower and r[i] >= r[i - 1] + 2.0:
                el[i - lo] = True
            if r[i - 1] >= upper and r[i] <= r[i - 1] - 2.0:
                es[i - lo] = True
        trades.extend(simulate(el, c[lo:hi], h[lo:hi], l[lo:hi], True, stop_mult=None, hold=32))
        trades.extend(simulate(es, c[lo:hi], h[lo:hi], l[lo:hi], False, stop_mult=None, hold=32))
    return np.asarray(trades)


def h_zscore_revert(data, window, zth, lo, hi):
    trades = []
    for s, d in data.items():
        c, h, l = d["close"], d["high"], d["low"]
        z = zscore(c, window)
        el = np.zeros(hi - lo, bool); es = np.zeros(hi - lo, bool)
        for i in range(lo + window, hi):
            if z[i] <= -zth: el[i - lo] = True
            if z[i] >= zth: es[i - lo] = True
        trades.extend(simulate(el, c[lo:hi], h[lo:hi], l[lo:hi], True,
                               stop_mult=2.0, atr_arr=d["atr"][lo:hi]))
        trades.extend(simulate(es, c[lo:hi], h[lo:hi], l[lo:hi], False,
                               stop_mult=2.0, atr_arr=d["atr"][lo:hi]))
    return np.asarray(trades)


def fetch_frames():
    log("Fetching 1h history, 365 days")
    import ccxt
    frames = {}
    for sym in UNIVERSE:
        try:
            ex = ccxt.okx({"enableRateLimit": True})
            m = sym.replace("/", "-") + "-SWAP"
            now = ex.milliseconds()
            since = now - 365 * 86400 * 1000
            rows = []
            guard = 0
            while since < now and guard < 200:
                guard += 1
                b = ex.fetch_ohlcv(m, timeframe="1h", since=since, limit=300)
                if not b:
                    break
                rows.extend(b)
                if b[-1][0] <= since:
                    break
                since = b[-1][0] + 3600000
            if len(rows) < 500:
                log(f"  skip {sym}: {len(rows)} bars")
                continue
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
            for col in ("high", "low", "close", "volume"):
                df[col] = df[col].astype(float)
            frames[sym] = df
            log(f"  {sym}: {len(df)} bars")
        except Exception as exc:
            log(f"  fetch error {sym}: {type(exc).__name__}: {exc}")
    return frames


def main():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as fh:
            frames = pickle.load(fh)
        log(f"Loaded cached 1h data: {len(frames)} symbols")
    else:
        frames = fetch_frames()
        with open(CACHE_PATH, "wb") as fh:
            pickle.dump(frames, fh)
        log(f"Cached 1h data to {CACHE_PATH}")

    if len(frames) < 5:
        log("insufficient data"); return
    data = build(frames)
    n = len(next(iter(data.values()))["close"])
    cut = int(n * 0.75)
    log(f"symbols={len(data)} bars={n} dev={cut} holdout={n-cut}")

    hypotheses = {
        "cs_mom_8_3_24": lambda lo, hi: h_cross_sectional(data, 8, 3, 24, lo, hi),
        "cs_mom_24_3_24": lambda lo, hi: h_cross_sectional(data, 24, 3, 24, lo, hi),
        "cs_mom_24_5_72": lambda lo, hi: h_cross_sectional(data, 24, 5, 72, lo, hi),
        "breakout20_2atr": lambda lo, hi: h_breakout(data, 20, 2.0, lo, hi),
        "breakout55_3atr": lambda lo, hi: h_breakout(data, 55, 3.0, lo, hi),
        "rsi_35_65": lambda lo, hi: h_rsi_reversal(data, 35, 65, lo, hi),
        "rsi_30_70": lambda lo, hi: h_rsi_reversal(data, 30, 70, lo, hi),
        "zscore20_2": lambda lo, hi: h_zscore_revert(data, 20, 2.0, lo, hi),
    }
    n_tests = len(hypotheses)
    dev_scores = {}
    for name, fn in hypotheses.items():
        d = stats(fn(0, cut))
        dev_scores[name] = d
        log(f"DEV {name:20s} n={d['n']:5d} mean={d['mean']:7.3f} t={d['t_stat']:7.3f} PF={d['profit_factor']:.3f}")

    ranked = sorted(dev_scores.items(), key=lambda kv: kv[1]["t_stat"], reverse=True)
    best_name, best_dev = ranked[0]
    log(f"FROZEN dev winner: {best_name} (dev t={best_dev['t_stat']:.3f})")

    hold = stats(hypotheses[best_name](cut, n))
    log(f"HOLDOUT {best_name}: n={hold['n']} mean={hold['mean']:.3f} t={hold['t_stat']:.3f} "
        f"PF={hold['profit_factor']:.3f} win={hold['win_rate']:.3f}")

    # Bonferroni z for the pre-registered hypothesis count
    p = 1 - 0.05 / (2 * n_tests)
    # inverse normal (Acklam, simplified)
    def inv_norm(pv):
        a=[-3.969683028665376e1,2.209460984245205e2,-2.759285104469687e2,1.383577518672690e2,-3.066479806614716e1,2.506628277459239]
        b=[-5.447609879822406e1,1.615858368580409e2,-1.556989798598866e2,6.680131188771972e1,-1.328068155288572e1]
        q=pv-0.5; r=q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    bonf = inv_norm(p)
    naive = inv_norm(0.975)
    passed = hold["n"] >= 40 and hold["t_stat"] >= bonf and hold["mean"] > 0
    passed_naive = hold["n"] >= 40 and hold["t_stat"] >= naive and hold["mean"] > 0

    out = {"symbols": len(data), "bars": n, "cut": cut,
           "n_hypotheses": n_tests, "bonferroni_t": bonf, "naive_t": naive,
           "dev_scores": dev_scores, "frozen": best_name,
           "holdout": hold, "passes_naive": passed_naive, "passes_bonferroni": passed,
           "generated_at": datetime.now(timezone.utc).isoformat()}
    path = os.path.join(os.path.dirname(__file__), "alpha_discovery_results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    log(f"VERDICT: naive={passed_naive} bonferroni={passed} saved {path}")


if __name__ == "__main__":
    main()
