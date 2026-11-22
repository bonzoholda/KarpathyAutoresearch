#!/usr/bin/env python3
"""Regime-gated cross-sectional momentum on multi-year daily bars.

The ungated 30d cross-sectional momentum won development (t=2.17, 5/5 folds)
but failed holdout (t=-0.20). This study asks whether a market-regime gate,
chosen on economic grounds rather than fitted to the holdout, restores it.

Six pre-registered gate variants. Dev selection only; holdout touched once.
Cumulative-search disclosure: this is the 6th research program on the same
data, so the honest critical t is well above 1.96 - reported explicitly.

Run: python research/regime_xsection_study.py
"""
import json
import math
import os
import pickle
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
import exit_policy

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "daily_5y_cache.pkl")
LOOK, REBAL, TOPK = 30, 7, 3
LEV, FEE, SLIP = config.LEVERAGE, config.FEE_GROSS, config.SLIPPAGE


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def ema(x, n):
    a = 2.0/(n+1.0); out = np.full(len(x), np.nan)
    out[n-1] = np.mean(x[:n])
    for i in range(n, len(x)): out[i] = a*x[i]+(1-a)*out[i-1]
    return out


def build():
    with open(CACHE, "rb") as fh: frames = pickle.load(fh)
    data = {}
    for sym, f in frames.items():
        c = f["close"].to_numpy(float)
        data[sym] = {"close": c, "ema200": ema(c, 200),
                     "roc": {30: np.concatenate([np.full(30, np.nan), c[30:]/c[:-30]-1])}}
        # per-symbol own trend regime
        data[sym]["own_above"] = c > data[sym]["ema200"]
    n = min(len(d["close"]) for d in data.values())
    return {s: {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
            for s, d in data.items()}, n


def run(data, n, gate, lo, hi):
    syms = sorted(data)
    btc = data.get("BTC/USDT")
    breadth = np.zeros(n)
    for s in syms:
        breadth += np.where(data[s]["own_above"], 1.0, 0.0)
    breadth /= len(syms)
    trades = []
    for start in range(lo + LOOK, hi - 1, REBAL):
        ok = True
        if gate == "btc_above":
            ok = bool(btc["own_above"][start])
        elif gate == "btc_below":
            ok = not bool(btc["own_above"][start])
        elif gate == "breadth_high":
            ok = breadth[start] >= 0.60
        elif gate == "breadth_low":
            ok = breadth[start] <= 0.40
        elif gate == "breadth_extreme":
            ok = breadth[start] >= 0.70 or breadth[start] <= 0.30
        if not ok:
            continue
        sc = {s: data[s]["roc"][30][start] for s in syms if np.isfinite(data[s]["roc"][30][start])}
        if len(sc) < TOPK*2+2:
            continue
        rk = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)
        for s, _ in rk[:TOPK]:
            e = data[s]["close"][start]; x = data[s]["close"][min(start+REBAL, n-1)]
            trades.append((x-e)/e*LEV*100-FEE-SLIP*LEV*100)
        for s, _ in rk[-TOPK:]:
            e = data[s]["close"][start]; x = data[s]["close"][min(start+REBAL, n-1)]
            trades.append((e-x)/e*LEV*100-FEE-SLIP*LEV*100)
    return np.asarray(trades)


def stats(a): return exit_policy.trade_stats(a)


def main():
    data, n = build()
    cut = int(n*0.75)
    log(f"bars={n} dev={cut} holdout={n-cut}")
    gates = ("none", "btc_above", "btc_below", "breadth_high",
             "breadth_low", "breadth_extreme")
    fold = cut//5
    scores = []
    for g in gates:
        d = stats(run(data, n, g, 0, cut))
        fp = 0
        for f in range(5):
            lo, hi = f*fold, ((f+1)*fold if f < 4 else cut)
            r = stats(run(data, n, g, lo, hi))
            if r["mean"] > 0 and r["n"] >= 3: fp += 1
        scores.append({"gate": g, "dev": d, "folds": fp})
        log(f"DEV {g:16s} n={d['n']:5d} mean={d['mean']:7.3f} t={d['t_stat']:7.3f} folds={fp}/5")
    eligible = [s for s in scores if s["folds"] >= 4 and s["dev"]["n"] >= 40]
    pool = eligible or scores
    pool.sort(key=lambda s: s["dev"]["t_stat"], reverse=True)
    w = pool[0]
    hold = stats(run(data, n, w["gate"], cut, n))
    log(f"FROZEN {w['gate']} dev t={w['dev']['t_stat']:.3f}")
    log(f"HOLDOUT n={hold['n']} mean={hold['mean']:.3f} t={hold['t_stat']:.3f} PF={hold['profit_factor']:.3f}")

    total_programs = 6
    total_variants = 8+48+3+12+8+6
    bonf_z = 3.5  # conservative for ~85 cumulative comparisons
    passed = hold["n"] >= 40 and hold["mean"] > 0 and hold["t_stat"] >= 1.96
    passed_bonf = passed and hold["t_stat"] >= bonf_z
    out = {"programs": total_programs, "cumulative_variants": total_variants,
           "scores": scores, "winner": w, "holdout": hold,
           "naive_critical_t": 1.96, "cumulative_bonferroni_t": bonf_z,
           "passes_naive": bool(passed), "passes_cumulative_bonferroni": bool(passed_bonf),
           "generated_at": datetime.now(timezone.utc).isoformat()}
    p = os.path.join(HERE, "regime_xsection_results.json")
    with open(p, "w", encoding="utf-8") as fh: json.dump(out, fh, indent=2, default=str)
    log(f"VERDICT naive={passed} cumulative_bonferroni={passed_bonf}; saved {p}")


if __name__ == "__main__": main()
