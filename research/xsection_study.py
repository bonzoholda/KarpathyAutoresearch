#!/usr/bin/env python3
"""Second-stage search: cross-sectional rank strategies on a 21-symbol basket.

Motivation from alpha_discovery.py: the best family (cross-sectional momentum)
had positive GROSS edge but was erased by 0.36% round-trip cost. This stage
tests lower-turnover rebalances, reversal, and volatility-normalised ranking.

Pre-registered grid: 3 lookbacks x 2 rebalances x 2 basket sizes x 2 modes
x 2 score normalisations = 48 variants. Selection uses DEV ONLY, requires
at least 4 of 5 positive walk-forward folds, then the single winner is scored
once on the untouched holdout. Bonferroni over 48 variants.

Run: python research/xsection_study.py
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
CACHE = os.path.join(HERE, "alpha_1h_cache.pkl")
LEV, FEE, SLIP = config.LEVERAGE, config.FEE_GROSS, config.SLIPPAGE

LOOKBACKS = (8, 24, 72)
REBALANCES = (24, 72)
TOPKS = (3, 5)
MODES = ("momentum", "reversal")
NORMS = ("raw", "voladj")
N_FOLDS = 5
HOLDOUT_FRAC = 0.25


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def atr_pct(close, high, low, n=14):
    pc = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - pc), np.abs(low - pc)])
    out = np.full(len(close), np.nan)
    out[n] = np.nanmean(tr[1:n + 1])
    for i in range(n + 1, len(close)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out / close


def build():
    with open(CACHE, "rb") as fh:
        frames = pickle.load(fh)
    data = {}
    for sym, df in frames.items():
        c = df["close"].to_numpy(float); h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
        data[sym] = {"close": c, "volpct": atr_pct(c, h, l, 14),
                     "roc": {k: np.concatenate([np.full(k, np.nan), c[k:] / c[:-k] - 1]) for k in LOOKBACKS}}
    return data


def run_variant(data, look, rebal, topk, mode, norm, lo, hi):
    syms = sorted(data)
    trades = []
    for start in range(lo + look, hi - 1, rebal):
        scores = {}
        for s in syms:
            r = data[s]["roc"][look][start]
            v = data[s]["volpct"][start] if norm == "voladj" else 1.0
            if np.isfinite(r) and np.isfinite(v) and v > 0:
                scores[s] = r / v
        need = topk * 2 + 2
        if len(scores) < need:
            continue
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        longs = ranked[-topk:] if mode == "reversal" else ranked[:topk]
        shorts = ranked[:topk] if mode == "reversal" else ranked[-topk:]
        for s, _ in longs:
            e = data[s]["close"][start]
            x = data[s]["close"][min(start + rebal, len(data[s]["close"]) - 1)]
            trades.append((x - e) / e * LEV * 100 - FEE - SLIP * LEV * 100)
        for s, _ in shorts:
            e = data[s]["close"][start]; x = data[s]["close"][min(start + rebal, len(data[s]["close"]) - 1)]
            trades.append((e - x) / e * LEV * 100 - FEE - SLIP * LEV * 100)
    return np.asarray(trades, dtype=float)


def stats(a):
    return exit_policy.trade_stats(a)


def inv_norm(p):
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def main():
    data = build()
    n = len(next(iter(data.values()))["close"])
    cut = int(n * (1 - HOLDOUT_FRAC))
    fold = cut // N_FOLDS
    log(f"symbols={len(data)} bars={n} dev={cut} holdout={n-cut}")

    variants = [(lk, rb, tk, md, nm)
                for lk in LOOKBACKS for rb in REBALANCES for tk in TOPKS
                for md in MODES for nm in NORMS]
    n_tests = len(variants)
    log(f"pre-registered variants={n_tests}")

    scored = []
    for v in variants:
        lk, rb, tk, md, nm = v
        name = f"lk{lk}_rb{rb}_k{tk}_{md}_{nm}"
        dev = stats(run_variant(data, lk, rb, tk, md, nm, 0, cut))
        fold_positive = 0
        for f in range(N_FOLDS):
            lo, hi = f * fold, (f + 1) * fold if f < N_FOLDS - 1 else cut
            r = stats(run_variant(data, lk, rb, tk, md, nm, lo, hi))
            if r["mean"] > 0 and r["n"] >= 4:
                fold_positive += 1
        scored.append({"name": name, "params": {"lookback": lk, "rebalance": rb,
                       "topk": tk, "mode": md, "norm": nm},
                       "dev": dev, "folds_positive": fold_positive})

    eligible = [s for s in scored if s["folds_positive"] >= 4 and s["dev"]["n"] >= 40]
    pool = eligible if eligible else scored
    pool.sort(key=lambda s: s["dev"]["t_stat"], reverse=True)
    winner = pool[0]
    log(f"FROZEN: {winner['name']} dev t={winner['dev']['t_stat']:.3f} "
        f"folds={winner['folds_positive']}/{N_FOLDS} eligible={len(eligible)}")

    lk, rb, tk = (winner["params"][k] for k in ("lookback", "rebalance", "topk"))
    md, nm = winner["params"]["mode"], winner["params"]["norm"]
    hold = stats(run_variant(data, lk, rb, tk, md, nm, cut, n))
    log(f"HOLDOUT: n={hold['n']} mean={hold['mean']:.3f} t={hold['t_stat']:.3f} "
        f"PF={hold['profit_factor']:.3f} win={hold['win_rate']:.3f}")

    bonf = inv_norm(1 - 0.05 / (2 * n_tests))
    naive = inv_norm(0.975)
    passed = hold["n"] >= 40 and hold["t_stat"] >= bonf and hold["mean"] > 0
    passed_naive = hold["n"] >= 40 and hold["t_stat"] >= naive and hold["mean"] > 0
    log(f"critical t: naive={naive:.3f} bonferroni({n_tests})={bonf:.3f}")
    log(f"VERDICT: naive={passed_naive} bonferroni={passed}")

    out = {"variants": n_tests, "critical_t": {"naive": naive, "bonferroni": bonf},
           "eligible": len(eligible), "winner": winner, "holdout": hold,
           "passes_naive": passed_naive, "passes_bonferroni": passed,
           "all_scores": sorted(scored, key=lambda s: s["dev"]["t_stat"], reverse=True)[:12],
           "generated_at": datetime.now(timezone.utc).isoformat()}
    p = os.path.join(HERE, "xsection_results.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    log(f"saved {p}")


if __name__ == "__main__":
    main()
