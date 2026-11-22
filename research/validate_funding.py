#!/usr/bin/env python3
"""Stress-test the funding-persistence candidate before any implementation.

Checks:
  1. Time-ordered sub-folds inside dev and holdout (consistency, not one spike)
  2. Per-symbol contribution (is one coin carrying everything?)
  3. LONG vs SHORT legs (is this just bull-market beta?)
  4. Threshold sensitivity (0.75-0.95) and horizon sensitivity (1-3 days)
  5. Cost sensitivity (2x fee)

Run: python research/validate_funding.py
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
import exit_policy

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "funding_ts_cache.pkl")
LEV = config.LEVERAGE
WARMUP = 0.15


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def stats(a):
    return exit_policy.trade_stats(a)


def time_bounds(data):
    all_ts = np.concatenate([np.concatenate([d["funding_ts"][:1], d["funding_ts"][-1:]])
                             for d in data.values()])
    lo, hi = int(all_ts.min()), int(all_ts.max())
    span = hi - lo
    return lo, span, int(lo + span * WARMUP), int(lo + span * 0.75), hi


def run(data, q_hi_val=0.85, q_lo_val=0.15, horizon=1, fee_mult=1.0,
        lo_t=None, hi_t=None, side="BOTH", symbol_filter=None):
    lo0, span, warm, cut, hi0 = time_bounds(data)
    a = lo0 if lo_t is None else lo_t
    b = hi0 if hi_t is None else hi_t
    fee = config.FEE_GROSS * fee_mult
    trades = []
    for sym, d in data.items():
        if symbol_filter and sym != symbol_filter:
            continue
        fts, fr, pts, c = d["funding_ts"], d["funding"], d["price_ts"], d["close"]
        prior = fr[fts < warm]
        if len(prior) < 20:
            continue
        qhi, qlo = np.quantile(prior, q_hi_val), np.quantile(prior, q_lo_val)
        for i in range(len(fts)):
            t = int(fts[i])
            if not (a <= t < b):
                continue
            j = np.searchsorted(pts, t, side="right") - 1
            if j < 0 or j + horizon >= len(c):
                continue
            gross = (c[j + horizon] - c[j]) / c[j] * LEV * 100.0
            if side in ("BOTH", "LONG") and fr[i] >= qhi:
                trades.append(gross - fee)
            elif side in ("BOTH", "SHORT") and fr[i] <= qlo:
                trades.append(-gross - fee)
    return np.asarray(trades, dtype=float)


def main():
    with open(CACHE, "rb") as fh:
        data = pickle_load = __import__("pickle").load(fh)
    lo0, span, warm, cut, hi0 = time_bounds(data)
    log(f"warmup={warm} dev=({warm},{cut}) holdout=({cut},{hi0})")

    # 1) sub-folds, both periods
    fold_res = {}
    for label, a, b, nf in (("DEV", warm, cut, 5), ("HOLDOUT", cut, hi0, 5)):
        step = (b - a) // nf
        rows = []
        for f in range(nf):
            ta = a + f * step
            tb = a + (f + 1) * step if f < nf - 1 else b
            s = stats(run(data, lo_t=ta, hi_t=tb))
            rows.append({"fold": f, **s})
            log(f"{label} fold {f}: n={s['n']:4d} mean={s['mean']:7.3f} t={s['t_stat']:6.3f}")
        fold_res[label] = rows

    # 2) per-symbol on holdout
    per_symbol = {}
    for sym in sorted(data):
        p = {"LONG": stats(run(data, lo_t=cut, hi_t=hi0, side="LONG", symbol_filter=sym)),
             "SHORT": stats(run(data, lo_t=cut, hi_t=hi0, side="SHORT", symbol_filter=sym))}
        per_symbol[sym] = p
        log(f"SYM {sym:12s} LONG n={p['LONG']['n']:3d} t={p['LONG']['t_stat']:6.2f} | "
            f"SHORT n={p['SHORT']['n']:3d} t={p['SHORT']['t_stat']:6.2f}")

    # 3) legs
    legs = {s: stats(run(data, lo_t=cut, hi_t=hi0, side=s)) for s in ("LONG", "SHORT")}
    for s, v in legs.items():
        log(f"LEG {s:5s} n={v['n']} mean={v['mean']:.3f} t={v['t_stat']:.3f} PF={v['profit_factor']:.3f}")

    # 4) sensitivity
    sens = {"threshold": {}, "horizon": {}, "fee": {}}
    for q in (0.75, 0.80, 0.85, 0.90, 0.95):
        s = stats(run(data, q_hi_val=q, q_lo_val=1-q, lo_t=cut, hi_t=hi0))
        sens["threshold"][str(q)] = s
        log(f"Q {q:.2f}: n={s['n']:4d} mean={s['mean']:7.3f} t={s['t_stat']:6.3f}")
    for h in (1, 2, 3):
        s = stats(run(data, horizon=h, lo_t=cut, hi_t=hi0))
        sens["horizon"][str(h)] = s
        log(f"H {h}d: n={s['n']:4d} mean={s['mean']:7.3f} t={s['t_stat']:6.3f}")
    for m in (1.0, 2.0, 3.0):
        s = stats(run(data, fee_mult=m, lo_t=cut, hi_t=hi0))
        sens["fee"][str(m)] = s
        log(f"FEE x{m}: n={s['n']:4d} mean={s['mean']:7.3f} t={s['t_stat']:6.3f}")

    out = {"folds": fold_res, "per_symbol": per_symbol, "legs": legs,
           "sensitivity": sens, "generated_at": datetime.now(timezone.utc).isoformat()}
    p = os.path.join(HERE, "validate_funding_results.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    log(f"saved {p}")


if __name__ == "__main__":
    main()
