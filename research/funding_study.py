#!/usr/bin/env python3
"""OKX perpetual funding rate as a crowding/contrarian signal.

Funding is not derivable from OHLCV and has an economic mechanism:
persistently positive funding means longs pay shorts (long crowding).

Timestamps are fetched and aligned explicitly with daily price bars - funding
is 8-hourly while price is daily, so index alignment would be invalid.

Three pre-registered hypotheses, 75/25 time-ordered split.

Run: python research/funding_study.py
"""
import json
import os
import pickle
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
import exit_policy

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "funding_ts_cache.pkl")
SYMS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT",
        "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "LTC/USDT",
        "ATOM/USDT", "UNI/USDT", "OP/USDT"]
LEV, FEE = config.LEVERAGE, config.FEE_GROSS


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_all():
    import ccxt
    ex = ccxt.okx({"enableRateLimit": True})
    out = {}
    end = ex.milliseconds()
    for sym in SYMS:
        m = sym.replace("/", "-") + "-SWAP"
        try:
            fr, since_f = [], end - 730 * 86400000
            for _ in range(40):
                batch = ex.fetch_funding_rate_history(m, since=since_f, limit=100)
                if not batch:
                    break
                fr.extend(batch)
                ns = batch[-1].get("timestamp")
                if ns is None or ns <= since_f:
                    break
                since_f = ns + 1
            ohlcv, since_p = [], end - 730 * 86400000
            for _ in range(10):
                batch = ex.fetch_ohlcv(m, "1d", since=since_p, limit=300)
                if not batch:
                    break
                ohlcv.extend(batch)
                if batch[-1][0] <= since_p:
                    break
                since_p = batch[-1][0] + 86400000
            # de-duplicate and sort by timestamp
            fr = list({int(r["timestamp"]): r for r in fr if r.get("timestamp")}.values())
            ohlcv = list({int(r[0]): r for r in ohlcv}.values())
            fr.sort(key=lambda r: int(r["timestamp"]))
            ohlcv.sort(key=lambda r: int(r[0]))
            if len(fr) < 50 or len(ohlcv) < 100:
                log(f"  skip {sym}: funding={len(fr)} price={len(ohlcv)}")
                continue
            out[sym] = {
                "funding_ts": np.asarray([int(r["timestamp"]) for r in fr], dtype=np.int64),
                "funding": np.asarray([float(r["fundingRate"]) for r in fr], dtype=float),
                "price_ts": np.asarray([int(r[0]) for r in ohlcv], dtype=np.int64),
                "close": np.asarray([float(r[4]) for r in ohlcv], dtype=float),
            }
            log(f"  {sym}: funding={len(fr)} price={len(ohlcv)}")
        except Exception as exc:
            log(f"  error {sym}: {type(exc).__name__}: {exc}")
    return out


def stats(a):
    return exit_policy.trade_stats(a)


WARMUP = 0.15


def run(data, hypothesis, frac_lo, frac_hi):
    """Time-ordered evaluation over funding observations in [frac_lo, frac_hi).

    Thresholds always come from the warm-up prefix [0, WARMUP), so neither
    dev nor holdout can leak into their own threshold.
    """
    all_ts = np.sort(np.unique(np.concatenate(
        [np.asarray([d["funding_ts"].min() for d in data.values()]),
         np.asarray([d["funding_ts"].max() for d in data.values()])])))
    global_lo = int(all_ts.min()); global_hi = int(all_ts.max())
    span = global_hi - global_lo
    warm_cut = int(global_lo + span * WARMUP)
    cut_a = int(global_lo + span * frac_lo)
    cut_b = int(global_lo + span * frac_hi)
    trades = []
    for sym, d in data.items():
        fts, fr = d["funding_ts"], d["funding"]
        pts, c = d["price_ts"], d["close"]
        # threshold from warm-up prefix only, never from the evaluation window
        prior = fr[fts < warm_cut]
        if len(prior) < 20:
            continue
        q_hi = float(np.quantile(prior, 0.85))
        q_lo = float(np.quantile(prior, 0.15))
        for i in range(len(fts)):
            t = int(fts[i])
            if not (cut_a <= t < cut_b):
                continue
            j = np.searchsorted(pts, t, side="right") - 1
            if j < 0 or j + 1 >= len(c):
                continue
            gross = (c[j+1] - c[j]) / c[j] * LEV * 100.0
            r = fr[i]
            if hypothesis == "contrarian" and r >= q_hi:
                trades.append(-gross - FEE)
            elif hypothesis == "contrarian" and r <= q_lo:
                trades.append(gross - FEE)
            elif hypothesis == "persistence" and r >= q_hi:
                trades.append(gross - FEE)
            elif hypothesis == "persistence" and r <= q_lo:
                trades.append(-gross - FEE)
    return np.asarray(trades, dtype=float)


def main():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            data = pickle.load(fh)
        log(f"loaded cache: {len(data)} symbols")
    else:
        data = fetch_all()
        with open(CACHE, "wb") as fh:
            pickle.dump(data, fh)
    if len(data) < 4:
        log("insufficient data")
        return

    hyps = ("contrarian", "persistence")
    scores = {}
    for h in hyps:
        d = stats(run(data, h, WARMUP, 0.75))
        scores[h] = d
        log(f"DEV {h:12s} n={d['n']:5d} mean={d['mean']:7.3f} t={d['t_stat']:7.3f} PF={d['profit_factor']:.3f}")
    best = max(scores.items(), key=lambda kv: kv[1]["t_stat"])[0]
    hold = stats(run(data, best, 0.75, 1.0))
    log(f"FROZEN {best} dev t={scores[best]['t_stat']:.3f}")
    log(f"HOLDOUT n={hold['n']} mean={hold['mean']:.3f} t={hold['t_stat']:.3f} PF={hold['profit_factor']:.3f}")
    out = {"symbols": len(data), "scores": scores, "winner": best, "holdout": hold,
           "passes_naive": bool(hold["n"] >= 40 and hold["mean"] > 0 and hold["t_stat"] >= 1.96),
           "generated_at": datetime.now(timezone.utc).isoformat()}
    p = os.path.join(HERE, "funding_results.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    log(f"saved {p}")


if __name__ == "__main__":
    main()
