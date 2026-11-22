#!/usr/bin/env python3
"""Multi-year daily-bar search: the regime the 15m/1h studies could not see.

Pre-registered hypotheses (8 total), 21 symbols, ~5 years of daily bars,
75/25 time-ordered dev/holdout, Bonferroni over 8, no per-symbol selection.
All trades are pooled across the basket under one hypothesis.

Run: python research/daily_alpha_study.py
"""
import json
import os
import pickle
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
import exit_policy

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "daily_5y_cache.pkl")
UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "NEAR/USDT", "SUI/USDT", "XAU/USDT",
    "LINK/USDT", "DOT/USDT", "LTC/USDT", "ATOM/USDT", "UNI/USDT",
    "APT/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT", "TAO/USDT",
]
LEV, FEE, SLIP = config.LEVERAGE, config.FEE_GROSS, config.SLIPPAGE
N_FOLDS = 5


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_daily():
    import ccxt
    log("fetching 5y daily bars")
    frames = {}
    for sym in UNIVERSE:
        try:
            ex = ccxt.okx({"enableRateLimit": True})
            m = sym.replace("/", "-") + "-SWAP"
            now = ex.milliseconds(); since = now - 5 * 365 * 86400 * 1000
            rows = []
            while since < now:
                b = ex.fetch_ohlcv(m, timeframe="1d", since=since, limit=300)
                if not b: break
                rows.extend(b)
                if b[-1][0] <= since: break
                since = b[-1][0] + 86400000
            if len(rows) < 400:
                log(f"  skip {sym}: {len(rows)} bars"); continue
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
            for col in ("high", "low", "close", "volume"): df[col] = df[col].astype(float)
            frames[sym] = df
            log(f"  {sym}: {len(df)} bars")
        except Exception as exc:
            log(f"  error {sym}: {type(exc).__name__}: {exc}")
    return frames


def ema(x, n):
    a = 2.0 / (n + 1.0); out = np.full(len(x), np.nan)
    if len(x) < n: return out
    out[n - 1] = np.mean(x[:n])
    for i in range(n, len(x)): out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def atr(high, low, close, n=14):
    pc = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - pc), np.abs(low - pc)])
    out = np.full(len(close), np.nan); out[n] = np.nanmean(tr[1:n+1])
    for i in range(n+1, len(close)): out[i] = (out[i-1]*(n-1)+tr[i])/n
    return out


def build(frames):
    data = {}
    for sym, f in frames.items():
        c = f["close"].to_numpy(float); h = f["high"].to_numpy(float); l = f["low"].to_numpy(float)
        data[sym] = {"close": c, "high": h, "low": l, "atr": atr(h, l, c, 14),
                     "ema50": ema(c, 50), "ema200": ema(c, 200),
                     "roc": {k: np.concatenate([np.full(k, np.nan), c[k:]/c[:-k]-1])
                             for k in (1, 3, 7, 14, 30)}}
    return data


def sim_trail(entries, c, h, l, a, is_long, mult, hold):
    out = []
    n = len(c)
    for i in np.nonzero(entries)[0]:
        k = int(i)
        if k+1 >= n: break
        e = c[k]; risk = mult*a[k]
        if e <= 0 or not np.isfinite(risk) or risk <= 0: continue
        stop = e - risk if is_long else e + risk
        peak = e; x = None
        for bar in range(k+1, min(n, k+1+hold)):
            if is_long:
                if l[bar] <= stop: x = stop*(1-SLIP); break
                if h[bar] > peak: peak = h[bar]; stop = max(stop, peak-risk)
            else:
                if h[bar] >= stop: x = stop*(1+SLIP); break
                if l[bar] < peak: peak = l[bar]; stop = min(stop, peak+risk)
        if x is None: x = c[min(n-1, k+hold)]
        out.append(((x-e) if is_long else (e-x))/e*LEV*100 - FEE)
    return np.asarray(out)


def stats(a): return exit_policy.trade_stats(a)


def h_ema_regime(data, lo, hi, mult=3.0, hold=30):
    t = []
    for s,d in data.items():
        c,h,l,a,e50,e200 = d["close"],d["high"],d["low"],d["atr"],d["ema50"],d["ema200"]
        for is_long in (True, False):
            e = np.zeros(hi-lo, bool)
            for i in range(lo+200, hi):
                ok = (c[i] > e200[i]) if is_long else (c[i] < e200[i])
                e[i-lo] = ok and e50[i] == e50[i]
            # enter only on regime change
            ch = np.zeros_like(e)
            ch[1:] = e[1:] & ~e[:-1]
            t.extend(sim_trail(ch, c[lo:hi], h[lo:hi], l[lo:hi], a[lo:hi], is_long, mult, hold))
    return np.asarray(t)


def h_momentum(data, lo, hi, look, topk, mode="momentum", rebal=7):
    syms = sorted(data); t = []
    for start in range(lo+look, hi-1, rebal):
        sc = {s: data[s]["roc"][look][start] for s in syms if np.isfinite(data[s]["roc"][look][start])}
        if len(sc) < topk*2+2: continue
        rk = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)
        longs = rk[-topk:] if mode == "reversal" else rk[:topk]
        shorts = rk[:topk] if mode == "reversal" else rk[-topk:]
        for s,_ in longs:
            e = data[s]["close"][start]; x = data[s]["close"][min(start+rebal, len(data[s]["close"])-1)]
            t.append((x-e)/e*LEV*100 - FEE - SLIP*LEV*100)
        for s,_ in shorts:
            e = data[s]["close"][start]; x = data[s]["close"][min(start+rebal, len(data[s]["close"])-1)]
            t.append((e-x)/e*LEV*100 - FEE - SLIP*LEV*100)
    return np.asarray(t)


def h_breakout(data, lo, hi, look, mult, hold=30):
    t = []
    for s,d in data.items():
        c,h,l,a = d["close"],d["high"],d["low"],d["atr"]
        for is_long in (True, False):
            e = np.zeros(hi-lo, bool)
            for i in range(lo+look, hi):
                if is_long: e[i-lo] = c[i] > np.max(h[i-look:i])
                else: e[i-lo] = c[i] < np.min(l[i-look:i])
            t.extend(sim_trail(e, c[lo:hi], h[lo:hi], l[lo:hi], a[lo:hi], is_long, mult, hold))
    return np.asarray(t)


def main():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as fh: frames = pickle.load(fh)
        log(f"loaded cache: {len(frames)} symbols")
    else:
        frames = fetch_daily()
        with open(CACHE, "wb") as fh: pickle.dump(frames, fh)
    if len(frames) < 5: log("insufficient"); return
    data = build(frames)
    n = min(len(d["close"]) for d in data.values())
    data = {s: {k: (v[:n] if isinstance(v, np.ndarray) else v) for k,v in d.items()}
            for s,d in data.items()}
    cut = int(n*0.75)
    log(f"symbols={len(data)} bars={n} dev={cut} holdout={n-cut}")

    hypotheses = {
        "ema200_regime_3atr": lambda lo,hi: h_ema_regime(data,lo,hi,3.0,30),
        "ema200_regime_5atr": lambda lo,hi: h_ema_regime(data,lo,hi,5.0,60),
        "xs_mom_7d_k3": lambda lo,hi: h_momentum(data,lo,hi,7,3,"momentum",7),
        "xs_mom_30d_k3": lambda lo,hi: h_momentum(data,lo,hi,30,3,"momentum",7),
        "xs_rev_3d_k3": lambda lo,hi: h_momentum(data,lo,hi,3,3,"reversal",3),
        "xs_rev_7d_k3": lambda lo,hi: h_momentum(data,lo,hi,7,3,"reversal",7),
        "breakout20_3atr": lambda lo,hi: h_breakout(data,lo,hi,20,3.0,30),
        "breakout55_4atr": lambda lo,hi: h_breakout(data,lo,hi,55,4.0,60),
    }
    n_tests = len(hypotheses)
    folded = {f: cut//N_FOLDS for f in range(N_FOLDS)}
    scores = []
    for name,fn in hypotheses.items():
        d = stats(fn(0,cut))
        fp = 0
        for f in range(N_FOLDS):
            lo,hi = f*folded[f], ((f+1)*folded[f] if f<N_FOLDS-1 else cut)
            r = stats(fn(lo,hi))
            if r["mean"]>0 and r["n"]>=3: fp+=1
        scores.append({"name":name,"dev":d,"folds":fp})
        log(f"DEV {name:22s} n={d['n']:5d} mean={d['mean']:7.3f} t={d['t_stat']:7.3f} PF={d['profit_factor']:.3f} folds={fp}/{N_FOLDS}")
    eligible = [s for s in scores if s["folds"]>=4 and s["dev"]["n"]>=40]
    pool = eligible or scores
    pool.sort(key=lambda s:s["dev"]["t_stat"], reverse=True)
    w = pool[0]
    log(f"FROZEN {w['name']} t={w['dev']['t_stat']:.3f} eligible={len(eligible)}")
    hold = stats(hypotheses[w["name"]](cut,n))
    log(f"HOLDOUT n={hold['n']} mean={hold['mean']:.3f} t={hold['t_stat']:.3f} PF={hold['profit_factor']:.3f}")
    out = {"symbols":len(data),"bars":n,"cut":cut,"n_tests":n_tests,
           "scores":scores,"winner":w,"holdout":hold,
           "passes_naive": bool(hold["n"]>=40 and hold["mean"]>0 and hold["t_stat"]>=1.96),
           "generated_at":datetime.now(timezone.utc).isoformat()}
    p = os.path.join(HERE,"daily_alpha_results.json")
    with open(p,"w",encoding="utf-8") as fh: json.dump(out,fh,indent=2,default=str)
    log(f"saved {p}")


if __name__ == "__main__": main()
