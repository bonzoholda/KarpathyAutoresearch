#!/usr/bin/env python3
"""Can a trend lane be COMBINED with the existing RSI-reversal lane?

This test is deliberately stricter than signal_family_study.py:

  1. Trend candidates are PRE-REGISTERED (3 families x 2 exit modes x 2
     directions = 12 candidates). No per-symbol cherry-picking.
  2. Indicators are computed ONCE on the full series and then sliced, so a
     fold never changes the meaning of an indicator.
  3. The candidate is selected on the DEVELOPMENT period only, then scored
     exactly ONCE on a time-ordered 25% holdout.
  4. Both the naive gate (t >= 1.96) and a Bonferroni gate for 12 comparisons
     are reported.
  5. The COMBINED system is measured against the legacy lane alone on the
     same holdout. A combination is only useful if it improves risk-adjusted
     holdout return, not merely if it produces more trades.

The legacy lane mirrors strategy_engine.build_entries (RSI arm+confirm,
ADX<32 filter, trend inversion) and the production exit ladder.

Run: python research/combined_lane_study.py
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

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
           "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "NEAR/USDT", "SUI/USDT", "XAU/USDT"]

N_COMPARISONS = 3 * 2 * 2          # families x exit modes x directions
LEGACY_LOWER, LEGACY_UPPER = 35.0, 65.0
TREND_ADX_MIN = 25.0               # trend regime, opposite of legacy ADX<32
GATE_ADX = 25.0                    # regime split used by the combined system
ATR_STOP_MULT = 2.0


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def build_arrays(frame):
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    vol = frame["volume"].to_numpy(dtype=float)
    adx = indicators.wilder_adx(high, low, close, 14)
    atr = indicators.atr_series(high, low, close, 14)
    rsi = indicators.wilder_rsi(close, config.RSI_PERIOD)
    ema200 = indicators.ema_seeded(close, 200)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.abs(close - ema200) / atr
    legacy_filter = (
        (adx < config.ENTRY_FILTER_MAX_ADX)
        & (dist >= config.ENTRY_FILTER_MIN_DIST_ATR)
        & np.isfinite(dist)
    )
    trend_filter = np.isfinite(adx) & (adx >= TREND_ADX_MIN) & np.isfinite(atr) & (atr > 0)
    return {"close": close, "high": high, "low": low, "vol": vol, "adx": adx,
            "atr": atr, "rsi": rsi, "ema200": ema200,
            "legacy_filter": legacy_filter, "trend_filter": trend_filter}


def cross_above(a, b):
    return (a > b) & (np.roll(a, 1) <= np.roll(b, 1))


def cross_below(a, b):
    return (a < b) & (np.roll(a, 1) >= np.roll(b, 1))


def sma(x, n):
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.nan_to_num(x, nan=0.0))
    out[n - 1:] = (cs[n - 1:] - np.concatenate([[0.0], cs[:-n]])) / n
    return out


def rolling_high(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.max(x[i - n + 1:i])       # excludes current bar
    return out


def rolling_low(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.min(x[i - n + 1:i])
    return out


def make_trend_entries(d, family):
    """Full-series boolean entry arrays, no look-ahead (prior-bar breakout)."""
    close, vol, adx = d["close"], d["vol"], d["adx"]
    if family == "ema9_21":
        fast, slow = indicators.ema_seeded(close, 9), indicators.ema_seeded(close, 21)
        long_e, short_e = cross_above(fast, slow), cross_below(fast, slow)
    elif family == "donchian20_v15":
        rh, rl = rolling_high(d["high"], 20), rolling_low(d["low"], 20)
        vs = sma(vol, 20)
        long_e = (close > rh) & (vol > 1.5 * vs)
        short_e = (close < rl) & (vol > 1.5 * vs)
    elif family == "mom20_4":
        roc = np.full(len(close), np.nan)
        roc[20:] = (close[20:] - close[:-20]) / close[:-20] * 100.0
        long_e, short_e = roc >= 4.0, roc <= -4.0
    else:
        raise ValueError(family)
    return (np.nan_to_num(long_e, nan=False) & d["trend_filter"],
            np.nan_to_num(short_e, nan=False) & d["trend_filter"])


def make_legacy_entries(d, direction):
    """Mirror of the Go/Python RSI arm+confirm trigger on the full series."""
    rsi_l = d["rsi"].tolist()
    adx_l = d["adx"].tolist()
    ema_l = d["ema200"].tolist()
    close = d["close"]
    filt = d["legacy_filter"]
    n = len(close)
    is_long = direction == "LONG"
    threshold = LEGACY_LOWER if is_long else LEGACY_UPPER
    reversal = config.RSI_REVERSAL
    out = np.zeros(n, dtype=bool)
    armed, armed_long, peak, arm_idx = False, is_long, 0.0, -1

    for i in range(n):
        r = rsi_l[i]
        if r != r:
            continue
        if armed and (i - arm_idx) > 1:            # 15m arm expiry
            armed = False
        if not armed:
            eff_long, inverted = is_long, False
            a, e = adx_l[i], ema_l[i]
            if (config.INVERSION_ENABLED and a == a and a >= config.INVERSION_ADX
                    and e == e and e > 0 and close[i] > 0):
                if (not is_long) and close[i] > e:
                    eff_long, inverted = True, True
                elif is_long and close[i] < e:
                    eff_long, inverted = False, True
            arm = (r <= threshold) if eff_long else (r >= threshold)
            arm = arm or inverted
            if arm:
                armed, armed_long, peak, arm_idx = True, eff_long, r, i
            continue
        if armed_long:
            if r < peak:
                peak = r
            elif r >= peak + reversal:
                if filt[i]:
                    out[i] = True
                armed = False
        else:
            if r > peak:
                peak = r
            elif r <= peak - reversal:
                if filt[i]:
                    out[i] = True
                armed = False
    return out


def simulate_atr_trail(high, low, close, atr, entries, is_long,
                       mult=ATR_STOP_MULT, max_bars=config.MAX_HOLD_BARS):
    """Conservative 2xATR trailing stop, same fee/leverage units as production."""
    h, l, c, a = (np.asarray(x, dtype=float) for x in (high, low, close, atr))
    out = []
    n = c.size
    for idx in np.nonzero(entries)[0]:
        k = int(idx)
        if k + 1 >= n:
            break
        entry = c[k]
        risk = mult * a[k]
        if entry <= 0 or not np.isfinite(risk) or risk <= 0:
            continue
        stop = entry - risk if is_long else entry + risk
        peak = entry
        exit_px = None
        for bar in range(k + 1, min(n, k + 1 + max_bars)):
            if is_long:
                if l[bar] <= stop:                 # adverse first (pessimistic)
                    exit_px = stop * (1.0 - config.SLIPPAGE)
                    break
                if h[bar] > peak:
                    peak = h[bar]
                    stop = max(stop, peak - risk)
            else:
                if h[bar] >= stop:
                    exit_px = stop * (1.0 + config.SLIPPAGE)
                    break
                if l[bar] < peak:
                    peak = l[bar]
                    stop = min(stop, peak + risk)
        if exit_px is None:
            exit_px = c[min(n - 1, k + max_bars)]
        gross = ((exit_px - entry) if is_long else (entry - exit_px)) / entry
        out.append(gross * config.LEVERAGE * 100.0 - config.FEE_GROSS)
    return out


def trade_results(entries, d, direction, exit_mode, lo, hi):
    e = entries[lo:hi]
    is_long = direction == "LONG"
    if exit_mode == "ladder":
        return exit_policy.simulate_trades(
            d["high"][lo:hi], d["low"][lo:hi], d["close"][lo:hi], e, is_long)
    return simulate_atr_trail(
        d["high"][lo:hi], d["low"][lo:hi], d["close"][lo:hi],
        d["atr"][lo:hi], e, is_long)


def pooled_stats(per_symbol):
    arr = np.concatenate(per_symbol) if per_symbol else np.array([])
    return exit_policy.trade_stats(arr)


def z_critical(alpha, comparisons):
    # normal approx; exact t is slightly larger for small n
    p = 1.0 - alpha / (2.0 * comparisons)
    # Acklam inverse normal approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def main():
    log(f"Fetching {len(SYMBOLS)} symbols, 15m, {config.DATA_DAYS}d")
    data = {}
    for sym in SYMBOLS:
        frame = fetch_ohlcv(sym)
        if frame is None or len(frame) < 500:
            log(f"  skip {sym}: insufficient data")
            continue
        data[sym] = build_arrays(frame)
        log(f"  {sym}: {len(frame)} bars")

    if not data:
        log("no data, abort")
        return

    n = len(next(iter(data.values()))["close"])
    holdout_start = int(n * (1 - config.HOLDOUT_FRACTION))

    # Precompute all full-series entry arrays once.
    trend_entries = {}
    legacy_entries = {}
    for sym, d in data.items():
        for fam in ("ema9_21", "donchian20_v15", "mom20_4"):
            le, se = make_trend_entries(d, fam)
            trend_entries[(sym, fam, "LONG")] = le
            trend_entries[(sym, fam, "SHORT")] = se
        legacy_entries[(sym, "LONG")] = make_legacy_entries(d, "LONG")
        legacy_entries[(sym, "SHORT")] = make_legacy_entries(d, "SHORT")

    # ---- DEV: select ONE trend candidate, never touching the holdout ----
    dev_scores = []
    for fam in ("ema9_21", "donchian20_v15", "mom20_4"):
        for exit_mode in ("ladder", "atr_trail"):
            for direction in ("LONG", "SHORT"):
                per_sym = []
                for sym, d in data.items():
                    per_sym.append(trade_results(
                        trend_entries[(sym, fam, direction)], d,
                        direction, exit_mode, 0, holdout_start))
                s = pooled_stats(per_sym)
                dev_scores.append({"family": fam, "exit": exit_mode,
                                   "direction": direction, **s})
    dev_scores.sort(key=lambda r: r["t_stat"], reverse=True)
    best = dev_scores[0]
    log(f"DEV winner: {best['family']} {best['exit']} {best['direction']} "
        f"n={best['n']} mean={best['mean']:.3f} t={best['t_stat']:.3f}")

    # ---- HOLDOUT: score the selected candidate exactly once ----
    h_per_sym = []
    for sym, d in data.items():
        h_per_sym.append(trade_results(
            trend_entries[(sym, best["family"], best["direction"])], d,
            best["direction"], best["exit"], holdout_start, n))
    trend_hold = pooled_stats(h_per_sym)

    legacy_h_per_sym = []
    for sym, d in data.items():
        legacy_h_per_sym.append(trade_results(
            legacy_entries[(sym, best["direction"])], d,
            best["direction"], "ladder", holdout_start, n))
    legacy_hold = pooled_stats(legacy_h_per_sym)

    # ---- Combined regime-gated system: legacy ADX<25, trend ADX>=25 ----
    combined_per_sym = []
    for sym, d in data.items():
        adx_h = d["adx"][holdout_start:n]
        # simulate both directions of the legacy lane explicitly
        for direction in ("LONG", "SHORT"):
            e = legacy_entries[(sym, direction)][holdout_start:n] & (adx_h < GATE_ADX)
            combined_per_sym.extend(exit_policy.simulate_trades(
                d["high"][holdout_start:n], d["low"][holdout_start:n],
                d["close"][holdout_start:n], e, direction == "LONG"))
        combined_per_sym.extend(trade_results(
            trend_entries[(sym, best["family"], best["direction"])][holdout_start:n] & (adx_h >= GATE_ADX),
            d, best["direction"], best["exit"], holdout_start, n))

    combined_hold = pooled_stats([np.asarray(combined_per_sym, dtype=float)])
    naive_t = z_critical(0.05, 1)
    bonf_t = z_critical(0.05, N_COMPARISONS)

    trend_pass = (trend_hold["n"] >= config.MIN_POOLED_TRADES
                  and trend_hold["t_stat"] >= bonf_t and trend_hold["mean"] > 0)
    trend_pass_naive = (trend_hold["n"] >= config.MIN_POOLED_TRADES
                        and trend_hold["t_stat"] >= naive_t and trend_hold["mean"] > 0)
    less_bad = (combined_hold["t_stat"] > legacy_hold["t_stat"]
                and combined_hold["mean"] > legacy_hold["mean"])
    # "Beats" must mean a profitable, statistically credible system - not
    # merely a smaller loss than another losing system.
    improves = (less_bad and combined_hold["mean"] > 0
                and combined_hold["t_stat"] >= naive_t)

    verdict = {
        "question": "combine legacy RSI lane with a holdout-passing trend lane",
        "symbols": len(data), "bars": n, "holdout_start": holdout_start,
        "holdout_bars": n - holdout_start,
        "pre_registered_comparisons": N_COMPARISONS,
        "dev_winner": best,
        "all_dev_scores": dev_scores,
        "holdout": {
            "legacy_lane": legacy_hold,
            "selected_trend_lane": trend_hold,
            "combined_regime_gated": combined_hold,
        },
        "critical_t": {"naive": naive_t, "bonferroni_12": bonf_t},
        "trend_passes_naive": trend_pass_naive,
        "trend_passes_bonferroni": trend_pass,
        "combined_less_bad_than_legacy": less_bad,
        "combined_beats_legacy": improves,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    log(f"HOLDOUT legacy  : n={legacy_hold['n']:4d} mean={legacy_hold['mean']:7.3f} "
        f"t={legacy_hold['t_stat']:6.3f} PF={legacy_hold['profit_factor']:.3f}")
    log(f"HOLDOUT trend   : n={trend_hold['n']:4d} mean={trend_hold['mean']:7.3f} "
        f"t={trend_hold['t_stat']:6.3f} PF={trend_hold['profit_factor']:.3f}")
    log(f"HOLDOUT combined: n={combined_hold['n']:4d} mean={combined_hold['mean']:7.3f} "
        f"t={combined_hold['t_stat']:6.3f} PF={combined_hold['profit_factor']:.3f}")
    log(f"critical t: naive={naive_t:.3f}, Bonferroni({N_COMPARISONS})={bonf_t:.3f}")
    log(f"VERDICT: trend naive={trend_pass_naive}, Bonferroni={trend_pass}, "
        f"combined less-bad={less_bad}, combined genuinely beats legacy={improves}")

    out = os.path.join(os.path.dirname(__file__), "combined_lane_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2, default=str)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
