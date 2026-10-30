# --------------------------------------------------------------
# research/entry_vs_exit.py
# --------------------------------------------------------------
#
# WHERE THE EDGE ACTUALLY LIVES
# ------------------------------
# The rebuilt feeder rejects every candidate, because with the
# production exit ladder applied, every candidate loses money. That
# raises the obvious question: is the ENTRY bad, or is the EXIT bad?
#
# The two can be separated, because the forward return of an entry
# does not depend on any exit rule at all:
#
#   forward(i) = (close[i+H] - close[i]) / close[i] * leverage * 100
#
# where H is the maximum hold the executor allows (480 minutes = 32
# bars of 15m). If the forward return is positive but the realised
# result is negative, the exit ladder is the problem.
#
# Run:  python research/entry_vs_exit.py [SYMBOL ...]
# --------------------------------------------------------------
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import exit_policy
import main
from strategy_engine import BayesianStrategyEngine

HOLD_BARS = config.MAX_HOLD_BARS  # 32 bars of 15m = the executor's 480m cap


def forward_returns(close, entries, lo, hi, n):
    """Direction-signed forward return, gross percent, no exit rule."""
    idx = np.nonzero(entries[lo:hi])[0] + lo
    out = []
    for i in idx:
        j = min(i + HOLD_BARS, n - 1)
        out.append((close[j] - close[i]) / close[i] * config.LEVERAGE * 100.0)
    return np.asarray(out)


def t_stat(sample):
    if len(sample) < 2:
        return 0.0
    sd = sample.std(ddof=1)
    return float(sample.mean() / (sd / np.sqrt(len(sample)))) if sd > 0 else 0.0


def best_long_threshold(engine, n):
    """The LONG threshold with the best forward return (the entry-side
    evidence, with no exit rule in the way)."""
    best = None
    for lower in range(config.RSI_LOWER_RANGE[0], config.RSI_LOWER_RANGE[1] + 1):
        params = {
            "direction": "LONG",
            "rsi_lower": float(lower),
            "rsi_upper": 70.0,
        }
        entries = engine.build_entries(params, 0, n)
        fwd = forward_returns(engine.close, entries, 0, n, n)
        if len(fwd) >= 60 and (best is None or fwd.mean() > best[1]):
            best = (lower, float(fwd.mean()), entries, fwd)
    return best


def analyse(symbol):
    frame = main.fetch_ohlcv(symbol)
    if frame is None:
        print(f"{symbol}: no data")
        return

    engine = BayesianStrategyEngine(frame)
    n = len(frame)
    close = engine.close

    found = best_long_threshold(engine, n)
    if found is None:
        print(f"{symbol}: no threshold produced 60+ entries")
        return
    lower, fwd_mean, entries, fwd = found

    ladder = exit_policy.simulate_trades(
        engine.high, engine.low, engine.close, entries, True, "fav"
    )
    lad = np.asarray(ladder)

    print(f"\n=== {symbol} | {n} bars | LONG, RSI(14) <= {lower} ===")
    print(f"  forward return, no exit rule : mean {fwd.mean():>6.3f} gross  "
          f"sd {fwd.std(ddof=1):>5.2f}  t {t_stat(fwd):>6.2f}  n {len(fwd)}"
          f"   (fee {config.FEE_GROSS:.2f} -> net {fwd.mean() - config.FEE_GROSS:>6.3f})")
    if len(lad):
        print(f"  with the production ladder   : mean {lad.mean():>6.3f} gross  "
              f"sd {lad.std(ddof=1):>5.2f}  t {t_stat(lad):>6.2f}  n {len(lad)}")
        print(f"  the ladder gives up           : {fwd.mean() - lad.mean():>6.3f} gross per trade")

    # Per-fold consistency of the forward edge: an edge that only exists
    # in one stretch of the sample is not an edge.
    dev_end = int(n * 0.75)
    unit = dev_end // 6
    per = []
    for f in range(5):
        start = (f + 1) * unit
        end = min(dev_end, start + unit)
        if end <= start:
            continue
        seg = forward_returns(close, entries, start, end, n)
        if len(seg):
            per.append(round(float(seg.mean()), 3))
    print(f"  per-fold forward means        : {per}")

    # The decisive test: is the edge persistent, or does it live in one
    # regime? A 30-day chunking of the forward return shows this. On the
    # 90-day window the RSI fade looks like a 4 sigma edge; chunked over
    # 216 days the same signal swings from +1.7 to -1.9 gross, which is
    # why the search window has to be long enough to see a trend.
    chunk = 30 * 96
    chunks = []
    idx = np.nonzero(entries)[0]
    fwd_all = forward_returns(close, entries, 0, n, n)
    for c0 in range(0, n, chunk):
        mask = (idx >= c0) & (idx < min(n, c0 + chunk))
        if mask.sum() < 8:
            continue
        seg = fwd_all[mask]
        chunks.append(round(float(seg.mean()), 2))
    print(f"  per-30d forward means         : {chunks}")
    print(f"  (a signal whose edge lives in one regime is not an edge; "
          f"{len(chunks)} chunks were measured)")


def run():
    symbols = sys.argv[1:] or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    for symbol in symbols:
        analyse(symbol)


if __name__ == "__main__":
    run()
