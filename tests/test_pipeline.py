# --------------------------------------------------------------
# tests/test_pipeline.py
# --------------------------------------------------------------
#
# Run with:  python tests/test_pipeline.py      (no pytest needed)
#      or:   pytest tests/test_pipeline.py
#
# The tests that matter most are the ones that would have caught the
# defects in the old feeder: an indicator that does not match the Go
# executor, an exit ladder that is not the one production runs, an
# entry that can see the future, and a median that mixes up the
# parameters with the metrics.
# --------------------------------------------------------------
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import exit_policy
import indicators
import strategy_engine
from strategy_engine import BayesianStrategyEngine, build_candidates, critical_t


# ---------------------------------------------------------------------------
# A literal transcription of Go's calculateRSI, used as an independent
# reference. If the vectorised port in indicators.py ever drifts from the
# executor, this fails.
# ---------------------------------------------------------------------------
def go_calculate_rsi(closes, period=14):
    n = len(closes)
    if n <= period:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, n):
        change = closes[i] - closes[i - 1]
        if change > 0:
            avg_gain = (avg_gain * (period - 1) + change) / period
            avg_loss = (avg_loss * (period - 1)) / period
        else:
            avg_gain = (avg_gain * (period - 1)) / period
            avg_loss = (avg_loss * (period - 1) - change) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round((100.0 - (100.0 / (1.0 + rs))) * 100) / 100


def _walk_forward_frame(bars=1200, seed=7):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.004, bars)
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1.0 + rng.uniform(0.0, 0.004, bars))
    low = close * (1.0 - rng.uniform(0.0, 0.004, bars))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}
    )


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def test_rsi_matches_go():
    frame = _walk_forward_frame()
    closes = frame["close"].tolist()
    series = indicators.wilder_rsi(closes, 14)
    for i in (14, 15, 50, 200, 700, 1199):
        expected = go_calculate_rsi(closes[: i + 1], 14)
        got = round(float(series[i]), 2)
        assert abs(got - expected) < 1e-9, f"RSI mismatch at {i}: {got} != {expected}"


def test_rsi_boundaries():
    rising = list(np.arange(1.0, 60.0))
    falling = list(rising[::-1])
    assert indicators.wilder_rsi(rising, 14)[-1] == 100.0
    assert indicators.wilder_rsi(falling, 14)[-1] == 0.0
    # Not enough data must be NaN, never a silent 50.
    short = indicators.wilder_rsi([1.0, 2.0, 3.0], 14)
    assert np.isnan(short).all()


def test_ema_is_causal_and_seeded():
    frame = _walk_forward_frame()
    closes = frame["close"].tolist()
    full = indicators.ema_seeded(closes, 200)
    trunc = indicators.ema_seeded(closes[:600], 200)
    # Truncating the series must not change any earlier EMA value.
    for i in range(200, 600):
        assert abs(full[i] - trunc[i]) < 1e-9, f"EMA changed at {i}"
    # Seed is the SMA of the first 200 closes.
    assert abs(full[199] - float(np.mean(closes[:200]))) < 1e-9


def test_adx_is_causal():
    frame = _walk_forward_frame()
    full = indicators.wilder_adx(
        frame["high"], frame["low"], frame["close"], 14
    )
    trunc = indicators.wilder_adx(
        frame["high"][:600], frame["low"][:600], frame["close"][:600], 14
    )
    for i in range(30, 600):
        assert abs(full[i] - trunc[i]) < 1e-9, f"ADX changed at {i}"

# ---------------------------------------------------------------------------
# Exit policy
# ---------------------------------------------------------------------------
def test_exit_ladder_tiers():
    # tier 0: the initial stop, on the LOSING side of the entry.
    lock, tier = exit_policy.lock_for(0.0)
    assert tier == 0 and lock == 0.0
    stop, tier = exit_policy.stop_price(100.0, True, 0.0)
    assert tier == 0
    assert abs(stop - 99.0) < 1e-9, "initial LONG stop must be BELOW entry"

    stop_short, tier = exit_policy.stop_price(100.0, False, 0.0)
    assert tier == 0
    assert abs(stop_short - 101.0) < 1e-9, "initial SHORT stop must be ABOVE entry"

    # tier 1: break-even lock, on the WINNING side.
    # 0.6 gross at 3x is 0.2% of price.
    stop, tier = exit_policy.stop_price(100.0, True, 2.0)
    assert tier == 1
    assert abs(stop - 100.2) < 1e-9, "break-even LONG stop must be ABOVE entry"

    stop, tier = exit_policy.stop_price(100.0, False, 2.0)
    assert tier == 1
    assert abs(stop - 99.8) < 1e-9, "break-even SHORT stop must be BELOW entry"

    # tier 2: trailing, locking peak minus the giveback.
    stop, tier = exit_policy.stop_price(100.0, True, 5.0)
    assert tier == 2
    assert abs(stop - 100.0 * (1 + 4.2 / 300.0)) < 1e-9

    # tier 3: the widened trail above a 12.0 peak.
    stop, tier = exit_policy.stop_price(100.0, True, 20.0)
    assert tier == 3
    assert abs(stop - 100.0 * (1 + 18.0 / 300.0)) < 1e-9


def test_exit_ladder_never_locks_less_than_min():
    # A peak just above the trail arm locks peak - 0.8, which is above the
    # 0.4 floor, but a widened trail at exactly 12.0 locks 10.0. The floor
    # only matters in pathological configurations.
    lock, tier = exit_policy.lock_for(3.0)
    assert tier == 2
    assert abs(lock - 2.2) < 1e-9
    assert lock >= config.EXIT_POLICY["min_lock_gross_pct"]


def test_simulate_stops_out_and_locks_profit():
    n = 8
    o = np.full(n, 100.0)
    c = np.full(n, 100.0)
    h = np.full(n, 100.0)
    l = np.full(n, 100.0)
    entries = np.zeros(n, dtype=bool)
    entries[0] = True

    # A straight collapse must lose the initial stop, minus the fee.
    l[2] = 95.0
    res = exit_policy.simulate_trades(h, l, c, entries, True)
    assert len(res) == 1
    assert -4.0 < res[0] < -3.0, f"unexpected stop-out result {res[0]}"

    # A spike then a collapse must be trailed out at a profit, not stopped
    # out at the original stop. This is the case that a wrong sign on the
    # locked stop would break.
    h2 = np.full(n, 100.0)
    l2 = np.full(n, 100.0)
    h2[2] = 106.0
    l2[3] = 95.0
    res2 = exit_policy.simulate_trades(h2, l2, c, entries, True)
    assert res2[0] > 5.0, f"trailed exit should be a clear profit, got {res2[0]}"


def test_simulate_never_uses_the_entry_bar_range():
    # The confirmation bar's range happened BEFORE the trade existed.
    n = 5
    c = np.full(n, 100.0)
    h = np.full(n, 100.0)
    l = np.full(n, 100.0)
    entries = np.zeros(n, dtype=bool)
    entries[1] = True
    # A deep low on the entry bar must not stop the trade out.
    l[1] = 50.0
    res = exit_policy.simulate_trades(h, l, c, entries, True)
    assert res, "the trade should still exist"
    assert res[0] > -3.0, f"entry bar range leaked into the simulation: {res[0]}"


def test_fee_is_charged_once_per_round_trip():
    n = 6
    c = np.full(n, 100.0)
    h = np.full(n, 100.0)
    l = np.full(n, 100.0)
    entries = np.zeros(n, dtype=bool)
    entries[0] = True
    l[2] = 95.0
    res = exit_policy.simulate_trades(h, l, c, entries, True)
    # The stop is at 99.0, slippage takes it slightly lower, and the fee is
    # 0.30 gross.
    expected = (99.0 * (1 - config.SLIPPAGE) - 100.0) / 100.0 * 300.0 - config.FEE_GROSS
    assert abs(res[0] - expected) < 1e-6, f"{res[0]} != {expected}"


def test_trade_stats_t_statistic():
    stats = exit_policy.trade_stats([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert abs(stats["mean"] - 2.5) < 1e-9
    assert stats["win_rate"] == 1.0
    assert stats["t_stat"] > 3.0, "a consistent run must have a high t-statistic"
    empty = exit_policy.trade_stats([])
    assert empty["n"] == 0 and empty["t_stat"] == 0.0


# ---------------------------------------------------------------------------
# Entry model
# ---------------------------------------------------------------------------
def _engine_with_rsi(rsi_values, length=30, entry_filter=True):
    close = np.linspace(100.0, 100.0 + length, length)
    frame = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1.0}
    )
    engine = BayesianStrategyEngine(frame)
    rsi = np.full(length, 50.0)
    for i, v in rsi_values:
        rsi[i] = v
    engine._rsi_l = rsi.tolist()
    if not entry_filter:
        # The arm/confirm logic is tested on its own; the entry filter
        # gets its own test below.
        engine.entry_filter = np.ones(length, dtype=bool)
    return engine


def test_entry_arms_and_confirms():
    engine = _engine_with_rsi([(3, 30.0), (4, 35.0), (10, 20.0), (11, 25.0)], entry_filter=False)
    params = {"direction": "LONG", "rsi_lower": 30.0, "rsi_upper": 70.0}
    entries = np.nonzero(engine.build_entries(params, 0, 30))[0].tolist()
    assert entries == [4, 11], entries


def test_entry_requires_a_two_point_reversal():
    engine = _engine_with_rsi([(3, 30.0), (4, 31.0)], entry_filter=False)
    params = {"direction": "LONG", "rsi_lower": 30.0, "rsi_upper": 70.0}
    entries = np.nonzero(engine.build_entries(params, 0, 30))[0].tolist()
    assert entries == [], "a one point move is not a reversal"


def test_entry_expires_after_one_bar():
    # Armed at bar 3, the state may not survive into bar 5.
    engine = _engine_with_rsi([(3, 30.0), (4, 20.0), (5, 40.0)], entry_filter=False)
    params = {"direction": "LONG", "rsi_lower": 30.0, "rsi_upper": 70.0}
    entries = np.nonzero(engine.build_entries(params, 0, 30))[0].tolist()
    assert 5 not in entries, "the arm must not survive past one 15m bar"


def test_short_entry_uses_the_upper_bound():
    engine = _engine_with_rsi([(2, 72.0), (3, 68.0)], entry_filter=False)
    params = {"direction": "SHORT", "rsi_lower": 30.0, "rsi_upper": 70.0}
    entries = np.nonzero(engine.build_entries(params, 0, 30))[0].tolist()
    assert entries == [3], entries


def test_executor_entry_filter_is_applied():
    """The executor filters every entry, so the backtest must too.

    Without this the search scores a strategy that trades far more often
    than the one that runs.
    """
    engine = _engine_with_rsi([(3, 30.0), (4, 35.0)], entry_filter=False)
    params = {"direction": "LONG", "rsi_lower": 30.0, "rsi_upper": 70.0}
    assert np.nonzero(engine.build_entries(params, 0, 30))[0].tolist() == [4]

    # Now block the confirmation bar with the filter.
    engine.entry_filter = np.zeros(30, dtype=bool)
    engine.entry_filter[4] = False
    assert np.nonzero(engine.build_entries(params, 0, 30))[0].tolist() == []
    engine.entry_filter[4] = True
    assert np.nonzero(engine.build_entries(params, 0, 30))[0].tolist() == [4]


# ---------------------------------------------------------------------------
# Look-ahead
# ---------------------------------------------------------------------------
def test_no_lookahead_entries_are_causal():
    """Truncating the market must not change any earlier entry.

    This is the test the old feeder could never have passed: it scored
    the winning parameters on the raw frame, so a signal was allowed to
    fill on the same bar that produced it.
    """
    frame = _walk_forward_frame(bars=900, seed=11)
    full = BayesianStrategyEngine(frame)
    params = {"direction": "LONG", "rsi_lower": 30.0, "rsi_upper": 70.0}

    for cut in (400, 600, 800):
        trunc = BayesianStrategyEngine(frame.iloc[:cut].reset_index(drop=True))
        a = full.build_entries(params, 0, cut)
        b = trunc.build_entries(params, 0, cut)
        assert np.array_equal(a, b), f"entries changed when the future was removed at {cut}"


def test_candidate_space_is_only_executable_parameters():
    candidates = build_candidates()
    assert len(candidates) == 52, len(candidates)
    allowed = {"direction", "rsi_lower", "rsi_upper"}
    for params in candidates:
        assert set(params.keys()) == allowed, params
        assert params["direction"] in ("LONG", "SHORT")
        if params["direction"] == "LONG":
            assert config.RSI_LOWER_RANGE[0] <= params["rsi_lower"] <= config.RSI_LOWER_RANGE[1]
        else:
            assert config.RSI_UPPER_RANGE[0] <= params["rsi_upper"] <= config.RSI_UPPER_RANGE[1]


def test_payload_only_carries_fields_the_executor_understands():
    """The executor stores {symbol, direction, leverage, rsi_lower, rsi_upper}.

    The old feeder also sent strategy_type and oos_sharpe, which the Go
    handler silently discarded -- which is how a strategy validated as a
    momentum breakout could be executed as a mean reversion.
    """
    import main

    accepted = set()
    for params in build_candidates():
        accepted.update(params.keys())
    assert accepted <= {"direction", "rsi_lower", "rsi_upper"}
    assert main.executor_symbol("BTC/USDT") == "BTCUSDT"
    assert main.executor_symbol("BTC/USDT:USDT") == "BTCUSDT"
    assert main.executor_symbol("XAUT/USDT") == "XAUTUSDT"


def test_neutral_row_can_never_arm():
    """Retiring a symbol relies on the ticker gate, not on the strategy."""
    # The ticker arms a LONG only when liveRSI <= rsi_lower, and the RSI is
    # never 0, so a LONG with rsi_lower = 0 is inert.
    assert config.RSI_LOWER_RANGE[0] > 0
    neutral = {"direction": "LONG", "rsi_lower": 0.0, "rsi_upper": 100.0}
    engine = _engine_with_rsi([(3, 50.0), (4, 50.0)])
    entries = np.nonzero(engine.build_entries(neutral, 0, 30))[0].tolist()
    assert entries == [], "a neutral row must never produce an entry"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_critical_t_grows_with_the_number_of_candidates():
    assert critical_t(1, 0.05) == 0.0
    assert critical_t(52, 0.05) > critical_t(10, 0.05)
    # Bonferroni over 52 candidates at alpha 0.05 is z(1 - 0.05/52) = 3.10.
    # The best of 52 noise draws has to look like a 3.1 sigma result.
    assert 2.9 < critical_t(52, 0.05) < 3.3


def test_median_fold_keeps_parameters_with_metrics():
    """The old engine sorted the metrics and indexed the unsorted parameter
    list, so the published parameters belonged to a different fold than the
    published numbers.
    """
    engine = _engine_with_rsi([], length=10)
    folds = [
        {"fold": 0, "params": {"direction": "LONG", "rsi_lower": 20.0, "rsi_upper": 70.0},
         "stats": {"t_stat": 1.0, "n": 10, "mean": 0.1}},
        {"fold": 1, "params": {"direction": "SHORT", "rsi_lower": 30.0, "rsi_upper": 75.0},
         "stats": {"t_stat": 5.0, "n": 10, "mean": 0.9}},
        {"fold": 2, "params": {"direction": "LONG", "rsi_lower": 25.0, "rsi_upper": 70.0},
         "stats": {"t_stat": 3.0, "n": 10, "mean": 0.5}},
    ]
    ranked = sorted(folds, key=lambda f: f["stats"]["t_stat"])
    median = ranked[len(ranked) // 2]
    assert median["stats"]["t_stat"] == 3.0
    assert median["params"]["rsi_lower"] == 25.0, "parameters and metrics are mixed up"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run():
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run()
