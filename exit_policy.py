# --------------------------------------------------------------
# exit_policy.py - Python mirror of the Go ExitPolicy
# --------------------------------------------------------------
#
# The old backtest had its own exit rules (stop_loss_pct 2-4% of PRICE,
# take_profit_pct 4-6.5% of PRICE) while the executor ran the ladder in
# internal/strategy/exit_policy.go (3.0 gross initial stop, break-even
# at 2.0 locking 0.6, trailing from 3.0 with a 0.8 giveback, widening
# above a 12.0 peak, 480 minute time stop). At 3x leverage the old
# numbers were 6-12 gross percent of risk against the executor's 3.0:
# the backtest was measuring a robot that nobody runs.
#
# Everything in this file works in GROSS margin percent:
#
#     grossPct = (priceMovePct) * leverage * 100
#
# which is the unit the executor itself uses, so the geometry is
# leverage-agnostic and comparable to the journal.
# --------------------------------------------------------------
import numpy as np

import config


def lock_for(peak_gross: float):
    """Map a peak gross PnL to (locked gross PnL, tier).

    MIRROR of Go ExitPolicy.lockFor.
    """
    p = config.EXIT_POLICY
    if peak_gross >= p["trail_widen_peak_gross_pct"]:
        return (
            max(p["min_lock_gross_pct"], peak_gross - p["trail_widen_giveback_gross_pct"]),
            3,
        )
    if peak_gross >= p["trail_arm_gross_pct"]:
        return (
            max(p["min_lock_gross_pct"], peak_gross - p["trail_giveback_gross_pct"]),
            2,
        )
    if peak_gross >= p["be_arm_gross_pct"]:
        return p["be_lock_gross_pct"], 1
    return 0.0, 0


def stop_price(entry: float, is_long: bool, peak_gross: float):
    """The stop price the policy wants in place, and its tier.

    MIRROR of Go ExitPolicy.StopPriceFor.

    Note the sign flip between the tiers, which is the part that is
    easy to get wrong: tier 0 is the initial STOP and sits on the
    losing side of the entry, while tiers 1-3 are locked PROFITS and
    sit on the winning side. Putting the locked stop on the wrong side
    turns every trailed trade into a loss.
    """
    p = config.EXIT_POLICY
    lock, tier = lock_for(peak_gross)
    if tier == 0:
        px = p["initial_sl_gross_pct"] / (config.LEVERAGE * 100.0)
        if is_long:
            return _round_price(entry * (1.0 - px), entry), tier
        return _round_price(entry * (1.0 + px), entry), tier

    px = lock / (config.LEVERAGE * 100.0)
    if is_long:
        return _round_price(entry * (1.0 + px), entry), tier
    return _round_price(entry * (1.0 - px), entry), tier


def _round_price(px: float, ref: float) -> float:
    """MIRROR of Go roundPrice: 4 decimals, 2 when the reference is >= 1000."""
    scale = 100.0 if ref >= 1000 else 10000.0
    return round(px * scale) / scale


def simulate_trades(high, low, close, entries, is_long: bool, order: str = "fav"):
    """Walk every entry forward under the production ladder.

    `entries` marks the bar on which the entry was CONFIRMED. The trade
    opens at that bar's CLOSE, which is what the executor actually
    does: the ticker computes the live RSI from the closing candle,
    the confirmation fires at the end of that bar, and the market order
    is filled at approximately that price. The exit walk then starts on
    the NEXT bar, because this bar's range happened before the entry
    existed and using it would be look-ahead.

    Intrabar convention, identical to the executor's research harness
    (tethgard-multi-go/research/simulate.py): every bar is walked as
    open -> favourable extreme -> adverse extreme -> close, and the
    stop ratchets as the favourable extreme is passed. order="fav"
    resolves the favourable extreme first, which is what a 5-second
    poller usually sees on a trending bar; order="adv" is the
    pessimistic mirror. The two bracket the true result.

    Returns a list of net gross-percent results, one per closed trade.
    The round-trip fee is subtracted once per trade.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    e = np.asarray(entries, dtype=bool)

    lev = config.LEVERAGE
    fee = config.FEE_GROSS
    slip = config.SLIPPAGE
    hold_bars = config.MAX_HOLD_BARS

    results = []
    n = c.size

    for i in np.nonzero(e)[0]:
        k = int(i)
        if k + 1 >= n:
            break

        entry = c[k]
        if entry <= 0:
            continue

        def gross(px):
            return (
                (px - entry) if is_long else (entry - px)
            ) / entry * lev * 100.0

        stop, _tier = stop_price(entry, is_long, 0.0)
        peak = 0.0
        exit_gross = None

        for bar in range(k + 1, n):
            if is_long:
                pts = (h[bar], l[bar], c[bar]) if order == "fav" else (l[bar], h[bar], c[bar])
            else:
                pts = (l[bar], h[bar], c[bar]) if order == "fav" else (h[bar], l[bar], c[bar])

            for px in pts:
                if is_long and px <= stop:
                    exit_gross = gross(stop * (1.0 - slip)) - fee
                    break
                if (not is_long) and px >= stop:
                    exit_gross = gross(stop * (1.0 + slip)) - fee
                    break
                # ratchet the stop on the way
                g = gross(px)
                if g > peak:
                    peak = g
                    stop, _tier = stop_price(entry, is_long, peak)
            if exit_gross is not None:
                break

            if bar - k >= hold_bars:
                exit_gross = gross(c[bar]) - fee
                break

        if exit_gross is None:
            exit_gross = gross(c[n - 1]) - fee

        results.append(exit_gross)

    return results


def forward_returns(close, entries, is_long: bool, hold_bars: int = None):
    """Direction-signed forward return of every entry, gross percent.

    This is the ONE measure of a signal that no exit rule can influence:
    where the price goes after the entry, over the executor's maximum
    hold. It is what separates "the entry has no edge" from "the exit
    ladder is destroying the edge", because a strategy can be rejected
    for the second reason while the first is perfectly healthy.

    No fee is subtracted: this is the raw move, and the fee is a
    constant that applies to both the ladder and the forward measure.
    """
    hold_bars = hold_bars or config.MAX_HOLD_BARS
    c = np.asarray(close, dtype=float)
    e = np.asarray(entries, dtype=bool)
    n = c.size
    out = []
    for i in np.nonzero(e)[0]:
        entry = c[i]
        if entry <= 0:
            continue
        j = min(int(i) + hold_bars, n - 1)
        if is_long:
            out.append((c[j] - entry) / entry * config.LEVERAGE * 100.0)
        else:
            out.append((entry - c[j]) / entry * config.LEVERAGE * 100.0)
    return np.asarray(out, dtype=float)


def trade_stats(results):
    """Summary statistics for a list of net gross-percent trade results.

    The headline number is the t-statistic of the mean, not a Sharpe
    ratio. With tens of trades the annualisation factor is a constant
    that changes nothing about the ranking, while the t-statistic
    divides by the standard error and therefore penalises a strategy
    that was right once.
    """
    arr = np.asarray(results, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "t_stat": 0.0,
            "win_rate": 0.0,
            "total": 0.0,
            "profit_factor": 0.0,
        }

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    t_stat = (mean / (std / np.sqrt(n))) if (std > 0 and n > 1) else 0.0
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "t_stat": float(t_stat),
        "win_rate": float((arr > 0).mean()),
        "total": float(arr.sum()),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else 0.0,
    }
