# --------------------------------------------------------------
# strategy_engine.py - honest walk-forward search
# --------------------------------------------------------------
#
# WHAT THIS REPLACES
# ------------------
# The previous engine (upstream Karpathy fork) had four defects that
# made every result it produced meaningless:
#
#   1. The entry it validated was not the entry the executor runs. The
#      optimiser chose among three strategy types and seven parameters;
#      the executor stores only {symbol, direction, leverage, rsi_lower,
#      rsi_upper} and silently discards the rest. A "RSI_MOMENTUM_BREAKOUT"
#      winner was therefore executed as an RSI mean-reversion, i.e. the
#      opposite trade, while still being reported as a validated winner.
#
#   2. The exit it validated was not the exit the executor runs. It used
#      its own 2-4% price stop and 4-6.5% price take-profit, while the
#      executor runs the gross-percent ladder in exit_policy.go.
#
#   3. The final validation looked ahead. It scored the winning
#      parameters on the raw frame instead of the shifted one, so a
#      signal was allowed to fill on the same bar that produced it.
#
#   4. Its median selection was wrong. It sorted the metrics but indexed
#      the unsorted parameter list with the median position, so the
#      parameters it published belonged to a different fold than the
#      metrics it quoted.
#
# WHAT THIS DOES
# --------------
# The search space is exactly the set of knobs the executor honours:
# the direction, and one RSI arm threshold. The RSI period, the reversal
# confirmation size, the leverage and the entire exit ladder are
# constants, because in production they are constants.
#
# The entry is modelled the way the executor actually takes it:
#
#   ARM      long: rsi <= rsi_lower,  short: rsi >= rsi_upper
#   CONFIRM  long: rsi >= arm_low + 2.0,  short: rsi <= arm_high - 2.0
#   INVERT   if ADX(14) >= 25 and price is on the other side of EMA200,
#            the effective direction is flipped (Go: isInverted)
#
# An entry is executed at the open of the bar AFTER the confirmation, so
# no simulation ever uses information that was not available when the
# decision was made.
#
# WHY THERE IS NO OPTUNA
# ----------------------
# The search space is 26 LONG thresholds plus 26 SHORT thresholds, so an
# exhaustive grid is both faster and better behaved than a randomised
# TPE search: it is deterministic (the same data always gives the same
# answer, which is what a 30-minute cron needs), and the number of
# candidates tested is a known constant, which is what makes the
# multiple-testing correction below exact instead of approximate.
# --------------------------------------------------------------
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

import config
import exit_policy
import indicators


def critical_t(trials: int, alpha: float) -> float:
    """The t-value a result must reach to survive multiple testing.

    Bonferroni on the normal approximation: z(1 - alpha / trials). With
    52 candidates and alpha 0.05 this is 2.58, i.e. the best of 52 draws
    from pure noise still has to look like a 2.6 sigma result.
    """
    if trials <= 1:
        return 0.0
    p = 1.0 - alpha / trials
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return NormalDist().inv_cdf(p)


def load_active_config():
    """The strategies published by the last cycle, or None.

    Kept here because it used to live in this module; main.py writes
    the file and also exposes its own copy.
    """
    import json
    import os

    path = os.path.join("config", "strategy_config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("active") or None
    except (OSError, ValueError):
        return None


def build_candidates():
    """Every parameter set the executor can actually be given.

    For a LONG the executor arms on rsi_lower and ignores rsi_upper; for
    a SHORT it arms on rsi_upper and ignores rsi_lower. Only the arm
    threshold is therefore searchable, and the direction is the second
    axis. Nothing else exists in the executor's strategy table.
    """
    out = []
    for lower in range(config.RSI_LOWER_RANGE[0], config.RSI_LOWER_RANGE[1] + 1):
        out.append(
            {
                "direction": "LONG",
                "rsi_lower": float(lower),
                "rsi_upper": 70.0,
            }
        )
    for upper in range(config.RSI_UPPER_RANGE[0], config.RSI_UPPER_RANGE[1] + 1):
        out.append(
            {
                "direction": "SHORT",
                "rsi_lower": 30.0,
                "rsi_upper": float(upper),
            }
        )
    return out


class BayesianStrategyEngine:
    """Walk-forward search over the parameters the executor understands."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.open = df["open"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)

        self.rsi = indicators.wilder_rsi(self.close, config.RSI_PERIOD)
        self.adx = indicators.wilder_adx(self.high, self.low, self.close, 14)
        self.ema200 = indicators.ema_seeded(self.close, 200)
        self.atr = indicators.atr_series(self.high, self.low, self.close, 14)

        # The executor applies its entry filter to every entry before it
        # commits money, so the backtest has to apply it too. Otherwise
        # the search scores a strategy that trades roughly 40% more
        # often than the one that will actually run.
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = np.abs(self.close - self.ema200) / self.atr
        self.entry_filter = (
            (self.adx < config.ENTRY_FILTER_MAX_ADX)
            & (distance >= config.ENTRY_FILTER_MIN_DIST_ATR)
            & np.isfinite(distance)
        )

        # Plain lists for the hot loop: scalar indexing a numpy array is
        # roughly an order of magnitude slower than a list, and the entry
        # machine runs once per candidate per fold.
        self._rsi_l = self.rsi.tolist()
        self._adx_l = self.adx.tolist()
        self._ema_l = self.ema200.tolist()
        self._close_l = self.close.tolist()

        self.candidates = build_candidates()
        self.trials_run = 0

    # ------------------------------------------------------------- entries
    def build_entries(self, params: dict, lo: int, hi: int) -> np.ndarray:
        """Entry signal over bars [lo, hi), faithful to the Go trigger.

        MIRROR of Go SlotMonitor.EvaluateRSITrailingTrigger, including
        the 15-minute arm expiry (one 15m bar) and the ADX/EMA200 trend
        inversion.
        """
        rsi = self._rsi_l[lo:hi]
        adx = self._adx_l[lo:hi]
        ema200 = self._ema_l[lo:hi]
        close = self._close_l[lo:hi]
        n = hi - lo
        entries = np.zeros(n, dtype=bool)
        if n == 0:
            return entries

        is_long = params["direction"] == "LONG"
        # The executor arms a LONG on rsi_lower and a SHORT on rsi_upper.
        threshold = float(params["rsi_lower"] if is_long else params["rsi_upper"])
        reversal = config.RSI_REVERSAL

        armed = False
        armed_long = is_long
        peak = 0.0
        arm_idx = -1

        for i in range(n):
            r = rsi[i]
            if r != r:  # NaN: the indicator does not exist yet
                continue

            # Go: an armed state that has not confirmed within 15 minutes
            # is discarded. One 15m bar is one 15-minute window.
            if armed and (i - arm_idx) > 1:
                armed = False

            if not armed:
                eff_long = is_long
                inverted = False
                a = adx[i]
                e = ema200[i]
                if (
                    config.INVERSION_ENABLED
                    and a == a
                    and a >= config.INVERSION_ADX
                    and e == e
                    and e > 0
                    and close[i] > 0
                ):
                    if (not is_long) and close[i] > e:
                        eff_long, inverted = True, True
                    elif is_long and close[i] < e:
                        eff_long, inverted = False, True

                if eff_long:
                    arm = (r <= threshold) or inverted
                else:
                    arm = (r >= threshold) or inverted

                if arm:
                    armed = True
                    armed_long = eff_long
                    peak = r
                    arm_idx = i
                continue

            if armed_long:
                if r < peak:
                    peak = r
                elif r >= peak + reversal:
                    # The executor runs its entry filter at the moment of
                    # the entry decision, so the filter is applied to the
                    # confirmation bar.
                    if self._filter_ok(i):
                        entries[i] = True
                    armed = False
            else:
                if r > peak:
                    peak = r
                elif r <= peak - reversal:
                    if self._filter_ok(i):
                        entries[i] = True
                    armed = False

        return entries

    def _filter_ok(self, i: int) -> bool:
        """Whether the executor's entry filter would allow this entry."""
        if not config.ENTRY_FILTER_ENABLED:
            return True
        return bool(self.entry_filter[i])

    # ------------------------------------------------------------ evaluate
    def evaluate(self, params: dict, lo: int, hi: int, order: str = "fav") -> dict:
        """Simulate the strategy over bars [lo, hi) and summarise it."""
        if hi <= lo:
            stats = exit_policy.trade_stats([])
            stats["entries"] = 0
            return stats
        entries = self.build_entries(params, lo, hi)
        results = exit_policy.simulate_trades(
            self.high[lo:hi],
            self.low[lo:hi],
            self.close[lo:hi],
            entries,
            params["direction"] == "LONG",
            order=order,
        )
        stats = exit_policy.trade_stats(results)
        stats["entries"] = int(entries.sum())
        return stats

    # ------------------------------------------------------------- search
    def _optimise(self, lo: int, hi: int):
        """Exhaustive grid over the 52 executable parameter sets.

        Returns (best_params, best_stats, evaluated_count).

        The best candidate is chosen by evidence (t-statistic), NOT by
        whether it happens to be profitable. The profitability gates
        live in one place, in the validation section, so that there is
        exactly one place where a strategy can be rejected and the
        operator can always see what the best available option was.
        """
        best_params = None
        best_stats = None
        best_key = None
        for params in self.candidates:
            self.trials_run += 1
            stats = self.evaluate(params, lo, hi)
            if stats["n"] < config.MIN_TRAIN_TRADES:
                continue
            # Rank by the t-statistic (evidence per unit of noise), then by
            # the mean, then by the trade count. The t-statistic divides by
            # the standard error, so it penalises a strategy that was right
            # once, which is exactly what the old Sharpe-on-166-bars
            # objective failed to do.
            key = (stats["t_stat"], stats["mean"], stats["n"])
            if best_key is None or key > best_key:
                best_key = key
                best_params = dict(params)
                best_stats = stats
        return best_params, best_stats, len(self.candidates)

    def _optimise_forward(self, lo: int, hi: int):
        """The candidate with the best FORWARD return over [lo, hi).

        Selection happens on the development window only, so scoring it
        on the holdout afterwards stays unbiased. This is not the
        strategy that gets published -- it is the one that shows how
        much edge the signal actually has, and therefore whether a
        rejection is the signal's fault or the exit ladder's.
        """
        best_params = None
        best_stats = None
        for params in self.candidates:
            entries = self.build_entries(params, lo, hi)
            fwd = exit_policy.forward_returns(
                self.close[lo:hi], entries, params["direction"] == "LONG"
            )
            stats = exit_policy.trade_stats(fwd)
            if stats["n"] < config.MIN_TRAIN_TRADES:
                continue
            if best_stats is None or stats["mean"] > best_stats["mean"]:
                best_stats = stats
                best_params = dict(params)
        return best_params, best_stats

    # ------------------------------------------------------------- report
    def find_winner(self, pair: str = ""):
        """Run the whole procedure and decide whether anything is worth trading.

        Structure:
          development set = first (1 - HOLDOUT_FRACTION) of the sample
          holdout        = the rest, used ONCE and never for selection

        Inside the development set a walk-forward loop measures how the
        procedure behaves out of sample. The final parameters are then
        refit on the whole development set and scored on the holdout.
        """
        total = len(self.df)
        if total < 500:
            return None, {
                "status": "insufficient_data",
                "bars": total,
                "detail": f"need at least 500 bars, got {total}",
            }

        holdout_bars = int(total * config.HOLDOUT_FRACTION)
        dev_end = total - holdout_bars
        if dev_end < 300 or holdout_bars < 100:
            return None, {
                "status": "insufficient_data",
                "bars": total,
                "detail": "not enough bars to split into development and holdout",
            }

        # ---- inner walk-forward inside the development set -------------
        unit = dev_end // (config.N_FOLDS + 1)
        folds = []
        for f in range(config.N_FOLDS):
            train_end = (f + 1) * unit
            oos_end = min(dev_end, train_end + unit)
            if oos_end <= train_end:
                continue
            params, train_stats, _count = self._optimise(0, train_end)
            if params is None:
                print(
                    f"    fold {f}: no parameter set produced "
                    f"{config.MIN_TRAIN_TRADES}+ trades on the training window"
                )
                continue
            stats = self.evaluate(params, train_end, oos_end)
            folds.append(
                {
                    "fold": f,
                    "params": params,
                    "stats": stats,
                    "train_stats": train_stats,
                }
            )

            passed = stats["n"] >= config.MIN_FOLD_TRADES and stats["mean"] > 0
            print(
                f"    fold {f}: {params['direction']:<5} "
                f"rsi={params['rsi_lower']:.0f}/{params['rsi_upper']:.0f} "
                f"-> OOS n={stats['n']:>3} mean={stats['mean']:>6.2f} "
                f"t={stats['t_stat']:>6.2f}  {'PASS' if passed else 'fail'}"
            )

        if not folds:
            return None, {
                "status": "no_folds",
                "bars": total,
                "detail": (
                    "no walk-forward fold produced a parameter set with "
                    f"{config.MIN_TRAIN_TRADES}+ trades"
                ),
            }

        # The median fold is chosen with the metrics AND the parameters
        # sorted together. The old code sorted only the metrics and then
        # indexed the unsorted parameter list, so the parameters it
        # published belonged to a different fold than the numbers it
        # quoted.
        ranked = sorted(folds, key=lambda f: f["stats"]["t_stat"])
        median_fold = ranked[len(ranked) // 2]

        folds_pass = sum(
            1
            for f in folds
            if f["stats"]["n"] >= config.MIN_FOLD_TRADES and f["stats"]["mean"] > 0
        )
        pooled_trades = sum(f["stats"]["n"] for f in folds)
        pooled_mean = (
            sum(f["stats"]["total"] for f in folds) / pooled_trades
            if pooled_trades
            else 0.0
        )

        # ---- final parameters, refit on the whole development set ------
        final_params, dev_stats, _count = self._optimise(0, dev_end)
        if final_params is None:
            return None, {
                "status": "no_candidate",
                "bars": total,
                "detail": (
                    "no parameter set produced "
                    f"{config.MIN_TRAIN_TRADES}+ trades on the development set"
                ),
            }

        # ---- the holdout, scored once ---------------------------------
        holdout = self.evaluate(final_params, dev_end, total)
        holdout_adv = self.evaluate(final_params, dev_end, total, order="adv")

        # The forward return is the entry's own evidence, with no exit
        # rule in the way. When the ladder rejects a strategy whose
        # forward return is solidly positive, the problem is the EXIT,
        # not the entry, and the operator needs to be told which one it
        # is. The gate itself still uses the ladder, because the ladder
        # is what production runs: publishing a strategy that only
        # works without it would be the same lie in a new place.
        holdout_entries = self.build_entries(final_params, dev_end, total)
        forward = exit_policy.forward_returns(
            self.close[dev_end:total],
            holdout_entries,
            final_params["direction"] == "LONG",
        )
        forward_stats = exit_policy.trade_stats(forward)

        # The best candidate by FORWARD return, selected on the
        # development window and then scored on the holdout. If this one
        # has a solid edge that the ladder destroys, the binding
        # constraint is the exit policy, not the signal.
        fwd_params, fwd_dev_stats = self._optimise_forward(0, dev_end)
        forward_best = None
        if fwd_params is not None:
            fwd_hold_entries = self.build_entries(fwd_params, dev_end, total)
            fwd_hold = exit_policy.forward_returns(
                self.close[dev_end:total],
                fwd_hold_entries,
                fwd_params["direction"] == "LONG",
            )
            fwd_hold_stats = exit_policy.trade_stats(fwd_hold)
            forward_best = {
                "params": fwd_params,
                "dev_forward": fwd_dev_stats,
                "holdout_forward": fwd_hold_stats,
                "holdout_ladder": self.evaluate(fwd_params, dev_end, total),
            }

        n_candidates = len(self.candidates)
        critical = critical_t(n_candidates, config.ALPHA)
        required_t = max(config.MIN_T, critical) if config.REQUIRE_DEFLATED else config.MIN_T

        reasons = []
        if dev_stats is not None and dev_stats["mean"] <= 0:
            reasons.append(
                f"development mean is {dev_stats['mean']:.2f} gross (not positive)"
            )
        if holdout["n"] < config.MIN_VALIDATION_TRADES:
            reasons.append(
                f"holdout has only {holdout['n']} trades "
                f"(need {config.MIN_VALIDATION_TRADES})"
            )
        if holdout["mean"] <= 0:
            reasons.append(f"holdout mean is {holdout['mean']:.2f} gross (not positive)")
        if holdout["t_stat"] < required_t:
            reasons.append(
                f"holdout t {holdout['t_stat']:.2f} < required {required_t:.2f}"
            )
        if folds_pass < config.REQUIRED_FOLD_PASS:
            reasons.append(
                f"only {folds_pass}/{len(folds)} walk-forward folds were profitable "
                f"(need {config.REQUIRED_FOLD_PASS})"
            )
        if pooled_trades < config.MIN_POOLED_TRADES:
            reasons.append(
                f"pooled out-of-sample sample is {pooled_trades} trades "
                f"(need {config.MIN_POOLED_TRADES})"
            )

        # If the entry is healthy and only the ladder is not, say so
        # explicitly. This is the difference between "search harder for
        # a better signal" and "the executor's exit policy is the
        # binding constraint", and they call for completely different
        # work.
        exit_is_the_problem = False
        if forward_best is not None:
            fb = forward_best["holdout_forward"]
            fb_ladder = forward_best["holdout_ladder"]
            exit_is_the_problem = (
                fb["n"] >= config.MIN_VALIDATION_TRADES
                and fb["mean"] > config.FEE_GROSS
                and fb["t_stat"] >= config.MIN_T
                and fb_ladder["mean"] <= 0
            )
            if exit_is_the_problem:
                reasons.append(
                    "the ENTRY has edge ("
                    f"{forward_best['params']['direction']} "
                    f"rsi_lower={forward_best['params']['rsi_lower']:.0f}: forward "
                    f"{fb['mean']:.2f} gross, t {fb['t_stat']:.2f}) but the production "
                    f"EXIT ladder turns it into {fb_ladder['mean']:.2f} -- fix the exit "
                    "geometry in the executor, not the signal"
                )

        report = {
            "status": "winner" if not reasons else "rejected",
            "pair": pair,
            "bars": total,
            "dev_bars": dev_end,
            "holdout_bars": holdout_bars,
            "params": final_params,
            "dev_stats": dev_stats,
            "median_fold_params": median_fold["params"],
            "median_fold_t": median_fold["stats"]["t_stat"],
            "folds": [
                {
                    "fold": f["fold"],
                    "params": f["params"],
                    "n": f["stats"]["n"],
                    "mean": f["stats"]["mean"],
                    "t_stat": f["stats"]["t_stat"],
                }
                for f in folds
            ],
            "folds_pass": folds_pass,
            "pooled_trades": pooled_trades,
            "pooled_mean": pooled_mean,
            "holdout": holdout,
            "holdout_pessimistic": holdout_adv,
            "holdout_forward": forward_stats,
            "forward_best": forward_best,
            "exit_is_the_problem": exit_is_the_problem,
            "candidates": n_candidates,
            "evaluations": self.trials_run,
            "critical_t": critical,
            "required_t": required_t,
            "reasons": reasons,
        }
        return (final_params if not reasons else None), report
