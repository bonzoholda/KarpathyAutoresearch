# New signal family study — 21 Nov 2026

## Question

Can trend-following / breakout + volume replace the RSI-reversal signal and
produce a strategy that actually clears the existing, unchanged gates?

## Method

`research/signal_family_study.py` backtested four families on 15m OKX data,
180 days, 11 production symbols, with the exact production exit ladder and
entry filter:

| Family | Parameter sets |
|---|---|
| EMA crossover + ADX | fast/slow 9/21 @ ADX≥20, 20/50 @ ADX≥25 |
| Donchian breakout + volume | lookback 20/10, volume 1.5×/2.0× SMA20 |
| Volume spike + RSI | volume 1.5×/2.0×, RSI>55/60 |
| Momentum (ROC) | lookback 10 @ 2%, 20 @ 4% |

Total: 8 param sets × 11 symbols × 2 directions = **176 comparisons**.
A naive full-sample scan of 176 comparisons needs Bonferroni t ≈ 3.79, not 1.96.

## Result

| Family | Passed t≥1.96 & n≥10 | Best in-sample t |
|---|---|---|
| EMA crossover | 0 | 1.52 (ETH SHORT, n=21) |
| Breakout + volume | 0 | 0.08 |
| Volume + RSI | 0 | −0.15 |
| Momentum ROC20 4% | 1 | 2.23 (NEAR SHORT, n=348) |

## The one apparent winner was an overfit artifact

`research/validate_momentum.py` re-tested NEAR/SHORT ROC20 4% with a proper
time-ordered development / walk-forward / 25% holdout split:

| Metric | Full sample (misleading) | Held-out (honest) |
|---|---|---|
| t-stat | 2.23 | **0.42** |
| mean gross | positive | +0.134 |
| trades | 348 | 121 |
| profitable folds | — | **2 / 5** |

It fails the naive gate (t 0.42 < 1.96) and fails Bonferroni (0.42 < 3.79).
The full-sample t of 2.23 was purely selection bias from scanning 176 variants.

## Verdict

**Do not implement any of these families.** Every family failed when scored
honestly. The one that looked good collapsed out-of-sample. Implementing it
would mean arming a strategy whose own new data says it has no edge —
exactly the failure mode we spent three weeks removing.

This is not a reason to give up. It is a measurement: simple technical
signals on these 11 liquid symbols at 15m, with this cost model and exit
ladder, do not have a durable edge in this period. A real edge would need
something structurally different — a genuinely different market (higher
volatility names, different sessions), a different execution style
(maker entries, wider ATR-normalised stops), or a non-price signal. Those
are separate, larger research projects and none is guaranteed to work.

## Follow-up (22 Nov 2026)

`research/COMBINED_LANE_STUDY.md` answers whether a trend lane can be combined
with the existing RSI lane. Twelve pre-registered trend candidates were
selected on development only and scored once on holdout: the best dev t was
1.137 (below the naive 1.96), and on holdout it collapsed to t −5.815. The
regime-gated combination remained loss-making (t −7.811). There is currently
no holdout-passing trend signal to combine.

## Files

- `research/combined_lane_study.py` — pre-registered combined-lane test
- `research/combined_lane_results.json` — combined-lane results
- `research/signal_family_study.py` — broad sweep
- `research/signal_family_results.json` — raw sweep output
- `research/validate_momentum.py` — honest holdout validation
- `research/validate_momentum_results.json` — validation output
