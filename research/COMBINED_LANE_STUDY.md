# Combined-lane study — 22 Nov 2026

## Question

Can the existing RSI-reversal ("legacy") lane be combined with a trend-following
lane that **genuinely passes holdout**, and would the combined system be
effective?

## Why this is not a trivial "OR" on the entry signal

The legacy lane is mean-reversion: it requires `ADX < 32` at confirmation and
rejects trend regimes on purpose. A trend lane needs the opposite regime
(`ADX >= 25`). Putting both into one trigger would produce contradictory
states, and the executor only receives `close + RSI` anyway — EMA/ADX/volume
never reach the arm decision. A real combination has to be **two lanes with
separate parameters, separate regime gates and separate exit geometry**.

## Method (`research/combined_lane_study.py`)

- 11 production symbols, 15m, 180 days (20,736 bars each).
- 12 **pre-registered** trend candidates: 3 families (EMA9/21, Donchian20 +
  volume, ROC20 4%) × 2 exit modes (production ladder, 2×ATR trailing) ×
  2 directions. No per-symbol cherry-picking.
- Indicators computed once on the full series, then sliced. Time-ordered
  75% development / 25% holdout.
- The trend candidate is selected on **development only**, then scored once
  on holdout. Bonferroni critical t for 12 comparisons = **2.836**.
- Legacy lane mirrors `strategy_engine.build_entries` (RSI arm + confirm,
  15m arm expiry, trend inversion, ADX/EMA200 filter) with the production
  exit ladder.
- Combined system: legacy active at `ADX < 25`, trend active at `ADX >= 25`.

## Result

Development period — only 2 of 12 candidates had a positive t, and the best
did not even clear the naive gate:

| Candidate | n | mean gross % | t |
|---|---:|---:|---:|
| ROC20 4% LONG, ATR trail | 2516 | +0.166 | **1.137** |
| ROC20 4% LONG, ladder | 2516 | +0.012 | 0.194 |
| Donchian20 LONG, ATR trail | 2439 | −0.177 | −2.303 |

Selected on dev, scored once on holdout:

| System | Trades | Mean gross % | t | Profit factor |
|---|---:|---:|---:|---:|
| Legacy lane alone | 1329 | −0.724 | −11.329 | 0.467 |
| Selected trend lane | 685 | −1.087 | −5.815 | 0.571 |
| Regime-gated combination | 671 | −0.704 | −7.811 | 0.477 |

The selected trend candidate fails the naive gate (t −5.82 < 1.96), fails
Bonferroni (−5.82 < 2.84), and has negative expectancy on held-out data.
The combined system is *less negative* than the legacy lane alone, but it is
still strongly loss-making. "Less bad than a losing system" is not an
improvement — the verdict field `combined_beats_legacy` is therefore `false`.
The apparent relative gain comes mostly from trading less, not from better
entries.

## Answer

**Architecturally yes, empirically no — not with any candidate we can
currently produce.** There is no trend signal here that "genuinely passes
holdout", so there is nothing valid to combine. Adding a below-gate signal to
the existing lane does not repair its edge; it only adds losing trades and
more implementation surface (ActiveStrategy, trigger, ticker cache, webhook
validation, exit policy).

This matches the live diagnosis: the market has been in a strong trend
(ADX 40–49), which is exactly the regime every RSI-reversal signal is built
to avoid. The entry filter and the research gate are both rejecting the same
regime mismatch, not malfunctioning.

## What would actually be required

A genuine combined system needs, in order:

1. A trend candidate that clears dev **and** holdout with multiple-testing
   correction, on a *pre-registered* (not swept) hypothesis.
2. A different market or execution style — this 15m / 11-symbol / taker-fee
   sample has shown no durable edge for simple price/volume signals.
   Higher-volatility names, maker entries, wider ATR-normalised stops or a
   non-price signal are separate research projects.
3. End-to-end executor changes with independent risk limits per lane.

Until (1) exists, combining is a way to launder a losing signal, not a way to
create an edge.

## Files

- `research/combined_lane_study.py` — pre-registered two-lane holdout test
- `research/combined_lane_results.json` — full scores and verdict
- `research/SIGNAL_FAMILY_STUDY.md` — preceding single-family sweep
- `research/validate_momentum.py` — holdout validation of the prior survivor
