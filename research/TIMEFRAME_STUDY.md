# Timeframe study — 21 Nov 2026

## Question

Would a different candle timeframe produce a winning strategy that clears the
**existing** gates? This is not a gate-relaxation exercise: `MIN_T=1.96`,
`REQUIRED_FOLD_PASS=4`, `MIN_VALIDATION_TRADES=10`, `MIN_POOLED_TRADES=40`
and the entry filter / exit policy were held constant. Only the timeframe and
the horizon derived from it were varied.

## Method

`research/timeframe_study.py` fetches OKX OHLCV for the 11 production symbols
at 15m, 1h, 4h and 1d, then runs the exact `BayesianStrategyEngine` whose
entries mirror the Go executor's RSI-trailing arming logic. Each timeframe got
180 days of history.

## Result

| Timeframe | Winners | Best holdout t | Note |
|---|---|---|---|
| 15m | 0 | 0.584 (XRP) | 48 trades, dev mean negative |
| 1h | 0 | 2.151 (AVAX) | **only 2 trades** — statistically meaningless |
| 4h | 0 | 2.681 (ADA) | **13 trades**, 0/3 folds — below sample floor |
| 1d | 0 | — | 180 bars total, engine cannot form folds |

The headline t-statistics above 1.96 are artifacts of tiny samples. A t of
2.68 on 13 trades is not evidence; it is one lucky run. None of the
timeframes produced a candidate that simultaneously had positive development
mean, ≥10 holdout trades, ≥40 pooled OOS trades, t ≥ 1.96 and ≥4/5 profitable
folds.

## Conclusion

The binding constraint is **not the timeframe and not the gate threshold**.
It is the signal itself: RSI-threshold + 2-point reversal + ADX/EMA200 filter,
scored against a 3% gross stop with no take-profit, has no measurable edge on
these symbols in this period. Changing the timeframe from 15m to 1h/4h only
reduces the sample size until a few lucky trades produce a misleading t-stat.

What this means operationally:

1. Do not lower gates to "make a winner appear" — that is how you arm a
   strategy that the data says loses money.
2. Keep real-money arming strict and off.
3. The value of the current system is the **paper observation lane**: it
   exercises the full execution path and collects forward evidence. Its rows
   are labelled `smoke_test`/`shadow`, never counted as winners.
4. A genuine fix requires a different signal family (e.g. trend-following
   with ATR-normalised stops, or a breakout with volume confirmation) that
   the executor can actually receive — that is a separate research project,
   not a threshold tweak.

Reproduce: `python research/timeframe_study.py` (needs ccxt, pandas, numpy).
Raw results: `research/timeframe_study_results.json`.
