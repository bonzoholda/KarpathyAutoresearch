# Signal discovery program — 22 Nov 2026

## Mandate

Find a profitable, consistent signal mechanism regardless of system,
timeframe or indicator, then design a cron-ready, non-invasive delivery module.

## What was searched (all with time-ordered dev/holdout)

| # | Program | Variants | Outcome |
|---|---|---:|---|
| 1 | `signal_family_study.py` + `validate_momentum.py` | 176 | all failed holdout |
| 2 | `timeframe_study.py` | 4 timeframes | 0 winners |
| 3 | `combined_lane_study.py` | 12 pre-registered | combined still loss-making |
| 4 | `alpha_discovery.py` (21 symbols, 1h, 1y) | 8 hypotheses | best dev t 0.34, holdout negative |
| 5 | `xsection_study.py` (48 rank variants) | 48 | best dev t 0.35, holdout −0.44 |
| 6 | `daily_alpha_study.py` (5y daily) | 8 | dev t 2.17 5/5 folds → holdout −0.20 |
| 7 | `regime_xsection_study.py` | 6 gates | no gate rescued momentum |
| 8 | `funding_study.py` + `validate_funding.py` | 2 | **first positive holdout** |

The repeated pattern: price/volume families produce dev winners that collapse
out-of-sample. This is selection bias, not an implementation bug — the same
gates, cost model and exit logic were used throughout.

## The candidate: funding-percentile SHORT

OKX funding rate is an economically meaningful crowding signal that is not
derivable from OHLCV. After fixing a fee-sign bug that had inflated the first
result, the honest numbers are:

| Metric | Dev | Holdout |
|---|---:|---:|
| n | 1494 | 601 |
| mean gross % | +0.023 | +1.491 |
| t | 0.067 | **2.994** |
| PF | 1.005 | 1.401 |

Validation findings:
- 4/5 positive holdout folds; fold 1 negative, fold 4 dominant.
- Asymmetric: LONG leg (high funding → long) t = −3.34; SHORT leg
  (low funding → short) t = 5.43. Only the SHORT leg is used.
- Threshold sweep q 0.75→0.95: t 3.89→1.96, all positive but decaying.
- Horizon 1d/2d/3d: t 2.99/6.36/7.04 — slower move is better.
- Fee x1/x2/x3: t 2.99/2.39/1.79 — fragile at triple cost.

## Verdict

This is the **best available alternative candidate**, not a confirmed edge.
Development was flat, the holdout is only ~23 days of OKX funding history,
and one fold carries much of the result. It is suitable for a **paper
observation lane** to gather more evidence, not for live capital.

## Delivery design

`alpha_funding/` publishes inert shadow rows through the existing single
`POST /webhook/strategy` door. RSI thresholds 0/100 make arming impossible in
the Go executor, so no Go changes are required. Real funding signals live in
the `research` diagnostic payload and in `signals.jsonl`. Promotion to live
is documented as a separate four-step additive change.

## Files

- `alpha_discovery.py`, `xsection_study.py`, `daily_alpha_study.py`,
  `regime_xsection_study.py`, `funding_study.py`, `validate_funding.py`
- matching `*_results.json` files with raw scores
- `../alpha_funding/README.md` for operations
