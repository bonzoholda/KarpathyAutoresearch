# Funding percentile observation lane

Standalone, non-invasive signal module added to autoresearch. It publishes
through the **same single webhook door** (`POST /webhook/strategy`) and never
changes existing scanner or executor code.

## Signal

OKX perpetual funding rate is a crowding measure that is not derivable from
OHLCV. The validated direction is:

> **SHORT** when the latest funding rate is at or below the `q=0.15` percentile
> of its trailing distribution, hold ~3 days.

The symmetric LONG leg (high funding → long) was **loss-making** in validation
(t = −3.34) and is deliberately never emitted.

## Evidence (honest summary)

From `research/funding_study.py` and `research/validate_funding.py`:

| Metric | Development | Holdout |
|---|---:|---:|
| Trades | 1494 | 601 |
| Mean gross % | +0.023 | **+1.491** |
| t-stat | 0.067 | **2.994** |
| Profit factor | 1.005 | 1.401 |

Robustness: 4/5 positive holdout folds; threshold sweep q=0.75→0.95 stays
positive (t 3.89→1.96); 3-day hold outperformed 1-day (t 7.04 vs 2.99).

**Caveats that matter:**
- Development t was essentially flat (0.067). The holdout is where the edge
  appeared, so this is a *candidate*, not a confirmed repeatable system.
- Funding history from OKX public API covers only ~94 days; holdout is ~23 days.
- One of five holdout folds was negative, and the last fold dominated the
  aggregate.
- At 3× the modelled fee the t-stat drops to 1.79 (below naive significance).

Therefore this module is **paper/observation only**.

## Why it is non-invasive

The Go executor only understands `rsi_lower` / `rsi_upper` as arm thresholds.
A funding signal cannot be expressed in those fields. Every published row is
therefore deliberately inert:

```json
{
  "mode": "shadow",
  "rsi_lower": 0.0,
  "rsi_upper": 100.0,
  "research": { "status": "shadow", "signal": {...}, "evidence": {...} }
}
```

`CanArm()` requires the threshold to be strictly inside `(0, 100)`, so these
rows are stored and visible in the existing UI research panel but **can never
arm a position**. No new endpoint, no Go struct change, no schema change.

Note: publishing uses the same symbol key, so it replaces the main scanner's
row for that symbol. The main scanner currently rejects all symbols, so this
has no live effect. Use `FUNDING_SYMBOLS` to restrict the set if needed.

## Cron usage

```bash
# Every 8 hours, after OKX funding settlement
0 */8 * * * cd /path/to/KarpathyAutoresearch && \
  /path/to/python -m alpha_funding.run >> alpha_funding/cron.log 2>&1
```

Dry run first:

```bash
python -m alpha_funding.run --dry-run
```

To test a subset:

```bash
python -m alpha_funding.run --dry-run --symbols BTC/USDT,ETH/USDT
```

State (`state.json`) prevents duplicate publishes for the same funding
timestamp; `signals.jsonl` is the append-only audit trail.

## Configuration

All env vars are optional:

| Variable | Default | Meaning |
|---|---|---|
| `FUNDING_SYMBOLS` | 13 symbols | comma-separated OKX perpetuals |
| `FUNDING_Q_LO` | `0.15` | lower funding percentile band |
| `FUNDING_LOOKBACK_DAYS` | `90` | trailing history window |
| `FUNDING_MIN_HISTORY` | `20` | minimum observations before firing |
| `FUNDING_HOLD_DAYS` | `3` | documented horizon |
| `EXECUTOR_WEBHOOK_URL` / `TETHGARD_WEBHOOK_URL` | production URL | same door as scanner |

## Promotion to live (separate, opt-in change)

Making this executable requires an **additive** Go extension, deliberately
not done here:

1. Add an optional `funding_signal` field to `database.ActiveStrategy`.
2. Add a branch in `ticker.evaluateArmAt` that arms on a fresh funding
   percentile cross instead of RSI.
3. Add an exit policy for a 3-day hold (current ladder is intraday-sized).
4. Require a second holdout on a larger funding sample before enabling.

Until all four are done, keep `mode="shadow"` and the unreachable thresholds.

## Files

- `alpha_funding/config.py` — standalone config
- `alpha_funding/signal.py` — pure signal math
- `alpha_funding/publisher.py` — single-door payload and audit
- `alpha_funding/run.py` — cron entry point
- `tests/test_alpha_funding.py` — 11 safety tests
