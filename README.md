# TethGard Autoresearch — the signal feeder

This service searches for RSI mean-reversion parameters and publishes the
ones that survive a real validation to the Go executor
(`tethgard-executor-go`), which trades them on OKX.

It runs as a Railway cron job. Every cycle it:

1. fetches 180 days of 15m candles for each pair from **OKX** (the same
   exchange and the same instrument the executor trades — the OKX
   perpetual, not spot),
2. runs a walk-forward search over the **52 parameter sets the executor
   can actually execute**,
3. publishes a strategy **only if it survives the gates**, and
4. pushes a neutral row for every symbol that failed, so a stale
   strategy stops trading.

The previous version of this service did none of steps 3 and 4 honestly.
It published whatever the search happened to like. This document records
what was wrong, what changed, and what the measurements actually say.

---

## The contract with the executor

The executor stores a strategy as five fields and ignores everything else:

```go
type ActiveStrategy struct {
    Symbol    string  // "BTCUSDT"
    Direction string  // "LONG" | "SHORT" (BUY/SELL are normalised)
    Leverage  string  // "3x"
    RSILower  float64 // LONG arms when liveRSI <= RSILower
    RSIUpper  float64 // SHORT arms when liveRSI >= RSIUpper
    UpdatedAt string  // ISO-8601; must be fresh (48 h) or the row cannot arm
    Retired   bool    // explicit retirement flag
}
```

`encoding/json` silently drops unknown fields. That is the whole reason
this service now searches only over `direction` and one RSI threshold:
**anything the executor discards must not be in the search space**, or the
backtest validates a robot that never runs.

Two fields are part of the *safety* contract, not the signal:

* `retired: true` for every symbol that fails validation. A `LONG` with
  `rsi_lower = 0` is **not** safe on its own — the executor's RSI can return
  exactly `0.0` on a pure downtrend, so the marker shape alone can arm. The
  explicit flag plus the executor's `CanArm()` guard is what makes retirement
  absolute.
* `updated_at` must be a current ISO-8601 timestamp. The executor expires rows
  older than 48 h and treats an empty/unparseable timestamp as expired. A
  cycle that publishes nothing therefore does **not** leave a tradeable row
  behind.

What the executor actually does with a signal:

| stage | behaviour | mirrored here |
|---|---|---|
| arm | LONG: `liveRSI <= rsi_lower`; SHORT: `liveRSI >= rsi_upper` | yes |
| confirm | RSI must reverse 2.0 points from the arm extreme, within 15 minutes | yes |
| invert | if ADX(14) >= 25 and price is on the far side of EMA200, the direction flips | yes |
| entry filter | reject if ADX >= 32, or if price is within 1 ATR of EMA200 | yes |
| exit | gross-percent ladder: 3.0 stop, break-even at 2.0 locking 0.6, trail from 3.0 with 0.8 giveback, widen above a 12.0 peak, 480 minute time stop | yes |
| fees | 0.05% of notional per side, i.e. 0.30 gross at 3x, plus 2 bps slippage | yes |

An entry is executed at the close of the bar on which it was confirmed,
and the exit walk starts on the next bar, so no simulation ever uses
information that did not exist when the decision was made.

---

## What was wrong, and what changed

| # | defect | before | after |
|---|---|---|---|
| 1 | the validated entry was not the executed entry | 7 parameters optimised, 3 survived the trip to production; `strategy_type`, `rsi_period`, `stop_loss_pct`, `take_profit_pct` were silently discarded, so a "momentum breakout" winner was executed as the *opposite* mean-reversion trade | the search space is exactly the 52 executable parameter sets |
| 2 | the validated exit was not the executed exit | the backtest used its own 2–4% price stop and 4–6.5% price take-profit; production risks 3.0 **gross** (1% of price at 3x). The geometries differed by 2–4x | `exit_policy.py` mirrors the Go ladder bar for bar |
| 3 | zero transaction costs | `evaluator.py` modelled no fees at all, so every "winner" it found was free money | 0.30 gross per round trip, plus 2 bps slippage |
| 4 | look-ahead in the final validation | the winning parameters were scored on the raw frame instead of the shifted one, so a signal could fill on the bar that produced it | entries are confirmed on closed bars and executed on the next one; `tests/test_pipeline.py` proves causality by truncating the market |
| 5 | the median mixed parameters with metrics | metrics were sorted, then the *unsorted* parameter list was indexed with the median position, so the published parameters belonged to a different fold than the published numbers | metrics and parameters are sorted together |
| 6 | a 10-day window | 1000 bars = 10.4 days. On a 166-bar test fold the standard error of an annualised Sharpe is ±14.5, and the acceptance gate was `oos_sharpe > 0.1` — **0.007 of one standard error** | 180 days, a 45-day holdout scored exactly once, and a t-statistic gate |
| 7 | no multiple-testing correction | 1000 backtests per cycle × 48 cycles per day = 48,000 backtests per day on the same 10-day window, then the maximum was published | 52 candidates, so the multiple-testing critical value is an exact constant (3.10) and is reported on every cycle |
| 8 | a dishonest fallback | when no candidate passed, the old engine fell back to `MA_CROSSOVER` — a strategy the executor cannot execute — and shipped it anyway | there is no fallback. A cycle that finds nothing says so and publishes nothing |
| 9 | stale strategies never expired | the executor's `GetActiveStrategies()` returned every row with no expiry, and the old feeder pushed one winner per cycle, so symbols accumulated strategies from older regimes forever | every symbol in the universe gets a decision each cycle; failures are pushed as a retirement row (`LONG`, `rsi_lower = 0`, `rsi_upper = 100`, `retired: true`). The executor also expires rows older than 48 h. **The threshold shape alone is unsafe** — RSI can be exactly `0.0` — so the explicit flag is mandatory |
| 10 | research halted when the bot was busy | the cycle was skipped whenever 3 slots were occupied, so the research stopped exactly when it mattered | the search always runs; the executor's own guards still protect the account |

### Why the old search "worked"

Re-implementing the old selection logic and feeding it a random walk with
zero drift: **30 out of 30 runs declared a winner**, with a median
out-of-sample Sharpe of 15.6 and 48% of pairs above 1.0. The number in
`TOURNAMENT WINNER FOUND (OOS Sharpe: X)` carried no information at all;
it selected the luckiest lottery ticket.

That is not an implementation artefact. The standard error of an
annualised Sharpe is `sqrt(35040 / n)`, so on 166 bars it is ±14.5 and a
"Sharpe > 0.1" threshold accepts noise 99.9% of the time. Distinguishing a
true Sharpe of 1.0 from zero at 2σ needs roughly four years of 15m data.

---

## What the measurements actually say

The service now reports what the data says, which for this signal is
**"no winner"** — on BTC, ETH and SOL alike.

### The edge is regime-dependent, not persistent

On a 90-day window the RSI fade looks like a genuine 4σ edge: BTC LONG
`RSI <= 26` has a forward return of +0.66 gross per trade, t = 4.45 over
266 trades, positive in every one of five walk-forward folds.

Extend the window to 216 days and chunk the forward return by 30 days:

```
BTC   +1.73(t 2.9)  -0.03(-0.1)  -1.62(t -3.9)  +0.40(1.1)  +0.82(2.8)  +0.55(2.4)  +0.06(0.2)  +1.62(3.0)
ETH   +0.75(1.3)   +0.37(1.1)   -1.94(t -3.8)  +0.21(0.8)  +1.49(5.2)  +0.51(1.1)  +1.37(2.7)  +0.45(0.6)
```

The edge lives in the mean-reverting regimes and is strongly *negative* in
the trending ones. Over the full 216 days the average forward return is
+0.10 gross on BTC and +0.26 on ETH — **below the 0.30 fee**. The 90-day
"edge" was a regime artefact, and 90 days is exactly the kind of window
that hides one.

### The exit ladder destroys what edge there is

With the production ladder applied, the same signal loses:

| | forward, no exit rule | with the production ladder | the ladder gives up |
|---|---|---|---|
| BTC | +0.72 gross (t 1.63) | −0.34 gross | **1.09 per trade** |
| ETH | +0.74 gross (t 1.76) | −0.30 gross | **1.02 per trade** |

The mechanism is specific. On BTC, 140 of 266 trades were stopped out, and
those trades have a **mean forward return of +1.04 gross** — 92 of them
would have recovered. The ladder's 3.0 gross stop is 1% of price at 3x,
which is inside the noise band of a 15m entry, so it kills the trades that
needed room while the trail only arms at +3.0.

### What that means for priorities

1. **The exit geometry is the binding constraint**, not the signal. It
   costs about 1.0 gross percent per trade. That lives in the executor
   (`internal/strategy/exit_policy.go`) and is tunable by environment
   variable without a rebuild.
2. **The signal has no persistent edge** either. It is a mean-reversion
   bet that only pays in chop. No parameter search can manufacture an edge
   that is not there; that is what this service now exists to prove.
3. **The search window must be long enough to see a trend.** 90 days was
   not enough; 180 is the minimum that exposed the artefact, and even 180
   gives a holdout of ~70-80 trades, which is why the holdout t-statistic
   of the *forward* return is only 1.6-1.8 rather than 2.

---

## Configuration

| variable | default | meaning |
|---|---|---|
| `EXECUTOR_WEBHOOK_URL` | the production executor | where strategies are published |
| `TETHGARD_WEBHOOK_URL` | — | alternative name for the same thing |
| `AUTORESEARCH_SYMBOLS` | the ten pairs in `main.py` | comma separated |
| `AUTORESEARCH_DATA_DAYS` | `180` | history window; below ~180 the regime artefact hides |
| `AUTORESEARCH_LEVERAGE` | `3.0` | must match the executor |
| `AUTORESEARCH_TAKER_FEE` | `0.0005` | per side, of notional |
| `AUTORESEARCH_SLIPPAGE` | `0.0002` | against the stop fill |
| `AUTORESEARCH_HOLDOUT_FRACTION` | `0.25` | share of the sample never used for selection |
| `AUTORESEARCH_MIN_VALIDATION_TRADES` | `10` | minimum holdout trades |
| `AUTORESEARCH_MIN_T` | `1.96` | holdout t-statistic required |
| `AUTORESEARCH_REQUIRE_DEFLATED` | `false` | also require the multiple-testing critical value |
| `AUTORESEARCH_ENTRY_FILTER` | `true` | mirror the executor's entry filter |
| `AUTORESEARCH_RETIRE_FAILED` | `true` | push a retirement row for symbols that failed (including a failed data fetch) |
| `AUTORESEARCH_FOLDS` | `5` | walk-forward folds inside the development set |
| `AUTORESEARCH_INVERSION` | `true` | mirror the executor's trend inversion |

### How a retired symbol becomes armed again

Re-arming never weakens the executor's guard. A later cycle that finds a
winner pushes a **fresh payload** with `retired: false`, reachable thresholds
and a current `updated_at`; the executor's `DeriveRetired()` clears the flag
and the row is UPSERTed over the old marker. A cycle with no winner publishes
a retirement row instead, so a symbol is either currently armed or explicitly
inactive — never silently left on yesterday's parameters.

A failed OHLCV fetch is recorded in `decisions` **even when retirement
pushing is disabled**, so the "symbols scanned" tally stays honest. The old
code skipped the symbol entirely and hid the failure.

### Gold uses the perpetual, not the spot token

`DEFAULT_PAIRS` uses `XAU/USDT`. OKX lists `XAU-USDT-SWAP` (the perpetual
the executor trades) and `XAUT-USDT` (a **spot** instrument), but there is no
`XAUT-USDT-SWAP` — requesting its candles returns `51001`. The old
`"XAUT/USDT"` entry therefore never fetched a single bar, hit the
"no data, skipping" branch on every run, and no row was ever published for
gold. `executor_symbol()` also collapses any legacy `XAUT*` spelling to
`XAU*` so the duplicate database row cannot come back.

A failed data fetch is now treated as a retirement decision, not a silent
skip: an unvalidated symbol must not keep trading on yesterday's row. The
next successful cycle re-arms it by pushing a fresh winner payload with
`retired: false`.

Verified live against OKX: `XAU-USDT-SWAP` candles return `code 0`, while
`XAUT-USDT-SWAP` returns `51001` ("Instrument ID ... doesn't exist").

---

## Running it

```bash
pip install -r requirements.txt

python main.py --dry-run                 # search and report, publish nothing
python main.py --dry-run --symbols BTC/USDT
python main.py                           # publish to the executor

python tests/test_pipeline.py            # 23 tests, no pytest needed
python research/entry_vs_exit.py         # the entry-vs-exit decomposition
```

### Verification (2026-11-09 audit)

| Check | Command | Result |
|---|---|---|
| Syntax | `python -m py_compile main.py config.py tests/test_pipeline.py` | ✅ |
| Test suite | `python tests/test_pipeline.py` | ✅ `23/23 passed` |
| Payload contract | `test_payloads_express_retirement_and_re_arming_explicitly` | ✅ `retired` + fresh `updated_at` on both winner and retirement rows |
| Gold instrument | `test_default_pairs_use_the_instrument_that_actually_exists` | ✅ `XAU/USDT`, `XAUT*` collapsed |
| Executor parity | `test_rsi_matches_go`, `test_rsi_boundaries`, `test_simulate_stops_out_and_locks_profit` | ✅ |

The runner in `tests/test_pipeline.py` needs **no pytest**. Local development
requires `numpy`, `pandas` and `requests`; `ccxt` is not required to inspect
market data because OKX's public REST endpoints are used directly.

### A note on the cron cadence

The old service was scheduled every 30 minutes. With a 180-day window,
two consecutive cycles are looking at essentially the same data: a
30-minute gap adds two bars out of 20,736, i.e. 0.01% new information,
while the 48,000-backtest multiple-testing problem grows with every cycle
that re-scores the same market.

**Run it daily**, and prefer to let the holdout move forward instead of
re-using the same window. The cadence is configured in the Railway
dashboard, not in this repository.

---

## Files

| file | role |
|---|---|
| `config.py` | every constant, and which executor behaviour it mirrors |
| `indicators.py` | bit-for-bit ports of the Go RSI, ADX, ATR and EMA200 |
| `exit_policy.py` | Python mirror of the Go exit ladder, plus the forward-return measure |
| `strategy_engine.py` | the entry model, the walk-forward search, the gates |
| `main.py` | the cycle: fetch, search, publish, retire |
| `tests/test_pipeline.py` | indicator parity, exit arithmetic, causality, the median fix |
| `research/entry_vs_exit.py` | separates the entry's edge from the exit's damage |

`program.md`, `strategy_candidate.py` and `evaluator.py` are inherited from
the upstream Karpathy fork and are **not** part of the cron cycle. They
describe an agent loop that was never wired into production.

---

## Recent hardening / changelog

| Commit | What |
|---|---|
| `bd552b9` | Retirement uses an explicit `retired: true` flag, not just an unreachable threshold; winner payloads express `retired: false` + fresh `updated_at` |
| `b2f3b98` | `DEFAULT_PAIRS` uses `XAU/USDT` (the OKX perpetual); `executor_symbol()` collapses `XAUT*` → `XAU*`; failed fetches retire the symbol and are always recorded in `decisions` |
| *this change* | Documentation brought in line with the executor's 48 h freshness gate, `retired` flag, webhook payload validation, single-writer trade journal and the full audit/verification matrix |

Executor side (see `tethgard-multi-go/README.md`): the same audit fixed
`POST /webhook/strategy` accepting junk payloads, corrected `.env.example`,
and removed the duplicate trade-log importer. This repo's branch is
**`master`**, not `main`:

```bash
git push origin master
git ls-remote origin HEAD
```
