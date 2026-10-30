# --------------------------------------------------------------
# indicators.py - bit-for-bit ports of the Go indicator maths
# --------------------------------------------------------------
#
# These functions are deliberately written as plain loops rather than
# vectorised pandas calls. The executor computes its indicators with a
# specific Wilder recursion, and a backtest that computes a different
# RSI is a backtest of a different strategy. Each function below names
# the Go function it mirrors.
#
#   wilder_rsi        <- internal/ticker/monitor.go calculateRSI
#   wilder_adx        <- internal/strategy/entry_filter.go atrAndADX
#   ema_seeded        <- internal/strategy/entry_filter.go emaLast
# --------------------------------------------------------------
import numpy as np


def wilder_rsi(close, period: int = 14) -> np.ndarray:
    """Wilder RSI, matching Go calculateRSI(candles, period).

    Go seeds with the simple average of the first `period` changes and
    then smooths with avg = (avg*(period-1) + change)/period. The Go
    version returns a single number for the most recent candle; this
    returns the whole series, where element i is what Go would return
    when the ticker was looking at bar i.

    Bars before the seed is available are NaN. Go returns 50.0 when
    there is not enough data; NaN is used here so that an indicator
    that does not exist yet can never be mistaken for a neutral 50.
    """
    c = np.asarray(close, dtype=float)
    n = c.size
    out = np.full(n, np.nan)
    if n <= period:
        return out

    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = c[i] - c[i - 1]
        if change > 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period

    def value(ag: float, al: float) -> float:
        if al == 0:
            return 100.0
        rs = ag / al
        return 100.0 - (100.0 / (1.0 + rs))

    out[period] = value(avg_gain, avg_loss)
    for i in range(period + 1, n):
        change = c[i] - c[i - 1]
        if change > 0:
            avg_gain = (avg_gain * (period - 1) + change) / period
            avg_loss = (avg_loss * (period - 1)) / period
        else:
            avg_gain = (avg_gain * (period - 1)) / period
            avg_loss = (avg_loss * (period - 1) - change) / period
        out[i] = value(avg_gain, avg_loss)
    return out


def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing, matching Go wilderSmooth.

    Seed is the simple average of the first `period` values; each
    subsequent value is (prev*(period-1) + v)/period.
    """
    v = np.asarray(values, dtype=float)
    if v.size < period or period <= 0:
        return np.empty(0, dtype=float)
    smoothed = float(v[:period].sum()) / period
    out = np.empty(v.size - period + 1, dtype=float)
    out[0] = smoothed
    for i in range(period, v.size):
        smoothed = (smoothed * (period - 1) + v[i]) / period
        out[i - period + 1] = smoothed
    return out


def wilder_adx(high, low, close, period: int = 14) -> np.ndarray:
    """ADX, matching Go atrAndADX(bars, atrP=14, adxP=14).

    Returns the ADX series aligned to the input bars; NaN where the
    value is not yet defined.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = c.size
    out = np.full(n, np.nan)
    if n < 2 * period + 1:
        return out

    tr = np.empty(n - 1, dtype=float)
    pdm = np.empty(n - 1, dtype=float)
    mdm = np.empty(n - 1, dtype=float)
    for i in range(1, n):
        prev_c, prev_h, prev_l = c[i - 1], h[i - 1], l[i - 1]
        tr[i - 1] = max(
            h[i] - l[i], abs(h[i] - prev_c), abs(l[i] - prev_c)
        )
        up = h[i] - prev_h
        down = prev_l - l[i]
        pdm[i - 1] = up if (up > down and up > 0) else 0.0
        mdm[i - 1] = down if (down > up and down > 0) else 0.0

    atr = _wilder_smooth(tr, period)
    pdi = _wilder_smooth(pdm, period)
    mdi = _wilder_smooth(mdm, period)
    if atr.size == 0:
        return out

    dx = np.zeros(atr.size, dtype=float)
    for i in range(atr.size):
        if atr[i] <= 0:
            dx[i] = 0.0
            continue
        p = 100.0 * pdi[i] / atr[i]
        m = 100.0 * mdi[i] / atr[i]
        total = p + m
        dx[i] = 0.0 if total == 0 else 100.0 * abs(p - m) / total

    adx = _wilder_smooth(dx, period)
    # Alignment. atrS[j] is the average of trs[j..j+period-1], and trs[i]
    # belongs to original bar i+1, so atrS[j] belongs to bar period+j.
    # dx[j] therefore belongs to bar period+j as well, and adx[k] is the
    # average of dx[k..k+period-1], so it belongs to bar
    # (period+k) + (period-1) = 2*period-1+k. ADX is undefined before that.
    start = 2 * period - 1
    for k in range(adx.size):
        idx = start + k
        if idx < n:
            out[idx] = adx[k]
    return out


def ema_seeded(close, period: int = 200) -> np.ndarray:
    """EMA seeded with the SMA of the first `period` values.

    Matches Go emaLast: seed = mean(vals[:period]), then
    e = v*k + e*(1-k) with k = 2/(period+1).
    """
    c = np.asarray(close, dtype=float)
    n = c.size
    out = np.full(n, np.nan)
    if n < period:
        return out
    ema = float(c[:period].sum()) / period
    out[period - 1] = ema
    k = 2.0 / (period + 1)
    for i in range(period, n):
        ema = c[i] * k + ema * (1.0 - k)
        out[i] = ema
    return out


def atr_series(high, low, close, period: int = 14) -> np.ndarray:
    """ATR series aligned to the bars, matching the ATR inside Go
    atrAndADX. Not used by the entry trigger itself, but needed to
    report the distance of price from EMA200 in ATR units.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = c.size
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    tr = np.empty(n - 1, dtype=float)
    for i in range(1, n):
        tr[i - 1] = max(
            h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])
        )
    smoothed = _wilder_smooth(tr, period)
    for k in range(smoothed.size):
        idx = period + k
        if idx < n:
            out[idx] = smoothed[k]
    return out
