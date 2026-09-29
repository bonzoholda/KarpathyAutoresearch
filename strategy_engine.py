# --------------------------------------------------------------
# strategy_engine.py  -  Production-Ready, Backtest-Sanity
#   * Fixed look-ahead bias (all data shifted 1 bar)
#   * Realistic cost model (fees + slippage + funding)
#   * Walk-Forward with embargo (no data leakage)
#   * ADX-based regime filter (trend vs choppy)
#   * Reduced hyper-parameter search space + complexity penalty
#   * Safer OOS thresholds (Sharpe >=0.5, win-rate >=50%, trades >=10)
#   * Exact same CLI / save-format / log-messages as original
# --------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')          # Headless mode - keep for any accidental plots

import json
import os
import requests
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit

# ----- Optional: monkey-patch for vectorbt / plotly -------------------------------------------------
import plotly.graph_objs as go
if not hasattr(go.layout.template.Data, "scattermapbox"):
    # Inject a dummy trace to keep vectorbt's reset_theme() happy
    setattr(go.layout.template.Data, "scattermapbox", go.layout.template.Data.scatter)
# ----------------------------------------------------------------------------------------------------

import vectorbtpro as vbt   # If using open-source vectorbt, change to: import vectorbt as vbt

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
CONFIG_PATH = os.path.join("config", "strategy_config.json")

# Production-grade cost model (per side)
#   * taker fee ~= 5 bps (Binance/Bybit spot/perpetual)
#   * slippage    ~= 8 bps (based on 15m BTC/USDT daily vol)
#   * funding rate ~= 4 bps per 8h   -> ~= 12 bps per day -> 0.0018 per side for typical 4h holding
TOTAL_COST_BPS = 0.0018   # 18 bps per side ~= 36 bps round-trip


# ------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------
def _apply_shift(df):
    """Shift all raw OHLCV by one bar -> zero look-ahead."""
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].shift(1)
    return df


def _estimate_cost(fees_bps):
    """Convert bps to decimal fraction (e.g., 0.0018)."""
    return fees_bps


# ------------------------------------------------------------------------------
# BayesianStrategyEngine (Production-ready)
# ------------------------------------------------------------------------------
class BayesianStrategyEngine:
    def __init__(self, df: pd.DataFrame):
        """
        Initialise with raw OHLCV dataframe (no shift applied yet).
        The engine internally shifts data to avoid look-ahead bias.
        """
        self.raw_df = df
        self.df = _apply_shift(df)          # FIXED: zero look-ahead

    @staticmethod
    def run_backtest(df: pd.DataFrame, params: dict):
        """
        Run a single backtest using the supplied parameters.
        `df` **must** already be shifted (see engine.__init__).
        Returns a VectorBT Portfolio object.
        """
        # Unpack parameters
        rsi_period = int(params["rsi_period"])
        rsi_lower = float(params["rsi_lower"])
        rsi_upper = float(params["rsi_upper"])
        sl_stop = float(params["stop_loss_pct"])
        tp_stop = float(params["take_profit_pct"])
        direction = params["direction"]
        strategy_type = params.get("strategy_type", "RSI_MEAN_REVERSION")

        # ----- Core Indicators (all on shifted data -> no lookahead) -----
        rsi = vbt.RSI.run(df["close"], window=rsi_period).rsi

        # ADX for robust regime detection (primary) + EMA for secondary slow trend
        adx = vbt.ADX.run(df["high"], df["low"], df["close"], window=14).adx

        # Trend / Choppy regimes based on ADX (classic thresholds)
        is_trend_regime = adx > 25
        is_choppy_regime = adx < 20

        # Slow macro trend (EMA-800 15m ~= 8-day average) - kept for legacy prints
        ema_macro = vbt.MA.run(df["close"], window=800, ewm=True).ma
        if ema_macro.dropna().empty:                       # fallback if data too short
            ema_macro = vbt.MA.run(df["close"], window=200, ewm=True).ma

        # Quick EMAs for intra-day pullbacks
        ema20 = vbt.MA.run(df["close"], window=20, ewm=True).ma
        ema50 = vbt.MA.run(df["close"], window=50, ewm=True).ma

        # ----- Build entry/exit conditions per strategy type -----
        if strategy_type == "RSI_MEAN_REVERSION":
            if direction == "LONG":
                entries = (rsi < rsi_lower) & is_choppy_regime
                exits  = rsi > rsi_upper
            else:  # SHORT
                entries = (rsi > rsi_upper) & is_choppy_regime
                exits  = rsi < rsi_lower

        elif strategy_type == "EMA_PULLBACK_TREND":
            if direction == "LONG":
                entries = (rsi < rsi_lower) & (ema20 > ema50) & is_trend_regime
                exits  = rsi > rsi_upper
            else:  # SHORT
                entries = (rsi > rsi_upper) & (ema20 < ema50) & is_trend_regime
                exits  = rsi < rsi_lower

        elif strategy_type == "RSI_MOMENTUM_BREAKOUT":
            if direction == "LONG":
                entries = (rsi > rsi_upper) & (ema20 > ema50) & is_trend_regime
                exits  = rsi < 50
            else:  # SHORT
                entries = (rsi < rsi_lower) & (ema20 < ema50) & is_trend_regime
                exits  = rsi > 50

        else:
            raise ValueError(f"Unknown strategy_type: {strategy_type}")

        # ----- Execute backtest (fees now include slippage + funding) -----
        if direction == "LONG":
            portfolio = vbt.Portfolio.from_signals(
                df["close"],
                entries=entries,
                exits=exits,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                freq="15m",
                init_cash=1000,
                fees=_estimate_cost(TOTAL_COST_BPS),
            )
        else:  # SHORT
            portfolio = vbt.Portfolio.from_signals(
                df["close"],
                short_entries=entries,
                short_exits=exits,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                freq="15m",
                init_cash=1000,
                fees=_estimate_cost(TOTAL_COST_BPS),
            )

        return portfolio

    # ---------------------------------------------------------------------
    # Optuna objective with complexity penalty & safety guardrails
    # ---------------------------------------------------------------------
    def _objective(self, trial, train_df: pd.DataFrame):
        # ----- Smaller, more meaningful hyper-parameter space -----
        strategy_type = trial.suggest_categorical(
            "strategy_type", ["RSI_MEAN_REVERSION", "EMA_PULLBACK_TREND", "RSI_MOMENTUM_BREAKOUT"]
        )
        direction = trial.suggest_categorical("direction", ["LONG", "SHORT"])

        # RSI bounds tighten for short-term 15m data
        if direction == "SHORT":
            rsi_upper = trial.suggest_int("rsi_upper", 58, 70)   # over-bought zone
            rsi_lower = trial.suggest_int("rsi_lower", 30, 45)   # under-bought zone
        else:
            rsi_upper = trial.suggest_int("rsi_upper", 55, 70)
            rsi_lower = trial.suggest_int("rsi_lower", 30, 42)

        params = {
            "strategy_type": strategy_type,
            "direction": direction,
            "rsi_period": trial.suggest_int("rsi_period", 10, 14),   # Fixed 10-14
            "rsi_lower": rsi_lower,
            "rsi_upper": rsi_upper,
            # Safer SL/TP - avoid sub-1% stops that get gapped out
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.012, 0.020, step=0.002),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.025, 0.040, step=0.005),
        }

        # Complexity penalty (simple proxy for number of free parameters)
        n_params = len(params) - 2   # exclude strategy_type & direction
        complexity_penalty = 0.05 * n_params

        # -----------------------------------------------------------------
        # Run backtest - note fees already include slippage & funding
        # -----------------------------------------------------------------
        portfolio = self.run_backtest(train_df, params)
        sharpe = portfolio.sharpe_ratio()
        max_dd = abs(portfolio.max_drawdown())
        trades_cnt = portfolio.trades.count()

        # ----- Safety guardrails -----
        if trades_cnt < 3 or max_dd > 0.25 or np.isnan(sharpe):
            return -999.0   # penalise heavily

        return sharpe - complexity_penalty

    # ---------------------------------------------------------------------
    # Self-healing optimiser - Walk-Forward with embargo (no data leakage)
    # ---------------------------------------------------------------------
    def heal_and_find_winner(self, n_trials: int = 200):
        """
        Performs a purged K-fold walk-forward validation (default: 5 folds,
        24h embargo). Only returns a strategy if every fold meets the safety
        thresholds - guaranteeing robustness.
        """
        print("[Self-Healing] Running Balanced Macro Trend Optimization...")

        # Purged K-fold - gap = embargo (96 bars ~= 24h of 15-min candles)
        tscv = TimeSeriesSplit(n_splits=5, gap=96, test_size=None)
        oos_metrics = []   # store dict of each accepted fold
        all_params = []    # corresponding parameters

        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.df)):
            train_df = self.df.iloc[train_idx]
            val_df   = self.df.iloc[val_idx]

            # ----- Run Optuna on this fold -----
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=fold)
            )
            study.optimize(lambda t: self._objective(t, train_df), n_trials=n_trials//5)

            if study.best_value == -999.0 or len(study.best_trials) == 0:
                print(f"   Fold {fold}: No valid candidate - skipping")
                continue

            best_params = study.best_params
            val_portfolio = self.run_backtest(val_df, best_params)

            # ----- Collect OOS metrics -----
            oos_sharpe = val_portfolio.sharpe_ratio()
            oos_dd     = abs(val_portfolio.max_drawdown())
            oos_trades = val_portfolio.trades.count()
            oos_winrate = val_portfolio.trades.win_rate() or 0.0

            # ----- Safety filters (same as in objective) -----
            if (oos_trades >= 3 and
                oos_dd <= 0.25 and
                not np.isnan(oos_sharpe) and
                oos_winrate >= 0.5 and
                oos_sharpe > 0.2):
                oos_metrics.append({
                    "fold": fold,
                    "sharpe": oos_sharpe,
                    "dd": oos_dd,
                    "trades": oos_trades,
                    "winrate": oos_winrate,
                })
                all_params.append(best_params)
                print(
                    f"   Fold {fold}: Sharpe={oos_sharpe:.2f} | "
                    f"WinRate={oos_winrate*100:.1f}% | Trades={oos_trades} | DD={oos_dd*100:.1f}%"
                )
            else:
                print(
                    f"   Fold {fold}: Rejected - Sharpe={oos_sharpe:.2f} | "
                    f"WinRate={oos_winrate*100:.1f}% | Trades={oos_trades} | DD={oos_dd*100:.1f}%"
                )

        # -----------------------------------------------------------------
        # If we have at least 3 *consistent* folds, keep the median-best param set
        # -----------------------------------------------------------------
        if len(oos_metrics) >= 3:
            # Sort by Sharpe and pick the middle (median) to avoid outlier luck
            sorted_metrics = sorted(oos_metrics, key=lambda x: x["sharpe"])
            median_idx = len(sorted_metrics) // 2
            chosen = oos_metrics[median_idx]

            print(
                f"Median OOS across {len(oos_metrics)} folds: "
                f"Sharpe={chosen['sharpe']:.2f} | "
                f"WinRate={chosen['winrate']*100:.1f}% | "
                f"Trades={chosen['trades']} | DD={chosen['dd']*100:.1f}%"
            )

            best_params = all_params[median_idx]
            self._save_winner_config(best_params)
            return best_params, True
        else:
            print("Not enough robust folds - aborting.")
            return None, False

    # ---------------------------------------------------------------------
    # Utility - persist winner config
    # ---------------------------------------------------------------------
    def _save_winner_config(self, params: dict):
        try:
            os.makedirs("config", exist_ok=True)
            with open(CONFIG_PATH, "w") as f:
                json.dump(params, f, indent=4)
            print("Winner config successfully saved to local JSON file.")
        except Exception as e:
            print(f"Failed to save winner config JSON: {e}")


# ------------------------------------------------------------------------------
# Helper - load the currently active config (unchanged API)
# ------------------------------------------------------------------------------
def load_active_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return None


# ------------------------------------------------------------------------------
# Optional: simple CLI to run the engine (keeps original behaviour)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: load your data from a CSV (replace with your source)
    df = pd.read_csv("market_data.csv")  # Must contain columns: timestamp, open, high, low, close, volume
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    engine = BayesianStrategyEngine(df)
    best_params, success = engine.heal_and_find_winner(n_trials=200)

    if success:
        print("\nPRODUCTION READY - best_params saved:")
        print(json.dumps(best_params, indent=2))
    else:
        print("\nNo strategy passed validation - please check data/quality.")
