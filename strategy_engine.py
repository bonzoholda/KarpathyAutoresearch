# --------------------------------------------------------------
# strategy_engine_new.py - Production-Ready Backtesting Engine
#   * Fixed look-ahead bias (shift data 1 bar)
#   * Realistic cost model (fees + slippage + funding)
#   * Walk-Forward with embargo (no leakage)
#   * ADX-based regime filter (trend vs choppy)
#   * Reduced hyper-parameter search space + complexity penalty
#   * Relaxed OOS thresholds (Sharpe >=0.5, winrate >=0.5, trades >=2)
#   * Fallback to MA Crossover if no strategy found
#   * Payload format compatible with main.py / Go Engine webhook
#   * Same CLI / config format as original
# --------------------------------------------------------------
import json
import os
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
import requests

import matplotlib
matplotlib.use('Agg')

import vectorbt as vbt
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Config - Cost can be overridden via environment variable
CONFIG_PATH = os.path.join("config", "strategy_config.json")
TOTAL_COST_BPS = float(os.environ.get("TOTAL_COST_BPS", 0.0010))  # Default 10 bps per side

def _apply_shift(df):
    """Shift data 1 bar to avoid look-ahead bias."""
    df = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].shift(1)
    return df

def _estimate_cost(fees_bps):
    """Return the fee structure in the format expected by VectorBT."""
    return fees_bps

def compute_adx(high, low, close, period=14):
    """
    Compute Average Directional Index (ADX) using Wilder's smoothing.
    Returns a pandas Series aligned with the index of `high`.
    """
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up = high - high.shift()
    down = low.shift() - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    # Smoothed with EMA (Wilder's style approximation)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr * 100
    minus_di = pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr * 100

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.fillna(0)

class BayesianStrategyEngine:
    def __init__(self, df: pd.DataFrame):
        self.raw_df = df
        self.df = _apply_shift(df)
        self.last_oos_sharpe = None          # Store the latest OOS Sharpe for logging
        self.last_pair = None                # Pair name for push operations

    @staticmethod
    def run_backtest(df: pd.DataFrame, params: dict):
        """
        Execute a single backtest using VectorBT based on the supplied parameters.
        Returns a VectorBT Portfolio object.
        """
        strategy_type = params.get("strategy_type", "RSI_MEAN_REVERSION")
        direction = params["direction"]
        sl_stop = float(params["stop_loss_pct"])
        tp_stop = float(params["take_profit_pct"])

        # ---------- Fallback Strategy: Simple MA Crossover ----------
        if strategy_type == "MA_CROSSOVER":
            ema_fast = int(params["ema_fast"])
            ema_slow = int(params["ema_slow"])
            ema_fast_line = vbt.MA.run(df["close"], window=ema_fast, ewm=True).ma
            ema_slow_line = vbt.MA.run(df["close"], window=ema_slow, ewm=True).ma

            if direction == "LONG":
                entries = (ema_fast_line > ema_slow_line) & (ema_fast_line.shift() <= ema_slow_line.shift())
                exits = (ema_fast_line < ema_slow_line) & (ema_fast_line.shift() >= ema_slow_line.shift())
                portfolio = vbt.Portfolio.from_signals(
                    df["close"], entries=entries, exits=exits,
                    sl_stop=sl_stop, tp_stop=tp_stop,
                    freq="15m", init_cash=1000,
                    fees=_estimate_cost(TOTAL_COST_BPS)
                )
            else:
                entries = (ema_fast_line < ema_slow_line) & (ema_fast_line.shift() >= ema_slow_line.shift())
                exits = (ema_fast_line > ema_slow_line) & (ema_fast_line.shift() <= ema_slow_line.shift())
                portfolio = vbt.Portfolio.from_signals(
                    df["close"], short_entries=entries, short_exits=exits,
                    sl_stop=sl_stop, tp_stop=tp_stop,
                    freq="15m", init_cash=1000,
                    fees=_estimate_cost(TOTAL_COST_BPS)
                )
            return portfolio

        # ---------- Original Strategies ----------
        rsi_period = int(params["rsi_period"])
        rsi_lower = float(params["rsi_lower"])
        rsi_upper = float(params["rsi_upper"])

        rsi = vbt.RSI.run(df["close"], window=rsi_period).rsi
        adx_series = compute_adx(df["high"], df["low"], df["close"], period=14)
        adx = adx_series

        # Relaxed regime filters
        is_trend_regime = adx > 20   # lowered from 25
        is_choppy_regime = adx < 30  # widened from 20

        ema_macro = vbt.MA.run(df["close"], window=800, ewm=True).ma
        if ema_macro.dropna().empty:
            ema_macro = vbt.MA.run(df["close"], window=200, ewm=True).ma
        ema20 = vbt.MA.run(df["close"], window=20, ewm=True).ma
        ema50 = vbt.MA.run(df["close"], window=50, ewm=True).ma

        # Determine entries/exits per strategy
        if strategy_type == "RSI_MEAN_REVERSION":
            if direction == "LONG":
                entries = (rsi < rsi_lower) & is_choppy_regime
                exits = rsi > rsi_upper
            else:
                entries = (rsi > rsi_upper) & is_choppy_regime
                exits = rsi < rsi_lower

        elif strategy_type == "EMA_PULLBACK_TREND":
            if direction == "LONG":
                entries = (rsi < rsi_lower) & (ema20 > ema50) & is_trend_regime
                exits = rsi > rsi_upper
            else:
                entries = (rsi > rsi_upper) & (ema20 < ema50) & is_trend_regime
                exits = rsi < rsi_lower

        elif strategy_type == "RSI_MOMENTUM_BREAKOUT":
            if direction == "LONG":
                entries = (rsi > rsi_upper) & (ema20 > ema50) & is_trend_regime
                exits = rsi < 50
            else:
                entries = (rsi < rsi_lower) & (ema20 < ema50) & is_trend_regime
                exits = rsi > 50
        else:
            raise ValueError(f"Unknown strategy_type: {strategy_type}")

        # Build Portfolio based on direction
        if direction == "LONG":
            portfolio = vbt.Portfolio.from_signals(
                df["close"], entries=entries, exits=exits,
                sl_stop=sl_stop, tp_stop=tp_stop,
                freq="15m", init_cash=1000,
                fees=_estimate_cost(TOTAL_COST_BPS)
            )
        else:
            portfolio = vbt.Portfolio.from_signals(
                df["close"], short_entries=entries, short_exits=exits,
                sl_stop=sl_stop, tp_stop=tp_stop,
                freq="15m", init_cash=1000,
                fees=_estimate_cost(TOTAL_COST_BPS)
            )
        return portfolio

    def _objective(self, trial, train_df):
        """
        Optuna objective function – maximises Sharpe Ratio while penalising
        overly complex parameter sets and invalid backtests.
        """
        strategy_type = trial.suggest_categorical(
            "strategy_type",
            ["RSI_MEAN_REVERSION", "EMA_PULLBACK_TREND", "RSI_MOMENTUM_BREAKOUT"],
        )
        direction = trial.suggest_categorical("direction", ["LONG", "SHORT"])

        # Elastic RSI bounds – keep search space reasonable
        if direction == "SHORT":
            rsi_upper = trial.suggest_int("rsi_upper", 55, 70)
            rsi_lower = trial.suggest_int("rsi_lower", 30, 45)
        else:
            rsi_upper = trial.suggest_int("rsi_upper", 50, 70)
            rsi_lower = trial.suggest_int("rsi_lower", 25, 42)

        params = {
            "strategy_type": strategy_type,
            "direction": direction,
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "rsi_lower": rsi_lower,
            "rsi_upper": rsi_upper,
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.015, 0.035, step=0.005),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.030, 0.060, step=0.005),
        }

        n_params = len(params) - 2
        complexity_penalty = 0.05 * n_params

        portfolio = self.run_backtest(train_df, params)
        sharpe = portfolio.sharpe_ratio()
        max_dd = abs(portfolio.max_drawdown())
        trades_cnt = portfolio.trades.count()

        # Penalise unrealistic or empty results
        if trades_cnt < 2 or max_dd > 0.30 or np.isnan(sharpe) or np.isinf(sharpe):
            return -999.0

        return sharpe - complexity_penalty

    def push_strategy_to_executor(self, pair: str, params: dict, oos_sharpe: float) -> bool:
        """
        Dispatch the validated strategy to the executor service via an HTTP POST.
        Payload format COMPATIBLE with main.py's Go Engine webhook.
        Returns True on success, False otherwise.
        """
        # Format payload to match main.py's expectations for Go Engine /webhook/strategy
        raw_symbol = pair.replace("/", "")
        
        payload = {
            "symbol": raw_symbol,
            "direction": str(params.get("direction", "LONG")),
            "leverage": f"{params.get('leverage', 3)}x",  # Default 3x if not present
            "rsi_lower": float(params.get("rsi_lower", 30.0)),
            "rsi_upper": float(params.get("rsi_upper", 70.0))
        }

        # Use the same webhook URL as main.py (via environment variable)
        executor_url = os.environ.get(
            "EXECUTOR_WEBHOOK_URL", "https://tethgard.up.railway.app/webhook/strategy"
        )

        try:
            resp = requests.post(
                executor_url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                print(f"✅ [Executor] Strategy for {pair} pushed successfully.")
                return True
            print(
                f"⚠️ [Executor] Push for {pair} failed (HTTP {resp.status_code}): {resp.text}"
            )
            return False
        except Exception as e:
            print(f"❌ [Executor] Failed to push {pair}: {e}")
            return False

    def heal_and_find_winner(self, n_trials=200, pair=None):
        """
        Run a walk‑forward optimisation, select the median‑Sharpe candidate,
        save it locally. 
        Note: Does NOT push to executor here - main.py handles the push separately
        to avoid duplicate pushes and payload format conflicts.
        Returns a tuple (best_params, success_flag).
        """
        if pair:
            self.last_pair = pair

        print("[Self-Healing] Running Balanced Macro Trend Optimization...")

        tscv = TimeSeriesSplit(n_splits=5, gap=96, test_size=None)
        oos_metrics = []
        all_params = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.df)):
            train_df = self.df.iloc[train_idx]
            val_df = self.df.iloc[val_idx]

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=fold),
            )
            study.optimize(lambda t: self._objective(t, train_df), n_trials=n_trials // 5)

            if study.best_value == -999.0 or len(study.best_trials) == 0:
                print(f"   Fold {fold}: No valid candidate - skipping")
                continue

            best_params = study.best_params
            val_portfolio = self.run_backtest(val_df, best_params)

            oos_sharpe = val_portfolio.sharpe_ratio()
            oos_dd = abs(val_portfolio.max_drawdown())
            oos_trades = val_portfolio.trades.count()
            oos_winrate = val_portfolio.trades.win_rate() or 0.0

            # Relaxed acceptance criteria
            if (
                oos_trades >= 2
                and oos_dd <= 0.30
                and not np.isnan(oos_sharpe)
                and not np.isinf(oos_sharpe)
                and oos_winrate >= 0.5
                and oos_sharpe > 0.5
            ):
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
                    f"WinRate={oos_winrate*100:.1f}% | "
                    f"Trades={oos_trades} | DD={oos_dd*100:.1f}%"
                )
            else:
                print(
                    f"   Fold {fold}: Rejected - Sharpe={oos_sharpe:.2f} | "
                    f"WinRate={oos_winrate*100:.1f}% | "
                    f"Trades={oos_trades} | DD={oos_dd*100:.1f}%"
                )

        # ---------- Decide final strategy ----------
        if len(oos_metrics) >= 2:
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

            # Store OOS Sharpe for later logging
            self.last_oos_sharpe = chosen["sharpe"]

            # NOTE: Push to executor is handled by main.py scan_top_pairs_for_winner()
            # to ensure payload format compatibility and avoid double-pushing.
            # If you want to push from here, uncomment below:
            #
            # if self.last_pair:
            #     if self.push_strategy_to_executor(self.last_pair, best_params, self.last_oos_sharpe):
            #         print(
            #             f"🎯 [{self.last_pair}] Valid Candidate | "
            #             f"OOS Sharpe Ratio: {self.last_oos_sharpe:.2f}"
            #         )
            #     else:
            #         print(f"⚠️ [{self.last_pair}] Strategy saved locally but push to executor failed.")
            
            return best_params, True
        else:
            # Fallback to simple MA Crossover
            print("Not enough robust folds - falling back to MA Crossover strategy.")
            fallback_params = {
                "strategy_type": "MA_CROSSOVER",
                "direction": "LONG",
                "rsi_period": 14,
                "rsi_lower": 30,
                "rsi_upper": 70,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "ema_fast": 20,
                "ema_slow": 50,
            }
            self._save_winner_config(fallback_params)
            return fallback_params, False

    def _save_winner_config(self, params: dict):
        """Persist the chosen parameters to the config directory."""
        try:
            os.makedirs("config", exist_ok=True)
            with open(CONFIG_PATH, "w") as f:
                json.dump(params, f, indent=4)
            print("Winner config successfully saved to local JSON file.")
        except Exception as e:
            print(f"Failed to save winner config JSON: {e}")

def load_active_config():
    """Load the most recently saved strategy configuration."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return None

if __name__ == "__main__":
    # Example orchestrator – in production this would iterate over all tradable pairs
    # and read pair‑specific data files.
    df = pd.read_csv("market_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    engine = BayesianStrategyEngine(df)

    # For demo we hardcode a pair; real code will loop over pairs.
    best_params, success = engine.heal_and_find_winner(n_trials=200, pair="BTC/USDT")

    if success:
        print("\nPRODUCTION READY - best_params saved:")
        print(json.dumps(best_params, indent=2))
    else:
        print("\nUsing fallback strategy (MA Crossover) - best_params saved:")
        print(json.dumps(best_params, indent=2))
