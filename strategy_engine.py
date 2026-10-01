# --------------------------------------------------------------
# strategy_engine.py - Production-Ready Backtesting Engine (TethGard 2.0)
# --------------------------------------------------------------
import json
import os
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit

import matplotlib
matplotlib.use('Agg')

import vectorbt as vbt

# 💡 Optuna Logging ke INFO agar pergerakan trial & hasil analisis tampil di konsol
optuna.logging.set_verbosity(optuna.logging.INFO)

# Config - Cost can be overridden via environment variable
CONFIG_PATH = os.path.join("config", "strategy_config.json")
TOTAL_COST_BPS = float(os.environ.get("TOTAL_COST_BPS", 0.0010))  # Default 10 bps per side

def _apply_shift(df):
    """Shift data 1 bar to avoid look-ahead bias."""
    df = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].shift(1)
    return df

def _estimate_cost(fees_bps):
    return fees_bps

def compute_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up = high - high.shift()
    down = low.shift() - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

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
        self.last_oos_sharpe = None          
        self.last_pair = None                

    @staticmethod
    def run_backtest(df: pd.DataFrame, params: dict):
        strategy_type = params.get("strategy_type", "RSI_MEAN_REVERSION")
        direction = params["direction"]
        sl_stop = float(params["stop_loss_pct"])
        tp_stop = float(params["take_profit_pct"])

        if strategy_type == "MA_CROSSOVER":
            ema_fast = int(params.get("ema_fast", 20))
            ema_slow = int(params.get("ema_slow", 50))
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

        rsi_period = int(params["rsi_period"])
        rsi_lower = float(params["rsi_lower"])
        rsi_upper = float(params["rsi_upper"])

        rsi = vbt.RSI.run(df["close"], window=rsi_period).rsi
        adx_series = compute_adx(df["high"], df["low"], df["close"], period=14)
        adx = adx_series

        # 🛡️ Dynamic ADX Regime Thresholds aligned with Go Engine
        is_trend_regime = adx >= 25.0   
        is_choppy_regime = adx < 22.0  

        ema20 = vbt.MA.run(df["close"], window=20, ewm=True).ma
        ema50 = vbt.MA.run(df["close"], window=50, ewm=True).ma

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
        strategy_type = trial.suggest_categorical(
            "strategy_type",
            ["RSI_MEAN_REVERSION", "EMA_PULLBACK_TREND", "RSI_MOMENTUM_BREAKOUT"],
        )
        direction = trial.suggest_categorical("direction", ["LONG", "SHORT"])

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
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.020, 0.040, step=0.005),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.040, 0.065, step=0.005),
        }

        n_params = len(params) - 2
        complexity_penalty = 0.05 * n_params

        portfolio = self.run_backtest(train_df, params)
        sharpe = portfolio.sharpe_ratio()
        max_dd = abs(portfolio.max_drawdown())
        trades_cnt = portfolio.trades.count()

        if trades_cnt < 2 or max_dd > 0.30 or np.isnan(sharpe) or np.isinf(sharpe):
            return -999.0

        return sharpe - complexity_penalty

    def heal_and_find_winner(self, n_trials=200, pair=None):
        if pair:
            self.last_pair = pair

        print(f"\n🔍 [Self-Healing] Running Optimization for pair: {pair or 'GLOBAL'}...")

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
            study.optimize(lambda t: self._objective(t, train_df), n_trials=max(10, n_trials // 5))

            if study.best_value == -999.0 or len(study.best_trials) == 0:
                print(f"❌ Fold {fold}: No valid candidate found during optimization.")
                continue

            best_params = study.best_params
            val_portfolio = self.run_backtest(val_df, best_params)

            oos_sharpe = val_portfolio.sharpe_ratio()
            oos_dd = abs(val_portfolio.max_drawdown())
            oos_trades = val_portfolio.trades.count()
            oos_winrate = val_portfolio.trades.win_rate() or 0.0

            if (
                oos_trades >= 1
                and oos_dd <= 0.35
                and not np.isnan(oos_sharpe)
                and not np.isinf(oos_sharpe)
                and oos_sharpe > 0.1
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
                    f"✅ Fold {fold} ACCEPTED -> Type: {best_params.get('strategy_type')} | "
                    f"Dir: {best_params.get('direction')} | Sharpe: {oos_sharpe:.2f} | "
                    f"WinRate: {oos_winrate*100:.1f}% | Trades: {oos_trades} | DD: {oos_dd*100:.1f}%"
                )
            else:
                print(
                    f"⚠️ Fold {fold} REJECTED -> Sharpe: {oos_sharpe:.2f} | "
                    f"WinRate: {oos_winrate*100:.1f}% | Trades: {oos_trades} | DD: {oos_dd*100:.1f}%"
                )

        # ---------- Decide final strategy ----------
        if len(oos_metrics) >= 1:
            sorted_metrics = sorted(oos_metrics, key=lambda x: x["sharpe"])
            median_idx = len(sorted_metrics) // 2
            chosen = sorted_metrics[median_idx]

            best_params = all_params[median_idx]
            self._save_winner_config(best_params)

            self.last_oos_sharpe = chosen["sharpe"]

            print("\n" + "="*60)
            print(f"🎯 WINNER STRATEGY FOUND FOR {pair or 'GLOBAL'}")
            print(f"   Strategy Type : {best_params.get('strategy_type')}")
            print(f"   Direction     : {best_params.get('direction')}")
            print(f"   RSI Bounds    : Lower={best_params.get('rsi_lower')} | Upper={best_params.get('rsi_upper')}")
            print(f"   Stop Loss / TP: SL={best_params.get('stop_loss_pct')*100:.1f}% | TP={best_params.get('take_profit_pct')*100:.1f}%")
            print(f"   OOS Performance: Sharpe={chosen['sharpe']:.2f} | WinRate={chosen['winrate']*100:.1f}% | Trades={chosen['trades']}")
            print("="*60 + "\n")

            return best_params, True
        else:
            print("\n⚠️ Not enough robust folds - falling back to MA Crossover strategy.")
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
        try:
            os.makedirs("config", exist_ok=True)
            with open(CONFIG_PATH, "w") as f:
                json.dump(params, f, indent=4)
            print("💾 Winner config successfully saved to config/strategy_config.json")
        except Exception as e:
            print(f"❌ Failed to save winner config JSON: {e}")

def load_active_config():
    """Load the most recently saved strategy configuration."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return None
