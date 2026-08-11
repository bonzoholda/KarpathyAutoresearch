import json
import os
import requests
import numpy as np
import pandas as pd
import vectorbt as vbt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
CONFIG_PATH = os.path.join("config", "strategy_config.json")


class BayesianStrategyEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def run_backtest(df: pd.DataFrame, params: dict):
        rsi_period = int(params["rsi_period"])
        rsi_lower = float(params["rsi_lower"])
        rsi_upper = float(params["rsi_upper"])
        sl_stop = float(params["stop_loss_pct"])
        tp_stop = float(params["take_profit_pct"])
        direction = params["direction"]
        strategy_type = params.get("strategy_type", "RSI_MEAN_REVERSION")

        # Indikator Dasar: RSI pada TF 15m
        rsi = vbt.RSI.run(df["close"], window=rsi_period).rsi

        # --- 🛡️ MULTI-TIMEFRAME TREND GUARD (1-HOUR EMA 200 FILTER) ---
        df_1h = df["close"].resample("1h").last().dropna()
        ema200_1h = vbt.MA.run(df_1h, window=200, ewm=True).ma
        ema200_macro = ema200_1h.reindex(df.index, method="ffill")

        is_macro_uptrend = df["close"] > ema200_macro
        is_macro_downtrend = df["close"] < ema200_macro

        # Indikator Tambahan untuk Momentum Breakout (EMA 20 & 50 15m)
        ema20_15m = vbt.MA.run(df["close"], window=20, ewm=True).ma
        ema50_15m = vbt.MA.run(df["close"], window=50, ewm=True).ma

        # --- LOGIKA SINYAL ANEKA STRATEGI ---
        if strategy_type == "RSI_MEAN_REVERSION":
            if direction == "LONG":
                entries = (rsi < rsi_lower) & is_macro_uptrend
                exits = rsi > rsi_upper
            else:  # SHORT
                entries = (rsi > rsi_upper) & is_macro_downtrend
                exits = rsi < rsi_lower

        elif strategy_type == "EMA_PULLBACK_TREND":
            if direction == "LONG":
                entries = (rsi < rsi_lower) & is_macro_uptrend
                exits = (rsi > rsi_upper) | (~is_macro_uptrend)
            else:  # SHORT
                entries = (rsi > rsi_upper) & is_macro_downtrend
                exits = (rsi < rsi_lower) | (~is_macro_downtrend)

        elif strategy_type == "RSI_MOMENTUM_BREAKOUT":
            # STRATEGI ARUS DERAS: Menunggangi akselerasi momentum saat RSI menembus threshold
            rsi_prev = rsi.shift(1)
            if direction == "LONG":
                # Entry saat RSI cross-above rsi_upper & EMA20 > EMA50 & Macro Uptrend
                entries = (rsi > rsi_upper) & (rsi_prev <= rsi_upper) & (ema20_15m > ema50_15m) & is_macro_uptrend
                exits = (rsi < 50) | (ema20_15m < ema50_15m)
            else:  # SHORT
                # Entry saat RSI cross-below rsi_lower & EMA20 < EMA50 & Macro Downtrend
                entries = (rsi < rsi_lower) & (rsi_prev >= rsi_lower) & (ema20_15m < ema50_15m) & is_macro_downtrend
                exits = (rsi > 50) | (ema20_15m > ema50_15m)

        # Backtest Engine via VectorBT
        if direction == "LONG":
            portfolio = vbt.Portfolio.from_signals(
                df["close"],
                entries=entries,
                exits=exits,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                freq="15m",
                init_cash=1000,
                fees=0.0006,  # Fee OKX Futures ~0.06%
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
                fees=0.0006,
            )

        return portfolio

    def _objective(self, trial, train_df: pd.DataFrame):
        strategy_type = trial.suggest_categorical(
            "strategy_type", ["RSI_MEAN_REVERSION", "EMA_PULLBACK_TREND", "RSI_MOMENTUM_BREAKOUT"]
        )
        direction = trial.suggest_categorical("direction", ["LONG", "SHORT"])

        # Rentang RSI disesuaikan dengan karakteristik masing-masing tipe strategi
        if strategy_type == "RSI_MOMENTUM_BREAKOUT":
            rsi_lower = trial.suggest_int("rsi_lower", 35, 45)  # Threshold Breakdown Short
            rsi_upper = trial.suggest_int("rsi_upper", 55, 65)  # Threshold Breakout Long
        elif strategy_type == "EMA_PULLBACK_TREND":
            rsi_lower = trial.suggest_int("rsi_lower", 35, 48)
            rsi_upper = trial.suggest_int("rsi_upper", 52, 65)
        else:  # RSI_MEAN_REVERSION
            rsi_lower = trial.suggest_int("rsi_lower", 20, 38)
            rsi_upper = trial.suggest_int("rsi_upper", 62, 80)

        params = {
            "strategy_type": strategy_type,
            "direction": direction,
            "rsi_period": trial.suggest_int("rsi_period", 5, 14),
            "rsi_lower": rsi_lower,
            "rsi_upper": rsi_upper,
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.010, 0.018, step=0.002),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.025, 0.080, step=0.005),
        }

        portfolio = self.run_backtest(train_df, params)
        sharpe = portfolio.sharpe_ratio()
        max_dd = abs(portfolio.max_drawdown())
        trades_count = portfolio.trades.count()

        if trades_count < 8 or max_dd > 0.18 or np.isnan(sharpe):
            return -999.0

        return sharpe

    def heal_and_find_winner(self, n_trials: int = 150):
        split_idx = int(len(self.df) * 0.7)
        train_df = self.df.iloc[:split_idx]
        val_df = self.df.iloc[split_idx:]

        print("🔍 [Self-Healing] Running Bayesian Futures Optimization (Multi-Strategy + HTF Guard)...")

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler()
        )
        study.optimize(lambda trial: self._objective(trial, train_df), n_trials=n_trials)

        best_params = study.best_params
        best_in_sample_sharpe = study.best_value

        print(
            f"📊 Candidate Found ({best_params['strategy_type']} | {best_params['direction']}) | In-Sample Sharpe: {best_in_sample_sharpe:.2f}"
        )

        val_portfolio = self.run_backtest(val_df, best_params)
        val_sharpe = val_portfolio.sharpe_ratio()
        val_trades = val_portfolio.trades.count()
        val_win_rate = val_portfolio.trades.win_rate()
        win_rate_pct = (val_win_rate * 100) if not np.isnan(val_win_rate) else 0.0

        print(
            f"🧪 [OOS Validation] Type: {best_params['strategy_type']} | Direction: {best_params['direction']} | Sharpe: {val_sharpe:.2f} | WinRate: {win_rate_pct:.1f}% | Trades: {val_trades}"
        )

        if val_sharpe > 1.0 and val_trades >= 3:
            print("🏆 FUTURES STRATEGY WINNER VALIDATED! Saving parameters...")
            self._save_winner_config(best_params)
            return best_params, True
        else:
            print("❌ Candidate Failed OOS Test (Overfitting / Counter-Trend Detected). Rejecting.")
            return None, False

    def _save_winner_config(self, params: dict):
        os.makedirs("config", exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(params, f, indent=4)

        EXECUTOR_URL = os.getenv(
            "EXECUTOR_WEBHOOK_URL",
            "https://okx-trade-executor.up.railway.app/webhook/strategy-update",
        )
        WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "my_secret_token_123")

        payload = {
            "symbol": str(params.get("symbol", "BTC/USDT")),
            "direction": str(params.get("direction", "LONG")),
            "rsi_period": int(params["rsi_period"]),
            "rsi_lower": float(params["rsi_lower"]),
            "rsi_upper": float(params["rsi_upper"]),
            "stop_loss_pct": float(params["stop_loss_pct"]),
            "take_profit_pct": float(params["take_profit_pct"]),
        }

        try:
            headers = {"Authorization": f"Bearer {WEBHOOK_SECRET}"}
            response = requests.post(EXECUTOR_URL, json=payload, headers=headers, timeout=5)
            if response.status_code == 200:
                print("🚀 Successfully pushed Futures Winner Strategy to OKX Executor!")
            else:
                print(f"⚠️ Webhook response error [{response.status_code}]: {response.text}")
        except Exception as e:
            print(f"⚠️ Failed to send webhook to Executor: {e}")


def load_active_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return None
