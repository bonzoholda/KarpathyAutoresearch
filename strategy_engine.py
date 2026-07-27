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

        rsi = vbt.RSI.run(df["close"], window=rsi_period).rsi

        # Skenario LONG
        if direction == "LONG":
            entries = rsi < rsi_lower
            exits = rsi > rsi_upper
            portfolio = vbt.Portfolio.from_signals(
                df["close"],
                entries=entries,
                exits=exits,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                freq="1h",
                init_cash=1000,
                fees=0.0006,  # Standard Futures Fee ~0.06%
            )
        # Skenario SHORT
        else:
            entries = rsi > rsi_upper
            exits = rsi < rsi_lower
            portfolio = vbt.Portfolio.from_signals(
                df["close"],
                short_entries=entries,
                short_exits=exits,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                freq="1h",
                init_cash=1000,
                fees=0.0006,
            )

        return portfolio

    def _objective(self, trial, train_df: pd.DataFrame):
        params = {
            "direction": trial.suggest_categorical("direction", ["LONG", "SHORT"]),
            "rsi_period": trial.suggest_int("rsi_period", 5, 25),
            "rsi_lower": trial.suggest_int("rsi_lower", 15, 40),
            "rsi_upper": trial.suggest_int("rsi_upper", 60, 85),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.015, 0.05, step=0.005),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.02, 0.10, step=0.005),
        }

        portfolio = self.run_backtest(train_df, params)
        sharpe = portfolio.sharpe_ratio()
        max_dd = abs(portfolio.max_drawdown())
        trades_count = portfolio.trades.count()

        if trades_count < 8 or max_dd > 0.20 or np.isnan(sharpe):
            return -999.0

        return sharpe

    def heal_and_find_winner(self, n_trials: int = 100):
        split_idx = int(len(self.df) * 0.7)
        train_df = self.df.iloc[:split_idx]
        val_df = self.df.iloc[split_idx:]

        print("🔍 [Self-Healing] Running Bayesian Futures Optimization...")

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler()
        )
        study.optimize(lambda trial: self._objective(trial, train_df), n_trials=n_trials)

        best_params = study.best_params
        best_in_sample_sharpe = study.best_value

        print(f"📊 Candidate Found ({best_params['direction']}) | In-Sample Sharpe: {best_in_sample_sharpe:.2f}")

        val_portfolio = self.run_backtest(val_df, best_params)
        val_sharpe = val_portfolio.sharpe_ratio()
        val_trades = val_portfolio.trades.count()
        val_win_rate = val_portfolio.trades.win_rate()
        win_rate_pct = (val_win_rate * 100) if not np.isnan(val_win_rate) else 0.0

        print(
            f"🧪 [OOS Validation] Direction: {best_params['direction']} | Sharpe: {val_sharpe:.2f} | WinRate: {win_rate_pct:.1f}% | Trades: {val_trades}"
        )

        if val_sharpe > 1.0 and val_trades >= 3:
            print("🏆 FUTURES STRATEGY WINNER VALIDATED! Saving parameters...")
            self._save_winner_config(best_params)
            return best_params, True
        else:
            print("❌ Candidate Failed OOS Test (Overfitting Detected). Rejecting.")
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
