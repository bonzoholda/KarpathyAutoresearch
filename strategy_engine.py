import json
import os
import numpy as np
import pandas as pd
import vectorbt as vbt
import optuna

# Sembunyikan verbose logs dari Optuna agar console tetap bersih
optuna.logging.set_verbosity(optuna.logging.WARNING)

CONFIG_PATH = os.path.join("config", "strategy_config.json")


class BayesianStrategyEngine:
    def __init__(self, df: pd.DataFrame):
        """
        df harus berupa DataFrame dengan kolom minimal: ['open', 'high', 'low', 'close', 'volume']
        """
        self.df = df

    @staticmethod
    def run_backtest(df: pd.DataFrame, params: dict):
        """
        Eksekusi backtest deterministik dengan VectorBT berdasarkan parameter input
        """
        rsi_period = int(params["rsi_period"])
        rsi_lower = float(params["rsi_lower"])
        rsi_upper = float(params["rsi_upper"])
        sl_stop = float(params["stop_loss_pct"])
        tp_stop = float(params["take_profit_pct"])

        # Hitung Indikator RSI
        rsi = vbt.RSI.run(df["close"], window=rsi_period).rsi

        # Signal Entry & Exit
        entries = rsi < rsi_lower
        exits = rsi > rsi_upper

        # Simulasi Portofolio
        portfolio = vbt.Portfolio.from_signals(
            df["close"],
            entries=entries,
            exits=exits,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            freq="1h",
            init_cash=1000,
            fees=0.001,  # Fee 0.1%
        )
        return portfolio

    def _objective(self, trial, train_df: pd.DataFrame):
        """
        Objective function untuk Bayesian Optimizer
        """
        params = {
            "rsi_period": trial.suggest_int("rsi_period", 5, 30),
            "rsi_lower": trial.suggest_int("rsi_lower", 15, 40),
            "rsi_upper": trial.suggest_int("rsi_upper", 60, 85),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.01, 0.05, step=0.005),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.02, 0.10, step=0.005),
        }

        portfolio = self.run_backtest(train_df, params)

        sharpe = portfolio.sharpe_ratio()
        max_dd = abs(portfolio.max_drawdown())
        trades_count = portfolio.trades.count()

        # Safety Guardrails: Eliminasi strategi ghoib/overfitted
        if trades_count < 10 or max_dd > 0.25 or np.isnan(sharpe):
            return -999.0

        return sharpe

    def heal_and_find_winner(self, n_trials: int = 100):
        """
        Mekanisme Self-Healing:
        1. Split Data (70% In-Sample / 30% Out-of-Sample)
        2. Bayesian Search via Optuna
        3. Validering Out-Of-Sample (OOS)
        """
        split_idx = int(len(self.df) * 0.7)
        train_df = self.df.iloc[:split_idx]
        val_df = self.df.iloc[split_idx:]

        print("🔍 [Self-Healing] Running Bayesian Optimization (TPESampler)...")

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler()
        )
        study.optimize(lambda trial: self._objective(trial, train_df), n_trials=n_trials)

        best_params = study.best_params
        best_in_sample_sharpe = study.best_value

        print(f"📊 Candidate Found | In-Sample Sharpe: {best_in_sample_sharpe:.2f}")

        # Out-of-Sample (OOS) Validation
        val_portfolio = self.run_backtest(val_df, best_params)
        val_sharpe = val_portfolio.sharpe_ratio()
        val_trades = val_portfolio.trades.count()
        val_win_rate = val_portfolio.win_rate()

        print(
            f"🧪 [OOS Validation] Sharpe: {val_sharpe:.2f} | WinRate: {val_win_rate*100:.1f}% | Trades: {val_trades}"
        )

        # Kriteria Mutlak "Strategy Winner"
        if val_sharpe > 1.0 and val_trades >= 3:
            print("🏆 STRATEGY WINNER VALIDATED! Saving parameters...")
            self._save_winner_config(best_params)
            return best_params, True
        else:
            print("❌ Candidate Failed OOS Test (Overfitting Detected). Rejecting.")
            return None, False

    def _save_winner_config(self, params: dict):
        """Menyimpan parameter pemenang ke file JSON secara otomatis"""
        os.makedirs("config", exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(params, f, indent=4)
        print(f"💾 Updated config saved to {CONFIG_PATH}")


def load_active_config():
    """Load parameter strategi pemenang yang sedang aktif"""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return None
