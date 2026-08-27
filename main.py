import matplotlib
matplotlib.use('Agg')  # Mencegah Matplotlib mencoba membuka window GUI

import sys
import gc
import requests
import os
import ccxt
import pandas as pd
import numpy as np
from strategy_engine import BayesianStrategyEngine, load_active_config

# Inisialisasi Public Client
exchange = ccxt.binance({
    'enableRateLimit': True,
})

# List Top 10 Pairs Paling Likuid
TOP_10_PAIRS = [
    'BTC/USDT',
    'ETH/USDT',
    'SOL/USDT',
    'BNB/USDT',
    'XRP/USDT',
    'ADA/USDT',
    'DOGE/USDT',
    'AVAX/USDT',
    'XAUT/USDT',
    'NEAR/USDT'
]

TIMEFRAME = '15m'
CANDLE_LIMIT = 1000

# Fallback disesuaikan ke endpoint Go Engine (/webhook/strategy)
EXECUTOR_URL = os.getenv("EXECUTOR_WEBHOOK_URL", "https://tethgard-executor-go.up.railway.app/webhook/strategy")


def get_executor_active_slots_count() -> int:
    """Mengecek jumlah slot aktif di Executor (Repo 2) via API GET /position"""
    try:
        base_url = EXECUTOR_URL.split("/webhook")[0]
        res = requests.get(f"{base_url}/position", timeout=5)
        if res.status_code == 200:
            data = res.json()
            return data.get("active_slots_count", 0)
    except Exception as e:
        print(f"⚠️ Could not check Executor slots status: {e}")
    return 0


def push_winning_strategy_to_executor(params: dict) -> bool:
    """Mengirim data winning strategy hasil Optuna ke Go Engine PostgreSQL (active_strategies)"""
    try:
        # Bersihkan symbol (misal 'BTC/USDT' -> 'BTCUSDT')
        raw_symbol = params.get('symbol', 'BTC/USDT')
        formatted_symbol = raw_symbol.replace("/", "")

        payload = {
            "symbol": formatted_symbol,
            "direction": str(params.get('direction', 'LONG')),
            "leverage": f"{params.get('leverage', 3)}x",  # Mengirim string "3x"
            "rsi_lower": float(params.get('rsi_lower', 30.0)),
            "rsi_upper": float(params.get('rsi_upper', 70.0))
        }

        print(f"📡 Pushing strategy to Go Engine: {payload}")
        res = requests.post(EXECUTOR_URL, json=payload, timeout=5)
        
        if res.status_code == 200:
            print("✅ Strategy successfully registered and saved to PostgreSQL 'active_strategies'!")
            return True
        else:
            print(f"⚠️ Failed to push strategy. HTTP Status: {res.status_code}, Response: {res.text}")
    except Exception as e:
        print(f"❌ Error pushing winning strategy to Go Engine: {e}")
    return False


def fetch_live_market_data(symbol: str, timeframe: str = TIMEFRAME, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"⚠️ Failed to fetch data for {symbol}: {e}")
        return None


def scan_top_pairs_for_winner():
    print("\n" + "=" * 65)
    print("🔎 Starting Multi-Asset Tournament Scan across Top 10 Pairs...")
    print("=" * 65 + "\n")

    best_candidate_params = None
    highest_oos_sharpe = -999.0
    winning_pair = None
    winning_engine = None

    for pair in TOP_10_PAIRS:
        print(f"--------------------------------------------------")
        print(f"🔍 Analyzing Pair: [{pair}]")
        
        df = fetch_live_market_data(pair)
        if df is None or len(df) < 500:
            print(f"⏩ Skipping {pair} due to insufficient data.")
            continue

        latest_price = float(df['close'].iloc[-1])
        print(f"📊 Market Price: ${latest_price:,.4f}")

        engine = BayesianStrategyEngine(df)
        params, success = engine.heal_and_find_winner(n_trials=100)

        if success and params:
            split_idx = int(len(df) * 0.7)
            val_df = df.iloc[split_idx:]
            val_portfolio = engine.run_backtest(val_df, params)
            val_sharpe = val_portfolio.sharpe_ratio()

            if np.isnan(val_sharpe):
                val_sharpe = -999.0

            print(f"🎯 [{pair}] Valid Candidate | OOS Sharpe Ratio: {val_sharpe:.2f}")

            if val_sharpe > highest_oos_sharpe:
                highest_oos_sharpe = val_sharpe
                params['symbol'] = pair
                params['latest_price'] = latest_price
                best_candidate_params = params
                winning_pair = pair
                winning_engine = engine
        else:
            print(f"❌ [{pair}] No valid strategy passed OOS Validation. Moving to next pair...\n")

        # Pembersihan memori setiap selesai memindai 1 koin
        del engine
        gc.collect()

    if best_candidate_params and winning_pair and winning_engine:
        print("\n" + "🎉 " * 15)
        print(f"🏆 TOURNAMENT WINNER FOUND! Selected Pair: [{winning_pair}] (OOS Sharpe: {highest_oos_sharpe:.2f})")
        print("🎉 " * 15)
        
        winning_engine._save_winner_config(best_candidate_params)
        return best_candidate_params, True
    else:
        print("\n⚠️ Tournament Scan Completed: No winner found across all Top 10 pairs for current market regime.")
        return None, False


def run_cron_job():
    print("🚀 [CRON TRIGGERED] Starting Bayesian Autoresearch Engine...")

    # 1. Cek berapa slot yang sedang aktif di Executor (Repo 2)
    active_slots = get_executor_active_slots_count()
    print(f"📊 Current Active Trades in Executor: {active_slots}/3 Slots occupied.")

    # 2. Jika slot di Executor penuh (3/3), lewati scanning untuk menghemat resources
    if active_slots >= 3:
        print("🔒 All 3 slots are occupied! Skipping scan for this cycle.")
    else:
        print(f"🎯 Available slots: {3 - active_slots} free. Executing tournament scan...")
        new_params, success = scan_top_pairs_for_winner()
        
        if success and new_params:
            print("\n🤖 [Research Engine] Winner Strategy Successfully Found:")
            print(f"   -> Target Pair : {new_params.get('symbol', 'BTC/USDT')}")
            print(f"   -> Direction   : {new_params.get('direction', 'LONG')} 🚀")
            print(f"   -> RSI Period  : {new_params.get('rsi_period', 14)}")
            print(f"   -> Entry Target: RSI {new_params.get('rsi_lower', 30)}")

            # OTO-PUSH: Kirim otomatis ke Go Engine & PostgreSQL
            push_winning_strategy_to_executor(new_params)

    print("\n🏁 [CRON FINISHED] Tournament scan completed. Shutting down container...")
    
    # Garbage collection akhir sebelum container mati
    gc.collect()
    sys.exit(0)


if __name__ == "__main__":
    try:
        run_cron_job()
    except Exception as e:
        print(f"❌ Critical Error in Cron Execution: {e}")
        sys.exit(1)
