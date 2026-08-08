import time
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
    'LINK/USDT',
    'NEAR/USDT'
]

TIMEFRAME = '15m'
CANDLE_LIMIT = 1000

# Waktu Tunggu Dynamic
IDLE_TIME_FULL_SLOTS = 3600   # 60 Menit jika 3 Slot penuh
IDLE_TIME_SEARCH_SLOTS = 300  # 5 Menit jika masih ada slot kosong (Fast Hunt)

EXECUTOR_URL = os.getenv("EXECUTOR_WEBHOOK_URL", "https://okx-trade-executor.up.railway.app/webhook/strategy-update")


def get_executor_active_slots_count() -> int:
    """Mengecek jumlah slot aktif di Executor (Repo 2) via API GET /position"""
    try:
        # Mengambil base URL dari EXECUTOR_URL
        base_url = EXECUTOR_URL.split("/webhook")[0]
        res = requests.get(f"{base_url}/position", timeout=3)
        if res.status_code == 200:
            data = res.json()
            return data.get("active_slots_count", 0)
    except Exception as e:
        print(f"⚠️ Could not check Executor slots status: {e}")
    return 0


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
        params, success = engine.heal_and_find_winner(n_trials=120)

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

    if best_candidate_params and winning_pair and winning_engine:
        print("\n" + "🎉 " * 15)
        print(f"🏆 TOURNAMENT WINNER FOUND! Selected Pair: [{winning_pair}] (OOS Sharpe: {highest_oos_sharpe:.2f})")
        print("🎉 " * 15)
        
        winning_engine._save_winner_config(best_candidate_params)
        return best_candidate_params, True
    else:
        print("\n⚠️ Tournament Scan Completed: No winner found across all Top 10 pairs for current market regime.")
        return None, False


def main():
    print("🚀 Starting Bayesian Autoresearch Engine (Dynamic Multi-Slot Scanner)...")

    active_params = load_active_config()

    while True:
        try:
            # 1. Cek berapa slot yang sedang aktif di Executor (Repo 2)
            active_slots = get_executor_active_slots_count()
            print(f"\n📊 Current Active Trades in Executor: {active_slots}/3 Slots occupied.")

            # 2. Tentukan berapa lama harus idle setelah scanning
            if active_slots >= 3:
                current_sleep_time = IDLE_TIME_FULL_SLOTS
                print(f"🔒 All 3 slots are occupied! Research engine will sleep for {IDLE_TIME_FULL_SLOTS // 60} minutes.")
            else:
                current_sleep_time = IDLE_TIME_SEARCH_SLOTS
                print(f"🎯 Slots available ({3 - active_slots} slots free)! Running fast tournament scan (5-min retry cycle)...")

                # Jalankan pencarian jika slot masih tersedia
                new_params, success = scan_top_pairs_for_winner()
                if success:
                    active_params = new_params
                    print("\n✅ Winner Strategy Hot-Reloaded to Executor!")

            # Log status
            if active_params:
                print("\n🤖 [Research Engine] Latest Winner Strategy Pushed:")
                print(f"   -> Target Pair : {active_params.get('symbol', 'BTC/USDT')}")
                print(f"   -> Direction   : {active_params.get('direction', 'LONG')} 🚀")
                print(f"   -> RSI Period  : {active_params['rsi_period']}")
                print(f"   -> Entry Target: RSI {active_params['rsi_lower']}")

            print(f"\n💤 Research Engine idling... Waiting {current_sleep_time // 60} minutes for next cycle.")
            time.sleep(current_sleep_time)

        except KeyboardInterrupt:
            print("\n🛑 Research Engine stopped gracefully.")
            break
        except Exception as e:
            print(f"❌ Error encountered in main loop: {e}")
            print("⏳ Retrying in 30 seconds...")
            time.sleep(30)


if __name__ == "__main__":
    main()
