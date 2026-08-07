import time
import ccxt
import pandas as pd
import numpy as np
from strategy_engine import BayesianStrategyEngine, load_active_config

# Inisialisasi Binance/OKX Public Client (Menggunakan Binance Public REST API untuk data feed cepat)
exchange = ccxt.binance({
    'enableRateLimit': True,
})

# List Top 10 Pairs Paling Likuid (Non-Stablecoin)
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
CANDLE_LIMIT = 1000  # ~10 hari data historis pada TF 15m

# Definisikan interval waktu tunggu untuk Research Loop
IDLE_TIME_SUCCESS = 3600  # 1 Jam jika strategi pemenang ditemukan
IDLE_TIME_RETRY = 300     # 5 Menit jika TIDAK menemukan strategi pemenang (Fast Retry)


def fetch_live_market_data(symbol: str, timeframe: str = TIMEFRAME, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """
    Menarik data OHLCV dari Public REST API untuk symbol tertentu
    """
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
    """
    Turnamen Pemindaian Multi-Asset: Memindai seluruh Top 10 pairs, membandingkan skor
    Sharpe Ratio, dan memilih 1 strategi dengan Sharpe Ratio OOS tertinggi sebagai Winner Mutlak.
    """
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

        # Inisialisasi Engine untuk pair saat ini
        engine = BayesianStrategyEngine(df)
        
        # Jalankan Bayesian Optimization (120 trials per pair)
        params, success = engine.heal_and_find_winner(n_trials=120)

        if success and params:
            # Uji ulang pada segmen Out-of-Sample Validation (30% data terakhir) untuk menghitung Sharpe Ratio OOS
            split_idx = int(len(df) * 0.7)
            val_df = df.iloc[split_idx:]
            val_portfolio = engine.run_backtest(val_df, params)
            val_sharpe = val_portfolio.sharpe_ratio()

            if np.isnan(val_sharpe):
                val_sharpe = -999.0

            print(f"🎯 [{pair}] Valid Candidate | OOS Sharpe Ratio: {val_sharpe:.2f}")

            # Seleksi Turnamen: Pilih kandidat dengan Sharpe Ratio tertinggi di antara seluruh Top 10 pairs
            if val_sharpe > highest_oos_sharpe:
                highest_oos_sharpe = val_sharpe
                params['symbol'] = pair
                params['latest_price'] = latest_price
                best_candidate_params = params
                winning_pair = pair
                winning_engine = engine
        else:
            print(f"❌ [{pair}] No valid strategy passed OOS Validation. Moving to next pair...\n")

    # Keputusan Akhir Turnamen Multi-Asset
    if best_candidate_params and winning_pair and winning_engine:
        print("\n" + "🎉 " * 15)
        print(f"🏆 TOURNAMENT WINNER FOUND! Selected Pair: [{winning_pair}] (OOS Sharpe: {highest_oos_sharpe:.2f})")
        print("🎉 " * 15)
        
        # Push strategi pemenang mutlak ke OKX Executor via Webhook & Simpan Config Lokal
        winning_engine._save_winner_config(best_candidate_params)
        return best_candidate_params, True
    else:
        print("\n⚠️ Tournament Scan Completed: No winner found across all Top 10 pairs for current market regime.")
        return None, False


def main():
    print("🚀 Starting Bayesian Autoresearch Engine (Multi-Asset Top 10 Scanner)...")

    active_params = load_active_config()
    is_healthy = active_params is not None

    while True:
        try:
            current_sleep_time = IDLE_TIME_SUCCESS

            # Selalu jalankan turnamen pencarian strategi baru di setiap siklus
            new_params, success = scan_top_pairs_for_winner()

            if success:
                active_params = new_params
                is_healthy = True
                current_sleep_time = IDLE_TIME_SUCCESS
                print("\n✅ Active Strategy Hot-Reloaded with Multi-Asset Winner!")
            else:
                # Gunakan waktu tunggu pendek jika gagal menemukan pemenang
                current_sleep_time = IDLE_TIME_RETRY
                print(f"\n⏳ No winner found. Retrying scan in {IDLE_TIME_RETRY // 60} minutes...")
                if active_params is None:
                    is_healthy = False

            # Log status parameter aktif
            if is_healthy and active_params:
                print("\n🤖 [Research Engine] Current Active Futures Winner Strategy:")
                print(f"   -> Target Pair : {active_params.get('symbol', 'BTC/USDT')}")
                print(f"   -> Direction   : {active_params.get('direction', 'LONG')} 🚀")
                print(f"   -> RSI Period  : {active_params['rsi_period']}")
                print(f"   -> Entry Threshold: {active_params['rsi_lower']}")
                print(f"   -> Exit Threshold : {active_params['rsi_upper']}")
                print(f"   -> Stop Loss   : {active_params['stop_loss_pct']*100:.2f}%")
                print(f"   -> Take Profit : {active_params['take_profit_pct']*100:.2f}%")

            # Dynamic Sleep
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
