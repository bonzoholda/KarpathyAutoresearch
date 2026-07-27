import time
import ccxt
import pandas as pd
from strategy_engine import BayesianStrategyEngine, load_active_config

# Inisialisasi Binance Public Client (Tidak Membutuhkan API Key)
exchange = ccxt.binance({
    'enableRateLimit': True, # Mencegah ip-ban akibat panggil berlebihan
})

SYMBOL = 'BTC/USDT'  # Pasangan aset yang ingin di-research
TIMEFRAME = '1h'     # Timeframe candlestick (1h, 4h, 1d)
CANDLE_LIMIT = 1000 # Jumlah candle historis untuk backtest & optimasi (~41 hari data)


def fetch_live_market_data(symbol: str = SYMBOL, timeframe: str = TIMEFRAME, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """
    Menarik data OHLCV historis asli dari Binance Public REST API
    """
    print(f"📥 Fetching live OHLCV market data from Binance Public API ({symbol} - {timeframe})...")
    
    try:
        # Panggil API Public Binance via CCXT
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        # Konversi array menjadi DataFrame Pandas
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Format Timestamp ms ke Datetime Index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"✅ Successfully fetched {len(df)} candles! (Latest Price: ${df['close'].iloc[-1]:,.2f})")
        return df

    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        raise e


def main():
    print("🚀 Starting Bayesian Autoresearch Engine (Live Binance Market Feed)...")

    # Load parameter winner aktif dari local config jika ada
    active_params = load_active_config()
    is_healthy = active_params is not None

    while True:
        try:
            # 1. Fetch Live Market Data dari Binance Public
            df = fetch_live_market_data()

            # 2. Self-Healing Trigger: Run Bayesian Optimization jika belum ada winner valid
            if not is_healthy or active_params is None:
                print("\n🚨 [Triggered] Self-Healing Process Initialized on Live Market Data...")
                engine = BayesianStrategyEngine(df)
                
                # Jalankan 100 iterasi pencarian strategi pemenang
                new_params, success = engine.heal_and_find_winner(n_trials=100)

                if success:
                    active_params = new_params
                    is_healthy = True
                    print("✅ Active Strategy Hot-Reloaded with Winner Parameters!")
                else:
                    print("⚠️ Self-Healing couldn't find a safe winner on current market regime. Will retry next cycle.")
                    is_healthy = False

            # 3. Log Status Parameter Aktif
            if is_healthy:
                print("\n🤖 [Research Engine] Active Strategy Parameters Ready for Trade Executor:")
                print(f"   -> RSI Period: {active_params['rsi_period']}")
                print(f"   -> Entry Threshold: {active_params['rsi_lower']}")
                print(f"   -> Exit Threshold: {active_params['rsi_upper']}")
                print(f"   -> Stop Loss: {active_params['stop_loss_pct']*100:.2f}%")
                print(f"   -> Take Profit: {active_params['take_profit_pct']*100:.2f}%")

            # 4. Sleep Interval: Lakukan re-optimasi/cek data setiap 1 jam
            print("\n💤 Research Engine idling... Waiting 1 hour for next analysis cycle.")
            time.sleep(3600)

        except KeyboardInterrupt:
            print("\n🛑 Research Engine stopped gracefully.")
            break
        except Exception as e:
            print(f"❌ Error encountered in main loop: {e}")
            print("⏳ Retrying in 30 seconds...")
            time.sleep(30)


if __name__ == "__main__":
    main()
