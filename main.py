import time
import ccxt
import pandas as pd
from strategy_engine import BayesianStrategyEngine, load_active_config

# Inisialisasi Binance Public Client
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

TIMEFRAME = '1h'
CANDLE_LIMIT = 1000  # ~41 hari data historis


def fetch_live_market_data(symbol: str, timeframe: str = TIMEFRAME, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """
    Menarik data OHLCV dari Binance Public REST API untuk symbol tertentu
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
    Rotasi scanning Top 10 pairs sampai menemukan Strategy Winner yang lulus OOS Test
    """
    print("🔎 Starting Multi-Asset Opportunity Scan across Top 10 Pairs...\n")

    for pair in TOP_10_PAIRS:
        print(f"--------------------------------------------------")
        print(f"🔍 Analyzing Pair: [{pair}]")
        
        df = fetch_live_market_data(pair)
        if df is None or len(df) < 500:
            print(f"⏩ Skipping {pair} due to insufficient data.")
            continue

        latest_price = df['close'].iloc[-1]
        print(f"📊 Market Price: ${latest_price:,.2f}")

        # Inisialisasi Engine untuk pair saat ini
        engine = BayesianStrategyEngine(df)
        
        # Jalankan Bayesian Optimization (100 trials per pair)
        params, success = engine.heal_and_find_winner(n_trials=100)

        if success:
            # Sisipkan informasi pair pemenang ke dalam dict params
            params['symbol'] = pair
            params['latest_price'] = latest_price
            
            print(f"\n🎉 WINNER FOUND! Selected Pair: [{pair}]")
            # Override simpan config dengan menyertakan nama symbol
            engine._save_winner_config(params)
            return params, True
        else:
            print(f"❌ [{pair}] No valid strategy passed OOS Validation. Moving to next pair...\n")

    print("⚠️ Scan Completed: No winner found across all Top 10 pairs for current market regime.")
    return None, False


def main():
    print("🚀 Starting Bayesian Autoresearch Engine (Multi-Asset Top 10 Scanner)...")

    active_params = load_active_config()
    is_healthy = active_params is not None

    while True:
        try:
            # Re-evaluate/heal jika belum ada strategi aktif
            if not is_healthy or active_params is None:
                new_params, success = scan_top_pairs_for_winner()

                if success:
                    active_params = new_params
                    is_healthy = True
                    print("\n✅ Active Strategy Hot-Reloaded with Multi-Asset Winner!")
                else:
                    print("\n⏳ Retrying full scan in next cycle...")
                    is_healthy = False

            # Log status parameter aktif
            if is_healthy:
                print("\n🤖 [Research Engine] Current Active Winner Strategy:")
                print(f"   -> Target Pair: {active_params.get('symbol', 'BTC/USDT')}")
                print(f"   -> RSI Period: {active_params['rsi_period']}")
                print(f"   -> Entry Threshold: {active_params['rsi_lower']}")
                print(f"   -> Exit Threshold: {active_params['rsi_upper']}")
                print(f"   -> Stop Loss: {active_params['stop_loss_pct']*100:.2f}%")
                print(f"   -> Take Profit: {active_params['take_profit_pct']*100:.2f}%")

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
