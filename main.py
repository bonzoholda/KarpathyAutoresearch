import time
import pandas as pd
import numpy as np
from strategy_engine import BayesianStrategyEngine, load_active_config


def fetch_dummy_market_data(days: int = 60) -> pd.DataFrame:
    """
    Fungsi generator data pasar.
    Gantikan fungsi ini dengan panggil API Exchange asli Anda (misal: Binance/CCXT/Bybit)
    """
    date_range = pd.date_range(end=pd.Timestamp.now(), periods=days * 24, freq="1h")
    np.random.seed(42)
    price_changes = np.random.normal(loc=0.0002, scale=0.01, size=len(date_range))
    price_path = 100 * np.exp(np.cumsum(price_changes))

    df = pd.DataFrame(
        {
            "open": price_path,
            "high": price_path * 1.002,
            "low": price_path * 0.998,
            "close": price_path,
            "volume": np.random.randint(100, 1000, size=len(date_range)),
        },
        index=date_range,
    )
    return df


def main():
    print("🚀 Starting Bayesian Autoresearch Trading Bot...")

    # Load config awal (jika ada)
    active_params = load_active_config()
    is_healthy = active_params is not None

    while True:
        try:
            # 1. Fetch Market Data Terbaru
            print("\n📥 Fetching latest market data...")
            df = fetch_dummy_market_data(days=60)

            # 2. Kondisi Self-Healing Trigger:
            #    - Jika bot baru berjalan pertama kali
            #    - Atau jika strategi aktif ditandai 'Unhealthy' / Performa Drop
            if not is_healthy or active_params is None:
                print("🚨 [Triggered] Self-Healing Process Initialized...")
                engine = BayesianStrategyEngine(df)
                new_params, success = engine.heal_and_find_winner(n_trials=100)

                if success:
                    active_params = new_params
                    is_healthy = True
                    print("✅ Active Strategy Hot-Reloaded with Winner Parameters!")
                else:
                    print("⚠️ Self-Healing Failed to find a valid strategy. Retrying next cycle...")
                    is_healthy = False

            # 3. Jalankan Logic Eksekusi Trade dengan Active Winner Params
            if is_healthy:
                print(f"🤖 Bot Executing Trade Logic using Active Winner Params:")
                print(f"   -> RSI Period: {active_params['rsi_period']}")
                print(f"   -> Entry Threshold: {active_params['rsi_lower']}")
                print(f"   -> Exit Threshold: {active_params['rsi_upper']}")
                print(f"   -> Stop Loss: {active_params['stop_loss_pct']*100}%")
                print(f"   -> Take Profit: {active_params['take_profit_pct']*100}%")
                
                # TODO: Panggil fungsi API order Binance / Smart Contract dApp Anda di sini

            # 4. Sleep Interval (misal: Cek / Re-evaluate setiap 1 jam)
            print("💤 Waiting for next interval...")
            time.sleep(3600)

        except KeyboardInterrupt:
            print("\n🛑 Bot stopped gracefully by user.")
            break
        except Exception as e:
            print(f"❌ Error encountered: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
