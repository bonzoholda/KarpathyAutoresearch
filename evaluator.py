# evaluator.py

import pandas as pd
import numpy as np
import requests
import importlib
import sys

def fetch_klines(symbol="BTCUSDT", interval="15m", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    res = requests.get(url).json()
    
    df = pd.DataFrame(res, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
    ])
    
    # PERBAIKAN DI SINI: Ubah seluruh kolom numerik dari String ke Float
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = df[col].astype(float)
        
    return df

def run_evaluation():
    try:
        df = fetch_klines()
        
        # Reload modul strategi agar membaca versi mutasi terbaru dari LLM
        if 'strategy_candidate' in sys.modules:
            importlib.reload(sys.modules['strategy_candidate'])
        else:
            import strategy_candidate
            
        signals = strategy_candidate.generate_signals(df)
        
        # Pastikan output sinyal berukuran sama dengan dataframe
        if len(signals) != len(df):
            return -999.0
            
        # Hitung Strategy Returns & Sharpe Ratio
        market_returns = df['close'].pct_change()
        strategy_returns = market_returns * signals.shift(1)
        
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        
        if std_ret == 0 or np.isnan(std_ret):
            return -999.0  # Penalti jika strategi pasif/invalid
            
        # Annualized Sharpe Ratio (untuk data 15m)
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(35040)
        
        # Penalti jika Drawdown terlalu tinggi (>15%)
        cum_returns = (1 + strategy_returns.fillna(0)).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        if max_drawdown > 0.15:
            sharpe_ratio -= 2.0
            
        return round(float(sharpe_ratio), 4)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return -999.0

if __name__ == "__main__":
    score = run_evaluation()
    print(f"EVALUATION_SCORE: {score}")
