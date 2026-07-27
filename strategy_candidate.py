# strategy_candidate.py
import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    Fungsi ini di-generate/di-edit oleh AI secara otonom.
    Return: Series berisi 1 (BUY), -1 (SELL), atau 0 (HOLD)
    """
    # Baseline Strategy: Simple Moving Average Crossover
    sma_fast = df['close'].rolling(10).mean()
    sma_slow = df['close'].rolling(30).mean()
    
    signals = np.where(sma_fast > sma_slow, 1, -1)
    return pd.Series(signals, index=df.index).fillna(0)
