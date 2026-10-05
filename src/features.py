import numpy as np
import pandas as pd
from typing import List, Tuple

FEATURE_LEVELS = {
    "level1_price": [
        'Open', 'High', 'Low', 'Close', 'Volume'
    ],
    "level2_returns": [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Return', 'LogReturn', 'Return_5d', 'Return_10d'
    ],
    "level3_volatility": [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Return', 'LogReturn', 'Return_5d', 'Return_10d',
        'RollingVol20', 'RollingStd20', 'HL_Range', 'ATR14'
    ],
    "level4_full": [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Return', 'LogReturn', 'Return_5d', 'Return_10d',
        'RollingVol20', 'RollingStd20', 'HL_Range', 'ATR14',
        'MA10', 'MA20', 'MA50', 'RSI14', 'MACD', 'MACD_Signal', 'MACD_Hist'
    ]
}


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR) over a rolling window."""
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculates MACD line, Signal line, and Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def engineer_financial_features(df: pd.DataFrame, level: str = "level4_full") -> Tuple[pd.DataFrame, List[str]]:
    """
    Computes hierarchical financial features from raw OHLCV data:
    - Level 1: Price and Volume
    - Level 2: Simple & Log Returns (1d, 5d, 10d)
    - Level 3: Realized Volatility, Rolling Std, High-Low Range, ATR(14)
    - Level 4: Moving Averages (10, 20, 50), RSI(14), MACD(12,26,9)

    Returns:
        pd.DataFrame: Cleaned dataframe with engineered features (NaNs dropped).
        List[str]: List of feature column names matching the requested level.
    """
    out = df.copy()

    # --- Level 2: Returns ---
    out['Return'] = out['Close'].pct_change()
    out['LogReturn'] = np.log(out['Close'] / out['Close'].shift(1))
    out['Return_5d'] = out['Close'].pct_change(5)
    out['Return_10d'] = out['Close'].pct_change(10)

    # --- Level 3: Volatility ---
    out['RollingStd20'] = out['Close'].rolling(window=20).std()
    out['RollingVol20'] = out['LogReturn'].rolling(window=20).std() * np.sqrt(252)
    out['HL_Range'] = (out['High'] - out['Low']) / out['Close']
    out['ATR14'] = calculate_atr(out, window=14)

    # --- Level 4: Momentum & Trend ---
    out['MA10'] = out['Close'].rolling(window=10).mean()
    out['MA20'] = out['Close'].rolling(window=20).mean()
    out['MA50'] = out['Close'].rolling(window=50).mean()
    out['RSI14'] = calculate_rsi(out['Close'], window=14)
    macd, signal, hist = calculate_macd(out['Close'])
    out['MACD'] = macd
    out['MACD_Signal'] = signal
    out['MACD_Hist'] = hist

    # Drop warm-up NaN rows (from 50-day MA and indicators)
    out = out.dropna().copy()

    if level not in FEATURE_LEVELS:
        raise ValueError(f"Unknown feature level '{level}'. Choose from: {list(FEATURE_LEVELS.keys())}")

    feature_cols = FEATURE_LEVELS[level]
    return out, feature_cols
