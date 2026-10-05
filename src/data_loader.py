import os
import yfinance as yf
import pandas as pd
from datetime import datetime

BENCHMARK_TICKERS = {
    "us_stocks": ["AAPL", "MSFT", "NVDA", "AMZN", "JPM"],
    "indian_stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
    "indices": ["^GSPC", "^NDX", "^NSEI"]
}

DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2024-01-01"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def clean_ticker_filename(ticker: str) -> str:
    """Sanitizes ticker symbols for safe filesystem storage (e.g. ^NSEI -> _NSEI)."""
    return ticker.replace("^", "_").replace(".", "_")


def fetch_raw_ohlcv(ticker: str, start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE) -> pd.DataFrame:
    """Fetches clean OHLCV data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if data is None or data.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(col[0]).strip().capitalize() for col in data.columns]
    else:
        data.columns = [str(col).strip().capitalize() for col in data.columns]

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing column '{col}' for ticker {ticker}")

    clean_df = data[required_cols].dropna().copy()
    clean_df.index = pd.to_datetime(clean_df.index)
    clean_df.index.name = "Date"
    return clean_df


def freeze_benchmark_datasets(tickers=None, start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE,
                              data_dir: str = RAW_DATA_DIR) -> dict:
    """
    Downloads and permanently freezes benchmark datasets to data/raw/ as CSV files.
    This guarantees reproducible experimental runs regardless of future API changes.
    """
    os.makedirs(data_dir, exist_ok=True)
    if tickers is None:
        tickers = (
            BENCHMARK_TICKERS["us_stocks"] +
            BENCHMARK_TICKERS["indian_stocks"] +
            BENCHMARK_TICKERS["indices"]
        )

    saved_files = {}
    print(f"Freezing benchmark datasets ({start_date} to {end_date}) into {data_dir}...")
    for ticker in tickers:
        safe_name = clean_ticker_filename(ticker)
        file_path = os.path.join(data_dir, f"{safe_name}.csv")
        try:
            df = fetch_raw_ohlcv(ticker, start_date=start_date, end_date=end_date)
            df.to_csv(file_path)
            saved_files[ticker] = file_path
            print(f"  [OK] {ticker:<14} -> {file_path} ({len(df)} rows)")
        except Exception as e:
            print(f"  [FAIL] Failed to freeze {ticker}: {e}")

    # Save freeze metadata
    meta_path = os.path.join(data_dir, "freeze_metadata.json")
    metadata = {
        "frozen_at": datetime.now().isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "tickers": list(saved_files.keys()),
        "files": saved_files
    }
    import json
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return saved_files


def load_frozen_dataset(ticker: str, data_dir: str = RAW_DATA_DIR,
                        fallback_fetch: bool = True) -> pd.DataFrame:
    """
    Loads frozen dataset from disk. If not yet frozen and fallback_fetch=True,
    downloads and freezes it on the fly.
    """
    safe_name = clean_ticker_filename(ticker)
    file_path = os.path.join(data_dir, f"{safe_name}.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        return df

    if fallback_fetch:
        print(f"Local frozen copy not found for {ticker}. Fetching from API and freezing...")
        os.makedirs(data_dir, exist_ok=True)
        df = fetch_raw_ohlcv(ticker)
        df.to_csv(file_path)
        return df

    raise FileNotFoundError(f"Frozen dataset not found at {file_path}")
