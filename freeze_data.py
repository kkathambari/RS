import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import freeze_benchmark_datasets

if __name__ == "__main__":
    # Freezing the Phase 2 first experiment group
    tickers_to_freeze = ["AAPL", "MSFT", "NVDA", "RELIANCE.NS", "TCS.NS", "^NSEI", "^GSPC"]
    print(f"Starting freeze of {len(tickers_to_freeze)} benchmark assets (2015 to 2024)...")
    results = freeze_benchmark_datasets(tickers=tickers_to_freeze, start_date="2015-01-01", end_date="2024-01-01")
    print(f"\nCompleted freezing {len(results)}/{len(tickers_to_freeze)} assets successfully.")
