import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class FinancialBacktester:
    """
    Simulates a long/cash execution strategy driven by model price forecasts.
    Evaluates:
      - Cumulative Return (%)
      - Annualized Sharpe Ratio
      - Maximum Drawdown (MDD %)
      - Win Rate (%)
    Compares against the Buy & Hold benchmark.
    """
    def __init__(self, threshold: float = 0.001, risk_free_rate: float = 0.02,
                 transaction_cost_bps: float = 5.0):
        self.threshold = threshold
        self.risk_free_rate = risk_free_rate
        self.cost_pct = transaction_cost_bps / 10000.0  # 5 bps = 0.05%

    def simulate(self, actual_prices: np.ndarray, predicted_prices: np.ndarray,
                 dates: pd.DatetimeIndex = None) -> Dict[str, Any]:
        actual_prices = np.squeeze(actual_prices)
        predicted_prices = np.squeeze(predicted_prices)
        N = len(actual_prices)

        # Observed asset returns
        asset_returns = np.diff(actual_prices) / actual_prices[:-1]

        # Forecasted expected return from today's price to tomorrow's forecast
        # signal_t decided at t using predicted_prices[t] vs actual_prices[t-1]
        pred_returns = (predicted_prices[1:] - actual_prices[:-1]) / actual_prices[:-1]

        # Binary or Directional Position (+1 for Long, 0 for Cash/Neutral)
        signals = (pred_returns > self.threshold).astype(float)

        # Transaction cost on signal changes
        turnover = np.abs(np.diff(np.concatenate(([0], signals))))
        tc = turnover * self.cost_pct

        # Strategy returns after transaction costs
        strategy_returns = (signals * asset_returns) - tc

        # Equity curves (starting at 1.0)
        strat_equity = np.cumprod(1.0 + strategy_returns)
        bh_equity = np.cumprod(1.0 + asset_returns)

        # Performance calculations
        days = len(strategy_returns)
        ann_factor = 252.0 / max(1, days)

        total_ret_strat = (strat_equity[-1] - 1.0) * 100.0
        total_ret_bh = (bh_equity[-1] - 1.0) * 100.0

        ann_ret_strat = ((1.0 + total_ret_strat / 100.0) ** ann_factor - 1.0) * 100.0
        ann_ret_bh = ((1.0 + total_ret_bh / 100.0) ** ann_factor - 1.0) * 100.0

        daily_rf = self.risk_free_rate / 252.0
        excess_strat = strategy_returns - daily_rf
        excess_bh = asset_returns - daily_rf

        std_strat = np.std(strategy_returns)
        std_bh = np.std(asset_returns)

        sharpe_strat = (np.mean(excess_strat) / (std_strat + 1e-8)) * np.sqrt(252.0)
        sharpe_bh = (np.mean(excess_bh) / (std_bh + 1e-8)) * np.sqrt(252.0)

        # Max Drawdown
        def max_drawdown(equity):
            peaks = np.maximum.accumulate(equity)
            drawdowns = (equity - peaks) / peaks
            return float(np.min(drawdowns)) * 100.0

        mdd_strat = max_drawdown(strat_equity)
        mdd_bh = max_drawdown(bh_equity)

        # Win Rate
        active_trades = signals > 0
        if np.sum(active_trades) > 0:
            win_rate = float(np.mean(strategy_returns[active_trades] > 0)) * 100.0
        else:
            win_rate = 0.0

        metrics = {
            "Strategy Return (%)": round(float(total_ret_strat), 2),
            "Buy & Hold Return (%)": round(float(total_ret_bh), 2),
            "Strategy Annualized (%)": round(float(ann_ret_strat), 2),
            "Buy & Hold Annualized (%)": round(float(ann_ret_bh), 2),
            "Strategy Sharpe": round(float(sharpe_strat), 3),
            "Buy & Hold Sharpe": round(float(sharpe_bh), 3),
            "Strategy Max Drawdown (%)": round(float(mdd_strat), 2),
            "Buy & Hold Max Drawdown (%)": round(float(mdd_bh), 2),
            "Win Rate (%)": round(float(win_rate), 2),
            "Trade Days": int(np.sum(active_trades)),
            "Total Days": int(days)
        }

        return {
            "metrics": metrics,
            "strategy_equity": strat_equity,
            "bh_equity": bh_equity,
            "signals": signals
        }
