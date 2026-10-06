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
    def __init__(self, threshold: float = 0.0005, risk_free_rate: float = 0.02,
                 transaction_cost_bps: float = 5.0):
        """
        Parameters:
          threshold: Expected-return entry threshold (default 0.0005 = 0.05% / 5 bps)
          risk_free_rate: Annualized cash benchmark rate for Sharpe calculation (default 0.02 = 2.0%)
          transaction_cost_bps: Per-turnover transaction cost friction (default 5.0 bps = 0.05%)
        """
        self.threshold = threshold
        self.risk_free_rate = risk_free_rate
        self.cost_pct = transaction_cost_bps / 10000.0  # 5 bps = 0.05%

    def simulate(self, actual_prices: np.ndarray = None, predicted_prices: np.ndarray = None,
                 dates: pd.DatetimeIndex = None, is_return_forecast: bool = False,
                 current_prices: np.ndarray = None, target_prices: np.ndarray = None,
                 predicted_values: np.ndarray = None,
                 dates_decision: pd.DatetimeIndex = None,
                 dates_target: pd.DatetimeIndex = None) -> Dict[str, Any]:
        """
        Simulates causal out-of-sample trading strategy with 5-bps transaction costs.
        
        Causal Timeline:
          Decision Time t:
            - Investor observes current price P_t (current_prices[k])
            - Model produces forecast for t+1 (predicted_values[k])
            - Signal generated: Long (+1) if forecasted return > threshold, else Cash (0)
          Holding Period t -> t+1:
            - Asset moves from P_t to P_{t+1} (target_prices[k])
            - Realized asset return: (P_{t+1} - P_t) / P_t
            - Strategy earns: (Signal * Realized Return) - (Turnover * 5 bps)
        """
        if current_prices is not None and target_prices is not None and predicted_values is not None:
            c_prices = np.squeeze(current_prices).astype(np.float64)
            t_prices = np.squeeze(target_prices).astype(np.float64)
            p_values = np.squeeze(predicted_values).astype(np.float64)
        else:
            # Fallback legacy compatibility: align strictly without forward shift
            raw_act = np.squeeze(actual_prices).astype(np.float64)
            raw_pred = np.squeeze(predicted_prices).astype(np.float64)
            c_prices = raw_act[:-1]
            t_prices = raw_act[1:]
            p_values = raw_pred[:-1]

        # 1. Realized asset return over each holding interval t -> t+1
        asset_returns = (t_prices - c_prices) / (c_prices + 1e-8)

        # 2. Expected forecast return formed at time t
        if is_return_forecast:
            pred_returns = p_values
        else:
            pred_returns = (p_values - c_prices) / (c_prices + 1e-8)

        # 3. Binary Position (+1 for Long, 0 for Cash/Neutral) decided at time t
        signals = (pred_returns > self.threshold).astype(np.float64)

        # 4. Transaction cost on signal transitions (turnover)
        turnover = np.abs(np.diff(np.concatenate(([0.0], signals))))
        tc = turnover * self.cost_pct

        # 5. Realized strategy return after 5-bps transaction costs
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

        std_strat = float(np.std(strategy_returns))
        std_bh = float(np.std(asset_returns))

        if std_strat < 1e-6 or np.sum(signals > 0) == 0:
            sharpe_strat_val = "N/A"
        else:
            sharpe_strat_val = round(float((np.mean(excess_strat) / std_strat) * np.sqrt(252.0)), 3)

        if std_bh < 1e-6:
            sharpe_bh_val = "N/A"
        else:
            sharpe_bh_val = round(float((np.mean(excess_bh) / std_bh) * np.sqrt(252.0)), 3)

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
            "Strategy Sharpe": sharpe_strat_val,
            "Buy & Hold Sharpe": sharpe_bh_val,
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
