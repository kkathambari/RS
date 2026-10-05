import numpy as np
import scipy.stats as stats
from typing import Dict, Any, Tuple


def diebold_mariano_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray,
                         h: int = 1, criterion: str = "MSE") -> Tuple[float, float]:
    """
    Computes the Diebold-Mariano (DM) test statistic to determine if the forecast
    accuracy difference between two models is statistically significant.
    
    Parameters:
      y_true: Actual ground truth values (N,)
      y_pred1: Forecasts from Model 1 (e.g., Proposed Regime-Adaptive Attention) (N,)
      y_pred2: Forecasts from Model 2 (e.g., Baseline Standard Attention or GRU) (N,)
      h: Forecast horizon (default 1-step ahead)
      criterion: "MSE" (squared error) or "MAE" (absolute error)
      
    Returns:
      dm_stat: Test statistic. Negative value indicates Model 1 has lower error than Model 2.
      p_value: Two-tailed p-value. If p < 0.05, Model 1 is significantly different.
    """
    y_true = np.squeeze(y_true)
    y_pred1 = np.squeeze(y_pred1)
    y_pred2 = np.squeeze(y_pred2)
    
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    
    if criterion.upper() == "MSE":
        d = (e1 ** 2) - (e2 ** 2)
    elif criterion.upper() == "MAE":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
        
    T = len(d)
    mean_d = np.mean(d)
    
    # Autocovariance estimation (Harvey, Leybourne & Newbold correction for h-step ahead)
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.sum((d[k:] - mean_d) * (d[:-k] - mean_d)) / T
        gamma_sum += 2.0 * gamma_k
        
    variance_d = (gamma_0 + gamma_sum) / T
    if variance_d <= 1e-12:
        return 0.0, 1.0
        
    dm_stat = mean_d / np.sqrt(variance_d)
    
    # Harvey-Leybourne-Newbold small-sample adjustment
    hln_stat = dm_stat * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(hln_stat), df=T - 1))
    
    return round(float(hln_stat), 4), round(float(p_value), 5)


def aggregate_seed_metrics(seed_records: list) -> Dict[str, str]:
    """
    Computes Mean +/- Std across multiple random seeds for publication tables.
    """
    import pandas as pd
    df = pd.DataFrame(seed_records)
    summary = {}
    for col in ['rmse', 'mae', 'mape', 'r2', 'directional_acc_pct']:
        if col in df.columns:
            m = df[col].mean()
            s = df[col].std()
            summary[f"{col}_mean_std"] = f"{m:.4f} +/- {s:.4f}"
            summary[f"{col}_mean"] = round(float(m), 4)
            summary[f"{col}_std"] = round(float(s), 4)
    return summary
