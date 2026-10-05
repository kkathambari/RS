import numpy as np
import scipy.stats as stats
from typing import Dict, Any, Tuple


def diebold_mariano_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray,
                         h: int = 1, criterion: str = "MSE") -> Tuple[float, float, str]:
    """
    Computes the Diebold-Mariano (DM) test statistic with Harvey-Leybourne-Newbold (HLN)
    correction to determine if the forecast accuracy difference between two models is
    statistically significant.
    
    Parameters:
      y_true: Actual ground truth values (N,)
      y_pred1: Forecasts from Model 1 (e.g., Proposed Regime-Adaptive Attention) (N,)
      y_pred2: Forecasts from Model 2 (e.g., Baseline Standard Attention or GRU) (N,)
      h: Forecast horizon (default 1-step ahead)
      criterion: "MSE" (squared error) or "MAE" (absolute error)
      
    Returns:
      dm_stat: HLN-corrected test statistic. Negative value indicates Model 1 has lower error.
      p_value: Raw two-tailed p-value.
      p_value_str: Publication-safe string formatting (e.g., '< 0.0001' or '0.0342').
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
    mean_d = float(np.mean(d))
    
    # Robust Newey-West / Bartlett autocovariance estimation for long-run variance
    # For h=1, we still allow autocorrelation up to Newey-West rule-of-thumb lag L
    # to account for financial volatility clustering in loss differentials.
    nw_lags = max(h - 1, int(np.floor(4.0 * ((T / 100.0) ** (2.0 / 9.0)))))
    gamma_0 = float(np.var(d, ddof=0))
    gamma_sum = 0.0
    for k in range(1, nw_lags + 1):
        gamma_k = float(np.sum((d[k:] - mean_d) * (d[:-k] - mean_d)) / T)
        weight = 1.0 - (k / (nw_lags + 1.0))  # Bartlett kernel
        gamma_sum += 2.0 * weight * gamma_k
        
    variance_d = (gamma_0 + gamma_sum) / T
    if variance_d <= 1e-12:
        return 0.0, 1.0, "1.0000", "No significant difference"
        
    dm_stat = mean_d / np.sqrt(variance_d)
    
    # Harvey-Leybourne-Newbold small-sample adjustment factor
    hln_factor = np.sqrt(max(1e-8, (T + 1 - 2 * h + (h * (h - 1)) / T) / T))
    hln_stat = float(dm_stat * hln_factor)
    
    # Student's t-distribution with (T - 1) degrees of freedom
    p_val_raw = float(2.0 * (1.0 - stats.t.cdf(np.abs(hln_stat), df=max(1, T - 1))))
    
    # Publication-safe string formatting: avoid literal 0.0
    if p_val_raw < 1e-4:
        p_val_str = "< 0.0001"
    else:
        p_val_str = f"{p_val_raw:.4f}"
        
    # Directional interpretation:
    # d = e1^2 - e2^2, so hln_stat > 0 means Model 1 has HIGHER error than Model 2.
    if p_val_raw < 0.05:
        if hln_stat < 0:
            conclusion_str = "Proposed has significantly LOWER error"
        else:
            conclusion_str = "Baseline has significantly LOWER error"
    else:
        conclusion_str = "No significant difference"
        
    return round(hln_stat, 4), p_val_raw, p_val_str, conclusion_str


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
