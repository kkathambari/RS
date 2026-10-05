import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_forecasting_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes regression and directional accuracy metrics."""
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    non_zero = np.abs(y_true) > 1e-5
    if np.any(non_zero):
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)
    else:
        mape = 0.0

    r2 = float(r2_score(y_true, y_pred))

    if len(y_true) > 1:
        actual_delta = np.diff(y_true)
        pred_delta = y_pred[1:] - y_true[:-1]
        mda = float(np.mean(np.sign(actual_delta) == np.sign(pred_delta)) * 100.0)
    else:
        mda = 0.0

    return {
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'mape': round(mape, 3),
        'r2': round(r2, 4),
        'directional_acc_pct': round(mda, 2)
    }


def log_experiment(results_dir: str, run_meta: Dict[str, Any],
                   train_metrics: Dict[str, float], val_metrics: Dict[str, float],
                   test_metrics: Dict[str, float]) -> str:
    """Appends experiment run metadata and metrics to CSV and saves JSON artifact."""
    os.makedirs(results_dir, exist_ok=True)
    runs_dir = os.path.join(results_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    record = {**run_meta}
    for k, v in train_metrics.items():
        record[f"train_{k}"] = v
    for k, v in val_metrics.items():
        record[f"val_{k}"] = v
    for k, v in test_metrics.items():
        record[f"test_{k}"] = v

    csv_path = os.path.join(results_dir, "experiments_log.csv")
    df = pd.DataFrame([record])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

    run_id = run_meta.get('run_id', 'run')
    run_file = os.path.join(runs_dir, f"{run_id}.json")
    with open(run_file, "w") as f:
        json.dump(record, f, indent=2)

    return csv_path
