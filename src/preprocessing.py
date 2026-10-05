import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.preprocessing import MinMaxScaler


def prepare_datasets(df: pd.DataFrame, feature_cols: List[str], target_col: str = 'Close',
                     time_step: int = 60, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Dict[str, Any]:
    """
    Splits financial time-series chronologically and prepares sequence windows.
    GUARANTEES:
    1. Zero Scaler Leakage: MinMaxScaler is fit ONLY on train slice (0 to train_cutoff).
    2. 3-way partition: 70% Train, 15% Validation, 15% Test.
    3. Strict out-of-sample test evaluation.
    """
    raw_features = df[feature_cols].values.astype(np.float32)
    N = len(raw_features)
    target_idx = feature_cols.index(target_col)

    train_end_idx = int(N * train_ratio)
    val_end_idx = int(N * (train_ratio + val_ratio))

    # Scaler fit STRICTLY on train partition
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(raw_features[:train_end_idx])
    scaled_data = scaler.transform(raw_features)

    # Construct sliding window sequences (X, y)
    X_all, y_all, dates_all = [], [], []
    for i in range(len(scaled_data) - time_step):
        X_all.append(scaled_data[i : i + time_step])
        y_all.append(scaled_data[i + time_step, target_idx])
        dates_all.append(df.index[i + time_step])

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)
    dates_all = np.array(dates_all)

    # Assign each sequence to train, val, or test based on prediction timestamp
    pred_indices = np.arange(len(y_all)) + time_step

    train_mask = pred_indices < train_end_idx
    val_mask = (pred_indices >= train_end_idx) & (pred_indices < val_end_idx)
    test_mask = pred_indices >= val_end_idx

    return {
        'X_train': X_all[train_mask],
        'y_train': y_all[train_mask],
        'dates_train': dates_all[train_mask],

        'X_val': X_all[val_mask],
        'y_val': y_all[val_mask],
        'dates_val': dates_all[val_mask],

        'X_test': X_all[test_mask],
        'y_test': y_all[test_mask],
        'dates_test': dates_all[test_mask],

        'scaler': scaler,
        'target_idx': target_idx,
        'feature_cols': feature_cols,
        'train_cutoff_date': df.index[train_end_idx],
        'val_cutoff_date': df.index[val_end_idx]
    }


def inverse_transform_target(scaler: MinMaxScaler, scaled_vals: np.ndarray,
                             target_col_idx: int, num_features: int) -> np.ndarray:
    """Inverts scaled predictions or actuals back to price scale."""
    dummy = np.zeros((len(scaled_vals), num_features), dtype=np.float32)
    dummy[:, target_col_idx] = np.squeeze(scaled_vals)
    return scaler.inverse_transform(dummy)[:, target_col_idx]
