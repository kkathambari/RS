import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from src.preprocessing import prepare_datasets, inverse_transform_target
from src.regime_detector import MarketRegimeDetector
from src.models.regime_attention_gru import train_regime_adaptive_model
from src.evaluation import compute_forecasting_metrics


def generate_walk_forward_folds(N: int, n_folds: int = 3, min_train_ratio: float = 0.50) -> List[Dict[str, int]]:
    """
    Generates expanding walk-forward splits:
    Fold 1: Train [0 -> N*0.50], Val [N*0.50 -> N*0.65], Test [N*0.65 -> N*0.80]
    Fold 2: Train [0 -> N*0.60], Val [N*0.60 -> N*0.75], Test [N*0.75 -> N*0.90]
    Fold 3: Train [0 -> N*0.70], Val [N*0.70 -> N*0.85], Test [N*0.85 -> N*1.00]
    """
    folds = []
    step = (1.0 - min_train_ratio) / (n_folds + 1)
    
    for k in range(n_folds):
        train_pct = min_train_ratio + k * step
        val_pct = train_pct + step * 0.75
        test_pct = min(1.0, val_pct + step * 0.75)
        
        folds.append({
            "fold": k + 1,
            "train_end": int(N * train_pct),
            "val_end": int(N * val_pct),
            "test_end": int(N * test_pct)
        })
    return folds


def run_walk_forward_validation(feat_df: pd.DataFrame, feature_cols: List[str],
                                time_step: int = 30, epochs: int = 15,
                                n_folds: int = 3, seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes expanding walk-forward validation for the proposed Regime-Adaptive Attention model.
    """
    N = len(feat_df)
    target_idx = feature_cols.index('Close')
    num_features = len(feature_cols)
    folds = generate_walk_forward_folds(N, n_folds=n_folds)
    
    fold_metrics = []
    
    for f in folds:
        fold_id = f['fold']
        train_end = f['train_end']
        val_end = f['val_end']
        test_end = f['test_end']
        
        sub_df = feat_df.iloc[:test_end].copy()
        
        # Train-only regime detector
        reg_det = MarketRegimeDetector(n_regimes=4, random_state=seed)
        reg_det.fit(sub_df.iloc[:train_end])
        _, probs_all, _ = reg_det.predict_regimes(sub_df)
        
        # Partition
        train_ratio = train_end / len(sub_df)
        val_ratio = (val_end - train_end) / len(sub_df)
        
        datasets = prepare_datasets(
            df=sub_df, feature_cols=feature_cols,
            target_col='Close', time_step=time_step,
            train_ratio=train_ratio, val_ratio=val_ratio
        )
        
        X_train_seq = datasets['X_train']
        y_train = datasets['y_train']
        X_val_seq = datasets['X_val']
        y_val = datasets['y_val']
        X_test_seq = datasets['X_test']
        y_test = datasets['y_test']
        scaler = datasets['scaler']
        
        feat_dates = sub_df.index
        d_map = {d: i for i, d in enumerate(feat_dates)}
        X_train_reg = np.array([probs_all[d_map[d]] for d in datasets['dates_train']], dtype=np.float32)
        X_val_reg = np.array([probs_all[d_map[d]] for d in datasets['dates_val']], dtype=np.float32)
        X_test_reg = np.array([probs_all[d_map[d]] for d in datasets['dates_test']], dtype=np.float32)
        
        model, dur, _ = train_regime_adaptive_model(
            X_train_seq, X_train_reg, y_train,
            X_val_seq, X_val_reg, y_val,
            time_step=time_step, epochs=epochs,
            ckpt_dir=f"artifacts/checkpoints/wf_fold_{fold_id}"
        )
        
        test_preds_scaled = model.predict([X_test_seq, X_test_reg], verbose=0)
        y_test_act = inverse_transform_target(scaler, y_test, target_idx, num_features)
        y_test_pred = inverse_transform_target(scaler, test_preds_scaled, target_idx, num_features)
        
        m = compute_forecasting_metrics(y_test_act, y_test_pred)
        fold_metrics.append({
            "Fold": f"Fold {fold_id}",
            "Train Samples": len(X_train_seq),
            "Val Samples": len(X_val_seq),
            "Test Samples": len(X_test_seq),
            "Test Period": f"{datasets['dates_test'][0].strftime('%Y-%m')} to {datasets['dates_test'][-1].strftime('%Y-%m')}",
            "Test RMSE": m['rmse'],
            "Test MAE": m['mae'],
            "Test MAPE (%)": m['mape'],
            "Test R2": m['r2'],
            "MDA (%)": m['directional_acc_pct'],
            "Train Time (s)": round(dur, 2)
        })
        
    df_folds = pd.DataFrame(fold_metrics)
    
    # Cross-fold aggregate
    agg = {
        "mean_rmse": round(float(df_folds["Test RMSE"].mean()), 4),
        "std_rmse": round(float(df_folds["Test RMSE"].std()), 4),
        "mean_mae": round(float(df_folds["Test MAE"].mean()), 4),
        "mean_r2": round(float(df_folds["Test R2"].mean()), 4),
        "mean_mda": round(float(df_folds["MDA (%)"].mean()), 2)
    }
    
    return df_folds, agg
