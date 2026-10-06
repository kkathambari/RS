import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from src.preprocessing import prepare_datasets, inverse_transform_target
from src.regime_detector import MarketRegimeDetector
from src.models.regime_attention_gru import train_regime_adaptive_model
from src.evaluation import compute_forecasting_metrics


def generate_walk_forward_folds(N: int, n_folds: int = 3) -> List[Dict[str, int]]:
    """
    Generates deterministic expanding walk-forward validation splits:
      - Fold 1: Train [0 -> 50%], Val [50% -> 65%], Test [65% -> 80%] (Span: 80% of data)
      - Fold 2: Train [0 -> 60%], Val [60% -> 75%], Test [75% -> 90%] (Span: 90% of data)
      - Fold 3: Train [0 -> 70%], Val [70% -> 85%], Test [85% -> 100%] (Span: 100% of data)
      
    This guarantees zero future lookahead bias while strictly testing temporal generalization
    across multiple distinct market cycles.
    """
    windows = [
        {"fold": 1, "train_pct": 0.50, "val_pct": 0.65, "test_pct": 0.80},
        {"fold": 2, "train_pct": 0.60, "val_pct": 0.75, "test_pct": 0.90},
        {"fold": 3, "train_pct": 0.70, "val_pct": 0.85, "test_pct": 1.00}
    ][:n_folds]
    folds = []
    for w in windows:
        folds.append({
            "fold": w["fold"],
            "train_end": int(N * w["train_pct"]),
            "val_end": int(N * w["val_pct"]),
            "test_end": int(N * w["test_pct"])
        })
    return folds


def run_walk_forward_validation(feat_df: pd.DataFrame, feature_cols: List[str],
                                time_step: int = 30, epochs: int = 15,
                                n_folds: int = 3, seed: int = 42,
                                target_col: str = 'Close', is_return: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes expanding walk-forward validation for the proposed Regime-Adaptive Attention model.
    Strictly aligns regime conditioning vectors to decision time t (end of lookback window).
    """
    N = len(feat_df)
    target_idx = feature_cols.index(target_col)
    num_features = len(feature_cols)
    folds = generate_walk_forward_folds(N, n_folds=n_folds)
    
    fold_metrics = []
    
    for f in folds:
        fold_id = f['fold']
        train_end = f['train_end']
        val_end = f['val_end']
        test_end = f['test_end']
        
        sub_df = feat_df.iloc[:test_end].copy()
        train_cutoff_date = sub_df.index[train_end - 1]
        
        # Train-only regime detector strictly up to train_cutoff_date
        reg_det = MarketRegimeDetector(n_regimes=4, random_state=seed)
        reg_det.fit(sub_df.loc[:train_cutoff_date])
        _, probs_all, _ = reg_det.predict_regimes(sub_df)
        
        # Partition
        train_ratio = train_end / len(sub_df)
        val_ratio = (val_end - train_end) / len(sub_df)
        
        datasets = prepare_datasets(
            df=sub_df, feature_cols=feature_cols,
            target_col=target_col, time_step=time_step,
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
        # CAUSAL SAMPLING: Sample regime vector strictly at decision time t (end of X sequence)
        X_train_reg = np.array([probs_all[d_map[d]] for d in datasets['dates_train_decision']], dtype=np.float32)
        X_val_reg = np.array([probs_all[d_map[d]] for d in datasets['dates_val_decision']], dtype=np.float32)
        X_test_reg = np.array([probs_all[d_map[d]] for d in datasets['dates_test_decision']], dtype=np.float32)
        
        model, dur, _ = train_regime_adaptive_model(
            X_train_seq, X_train_reg, y_train,
            X_val_seq, X_val_reg, y_val,
            time_step=time_step, epochs=epochs,
            ckpt_dir=f"artifacts/checkpoints/wf_fold_{fold_id}",
            ckpt_name=f"wf_regime_adapt_{fold_id}"
        )
        
        test_preds_scaled = model.predict([X_test_seq, X_test_reg], verbose=0)
        y_test_act = inverse_transform_target(scaler, y_test, target_idx, num_features)
        y_test_pred = inverse_transform_target(scaler, test_preds_scaled, target_idx, num_features)
        
        m = compute_forecasting_metrics(y_test_act, y_test_pred, is_return=is_return)
        f_row = {
            "Fold": f"Fold {fold_id}",
            "Train Samples": len(X_train_seq),
            "Val Samples": len(X_val_seq),
            "Test Samples": len(X_test_seq),
            "Test Period": f"{datasets['dates_test'][0].strftime('%Y-%m')} to {datasets['dates_test'][-1].strftime('%Y-%m')}",
            "Test RMSE": m['rmse'],
            "Test MAE": m['mae']
        }
        if not is_return and 'mape' in m:
            f_row["Test MAPE (%)"] = m['mape']
        f_row["Test R2"] = m['r2']
        f_row["MDA (%)"] = m['directional_acc_pct']
        f_row["Train Time (s)"] = round(dur, 2)
        fold_metrics.append(f_row)
        
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
