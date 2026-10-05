import os
import sys
import time
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_frozen_dataset
from src.features import engineer_financial_features
from src.regime_detector import MarketRegimeDetector, REGIME_NAMES
from src.preprocessing import prepare_datasets, inverse_transform_target
from src.evaluation import compute_forecasting_metrics, log_experiment

from src.models.regime_attention_gru import (
    build_regime_adaptive_model,
    train_regime_adaptive_model,
    RegimeAdaptive_GRU_Enhanced_Attention_Inspect
)
from src.models.attention_gru import build_gru_model, train_gru_model
from src.models.dl_baselines import build_standard_attention_gru, build_vanilla_gru, train_dl_baseline


def set_seed(seed=42):
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_regime_evaluation(ticker: str = "AAPL", feature_level: str = "level2_returns",
                          time_step: int = 30, epochs: int = 15, seed: int = 42,
                          results_dir: str = "results"):
    set_seed(seed)
    print(f"\n==================================================================")
    print(f"REGIME-ADAPTIVE FORECASTING EVALUATION: {ticker}")
    print(f"Features: {feature_level} | Time Step: {time_step} | Seed: {seed}")
    print(f"==================================================================")

    # 1. Load data & engineer features
    raw_df = load_frozen_dataset(ticker)
    feat_df, feature_cols = engineer_financial_features(raw_df, level=feature_level)
    target_idx = feature_cols.index('Close')
    num_features = len(feature_cols)

    # 2. Chronological Cutoffs
    N = len(feat_df)
    train_end = int(N * 0.70)
    val_end = int(N * 0.85)

    # 3. Fit Regime Detector STRICTLY on Train
    print("Fitting Unsupervised GMM Regime Detector on training data...")
    regime_detector = MarketRegimeDetector(n_regimes=4, random_state=seed)
    regime_detector.fit(raw_df.iloc[:train_end])
    regime_labels_all, regime_probs_all, _ = regime_detector.predict_regimes(raw_df)

    # Align regime arrays to match feat_df index
    feat_dates = feat_df.index
    date_to_idx = {d: i for i, d in enumerate(raw_df.index)}
    aligned_indices = [date_to_idx[d] for d in feat_dates]
    regime_probs = regime_probs_all[aligned_indices]
    regime_labels = regime_labels_all[aligned_indices]

    # 4. Leakage-free sequence preparation
    datasets = prepare_datasets(
        df=feat_df,
        feature_cols=feature_cols,
        target_col='Close',
        time_step=time_step,
        train_ratio=0.70,
        val_ratio=0.15
    )

    X_train_seq = datasets['X_train']
    y_train = datasets['y_train']
    X_val_seq = datasets['X_val']
    y_val = datasets['y_val']
    X_test_seq = datasets['X_test']
    y_test = datasets['y_test']
    scaler = datasets['scaler']

    # Generate aligned regime vectors for target prediction timesteps
    # Each sequence i predicts at timestamp (i + time_step)
    test_dates = datasets['dates_test']
    val_dates = datasets['dates_val']
    train_dates = datasets['dates_train']

    date_to_feat_idx = {d: i for i, d in enumerate(feat_dates)}
    X_train_reg = np.array([regime_probs[date_to_feat_idx[d]] for d in train_dates], dtype=np.float32)
    X_val_reg = np.array([regime_probs[date_to_feat_idx[d]] for d in val_dates], dtype=np.float32)
    X_test_reg = np.array([regime_probs[date_to_feat_idx[d]] for d in test_dates], dtype=np.float32)
    test_regime_labels = np.array([regime_labels[date_to_feat_idx[d]] for d in test_dates])

    y_test_act = inverse_transform_target(scaler, y_test, target_idx, num_features)

    print(f"Dataset Split: Train={len(X_train_seq)} | Val={len(X_val_seq)} | Test={len(X_test_seq)}")
    import collections
    print(f"Test Set Regime Distribution: {dict(collections.Counter(test_regime_labels))}")

    # -------------------------------------------------------------
    # MODEL 1: PROPOSED REGIME-ADAPTIVE ATTENTION GRU
    # -------------------------------------------------------------
    print("\nTraining Model 1: Proposed Regime-Adaptive Attention GRU...")
    prop_reg_model, dur_reg, _ = train_regime_adaptive_model(
        X_train_seq, X_train_reg, y_train,
        X_val_seq, X_val_reg, y_val,
        time_step=time_step, epochs=epochs, batch_size=64
    )
    pred_reg_scaled = prop_reg_model.predict([X_test_seq, X_test_reg], verbose=0)
    pred_reg = inverse_transform_target(scaler, pred_reg_scaled, target_idx, num_features)

    # -------------------------------------------------------------
    # MODEL 2: PROPOSED STATIC ATTENTION GRU (Without Regime)
    # -------------------------------------------------------------
    print("Training Model 2: GRU-Enhanced Attention (Static lookback)...")
    att_model, dur_att, _ = train_gru_model(
        X_train_seq, y_train, X_val_seq, y_val,
        use_attention=True, time_step=time_step, epochs=epochs, batch_size=64
    )
    pred_att_scaled = att_model.predict(X_test_seq, verbose=0)
    pred_att = inverse_transform_target(scaler, pred_att_scaled, target_idx, num_features)

    # -------------------------------------------------------------
    # MODEL 3: STANDARD ATTENTION GRU (Bahdanau)
    # -------------------------------------------------------------
    print("Training Model 3: Standard Attention GRU...")
    std_att = build_standard_attention_gru(time_step, num_features)
    trained_std, dur_std, _ = train_dl_baseline(std_att, X_train_seq, y_train, X_val_seq, y_val, epochs=epochs)
    pred_std_scaled = trained_std.predict(X_test_seq, verbose=0)
    pred_std = inverse_transform_target(scaler, pred_std_scaled, target_idx, num_features)

    # -------------------------------------------------------------
    # MODEL 4: VANILLA GRU
    # -------------------------------------------------------------
    print("Training Model 4: Vanilla GRU...")
    v_gru = build_vanilla_gru(time_step, num_features)
    trained_vgru, dur_vgru, _ = train_dl_baseline(v_gru, X_train_seq, y_train, X_val_seq, y_val, epochs=epochs)
    pred_vgru_scaled = trained_vgru.predict(X_test_seq, verbose=0)
    pred_vgru = inverse_transform_target(scaler, pred_vgru_scaled, target_idx, num_features)

    # -------------------------------------------------------------
    # OVERALL TEST METRICS
    # -------------------------------------------------------------
    m_reg = compute_forecasting_metrics(y_test_act, pred_reg)
    m_att = compute_forecasting_metrics(y_test_act, pred_att)
    m_std = compute_forecasting_metrics(y_test_act, pred_std)
    m_vgru = compute_forecasting_metrics(y_test_act, pred_vgru)

    overall_df = pd.DataFrame([
        {"Architecture": "Proposed: Regime-Adaptive Attn GRU", "Test RMSE": m_reg['rmse'], "MAE": m_reg['mae'], "MAPE (%)": m_reg['mape'], "R2": m_reg['r2'], "MDA (%)": m_reg['directional_acc_pct']},
        {"Architecture": "Proposed: GRU-Enhanced Attn (Static)", "Test RMSE": m_att['rmse'], "MAE": m_att['mae'], "MAPE (%)": m_att['mape'], "R2": m_att['r2'], "MDA (%)": m_att['directional_acc_pct']},
        {"Architecture": "Baseline: Standard Attention GRU",     "Test RMSE": m_std['rmse'], "MAE": m_std['mae'], "MAPE (%)": m_std['mape'], "R2": m_std['r2'], "MDA (%)": m_std['directional_acc_pct']},
        {"Architecture": "Baseline: Vanilla GRU",               "Test RMSE": m_vgru['rmse'], "MAE": m_vgru['mae'], "MAPE (%)": m_vgru['mape'], "R2": m_vgru['r2'], "MDA (%)": m_vgru['directional_acc_pct']}
    ])

    print("\n==================================================================")
    print("OVERALL OUT-OF-SAMPLE TEST PERFORMANCE")
    print("==================================================================")
    print(overall_df.to_string(index=False))

    # -------------------------------------------------------------
    # REGIME-SPECIFIC BREAKDOWN (PHASE 12)
    # -------------------------------------------------------------
    regime_results = []
    models_dict = {
        "Regime-Adaptive Attn GRU": pred_reg,
        "GRU-Enhanced Attn": pred_att,
        "Standard Attention GRU": pred_std,
        "Vanilla GRU": pred_vgru
    }

    for regime_name in REGIME_NAMES:
        mask = (test_regime_labels == regime_name)
        count = int(np.sum(mask))
        if count < 5:
            continue
        y_act_sub = y_test_act[mask]
        row = {"Regime": regime_name, "Samples": count}
        for m_name, preds in models_dict.items():
            rmse = round(float(np.sqrt(np.mean((y_act_sub - preds[mask]) ** 2))), 4)
            row[f"{m_name} (RMSE)"] = rmse
        regime_results.append(row)

    regime_df = pd.DataFrame(regime_results)
    print("\n\n==================================================================")
    print("REGIME-SPECIFIC PERFORMANCE BREAKDOWN (PHASE 12 EVIDENCE)")
    print("==================================================================")
    print(regime_df.to_string(index=False))
    print("==================================================================\n")

    # Persist results
    os.makedirs(results_dir, exist_ok=True)
    overall_path = os.path.join(results_dir, f"regime_overall_{ticker}.csv")
    breakdown_path = os.path.join(results_dir, f"regime_breakdown_{ticker}.csv")
    overall_df.to_csv(overall_path, index=False)
    regime_df.to_csv(breakdown_path, index=False)
    print(f"Saved results to:\n  - {overall_path}\n  - {breakdown_path}")

    return overall_df, regime_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Regime-Adaptive Model vs Baselines")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--feature_level", type=str, default="level2_returns")
    parser.add_argument("--timestep", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_regime_evaluation(
        ticker=args.ticker,
        feature_level=args.feature_level,
        time_step=args.timestep,
        epochs=args.epochs,
        seed=args.seed
    )
