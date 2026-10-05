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
from src.preprocessing import prepare_datasets, inverse_transform_target
from src.evaluation import compute_forecasting_metrics, log_experiment

# Models
from src.models.classical_baselines import NaiveForecaster, RandomForestForecaster, XGBoostForecaster
from src.models.dl_baselines import (
    build_vanilla_gru,
    build_vanilla_lstm,
    build_standard_attention_gru,
    build_transformer_model,
    train_dl_baseline
)
from src.models.attention_gru import build_gru_model, train_gru_model


def set_seed(seed=42):
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def benchmark_all_models(ticker: str = "AAPL", feature_level: str = "level2_returns",
                         time_step: int = 30, epochs: int = 15, seed: int = 42,
                         results_dir: str = "results"):
    set_seed(seed)
    print(f"\n==================================================================")
    print(f"BENCHMARKING ALL MODELS: Ticker={ticker} | Feature Tier={feature_level}")
    print(f"Time Step={time_step} | Max Epochs={epochs} | Seed={seed}")
    print(f"==================================================================")

    # 1. Load frozen data & engineer features
    raw_df = load_frozen_dataset(ticker)
    feat_df, feature_cols = engineer_financial_features(raw_df, level=feature_level)
    print(f"Data: {len(feat_df)} rows | {len(feature_cols)} features ({feature_level})")

    # 2. Leakage-free chronological preparation
    datasets = prepare_datasets(
        df=feat_df,
        feature_cols=feature_cols,
        target_col='Close',
        time_step=time_step,
        train_ratio=0.70,
        val_ratio=0.15
    )

    X_train, y_train = datasets['X_train'], datasets['y_train']
    X_val, y_val = datasets['X_val'], datasets['y_val']
    X_test, y_test = datasets['X_test'], datasets['y_test']
    scaler = datasets['scaler']
    target_idx = datasets['target_idx']
    num_features = X_train.shape[2]

    # Ground truth inverse scaled
    y_test_act = inverse_transform_target(scaler, y_test, target_idx, num_features)
    y_val_act = inverse_transform_target(scaler, y_val, target_idx, num_features)
    y_train_act = inverse_transform_target(scaler, y_train, target_idx, num_features)

    results = []

    def evaluate_and_record(name: str, y_pred_scaled: np.ndarray, duration: float, params: int = 0):
        y_test_pred = inverse_transform_target(scaler, y_pred_scaled, target_idx, num_features)
        metrics = compute_forecasting_metrics(y_test_act, y_test_pred)
        record = {
            "Model": name,
            "Parameters": params if params > 0 else "N/A",
            "Train Time (s)": round(duration, 2),
            "Test RMSE": metrics['rmse'],
            "Test MAE": metrics['mae'],
            "Test MAPE (%)": metrics['mape'],
            "Test R2": metrics['r2'],
            "MDA (%)": metrics['directional_acc_pct']
        }
        results.append(record)
        print(f"  [OK] {name:<30} -> RMSE: {metrics['rmse']:<8} | MAE: {metrics['mae']:<8} | R2: {metrics['r2']:<7} | MDA: {metrics['directional_acc_pct']}% ({round(duration, 1)}s)")

        # Log to registry
        run_meta = {
            'run_id': f"{ticker}_{name}_{seed}_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'feature_level': feature_level,
            'model': name,
            'seed': seed,
            'time_step': time_step,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'training_seconds': round(duration, 2)
        }
        log_experiment(results_dir, run_meta, {}, {}, metrics)

    # -------------------------------------------------------------
    # 1. NAIVE (PERSISTENCE) BASELINE
    # -------------------------------------------------------------
    t0 = time.time()
    naive = NaiveForecaster()
    naive_pred_scaled = naive.predict(X_test, target_idx)
    evaluate_and_record("1. Naive (Persistence)", naive_pred_scaled, time.time() - t0, 0)

    # -------------------------------------------------------------
    # 2. RANDOM FOREST
    # -------------------------------------------------------------
    t0 = time.time()
    rf = RandomForestForecaster(n_estimators=100, random_state=seed)
    rf.fit(X_train, y_train)
    rf_pred_scaled = rf.predict(X_test)
    evaluate_and_record("2. Random Forest", rf_pred_scaled, time.time() - t0, 0)

    # -------------------------------------------------------------
    # 3. XGBOOST
    # -------------------------------------------------------------
    t0 = time.time()
    xgb = XGBoostForecaster(n_estimators=100, random_state=seed)
    xgb.fit(X_train, y_train)
    xgb_pred_scaled = xgb.predict(X_test)
    evaluate_and_record("3. XGBoost", xgb_pred_scaled, time.time() - t0, 0)

    # -------------------------------------------------------------
    # 4. VANILLA GRU
    # -------------------------------------------------------------
    v_gru = build_vanilla_gru(time_step, num_features, units=64, dense_units=32)
    trained_v_gru, dur, _ = train_dl_baseline(v_gru, X_train, y_train, X_val, y_val, epochs=epochs, ckpt_name="v_gru")
    evaluate_and_record("4. Vanilla GRU", trained_v_gru.predict(X_test, verbose=0), dur, v_gru.count_params())

    # -------------------------------------------------------------
    # 5. VANILLA LSTM
    # -------------------------------------------------------------
    v_lstm = build_vanilla_lstm(time_step, num_features, units=64, dense_units=32)
    trained_v_lstm, dur, _ = train_dl_baseline(v_lstm, X_train, y_train, X_val, y_val, epochs=epochs, ckpt_name="v_lstm")
    evaluate_and_record("5. Vanilla LSTM", trained_v_lstm.predict(X_test, verbose=0), dur, v_lstm.count_params())

    # -------------------------------------------------------------
    # 6. STANDARD ATTENTION GRU (BAHDANAU-STYLE)
    # -------------------------------------------------------------
    std_att = build_standard_attention_gru(time_step, num_features, units=64, att_units=32, dense_units=32)
    trained_std_att, dur, _ = train_dl_baseline(std_att, X_train, y_train, X_val, y_val, epochs=epochs, ckpt_name="std_att")
    evaluate_and_record("6. Standard Attention GRU", trained_std_att.predict(X_test, verbose=0), dur, std_att.count_params())

    # -------------------------------------------------------------
    # 7. TRANSFORMER ENCODER
    # -------------------------------------------------------------
    trans = build_transformer_model(time_step, num_features, d_model=64, num_heads=4, ff_dim=64)
    trained_trans, dur, _ = train_dl_baseline(trans, X_train, y_train, X_val, y_val, epochs=epochs, ckpt_name="transformer")
    evaluate_and_record("7. Transformer", trained_trans.predict(X_test, verbose=0), dur, trans.count_params())

    # -------------------------------------------------------------
    # 8. PROPOSED: GRU-ENHANCED ATTENTION
    # -------------------------------------------------------------
    prop_model, dur, _ = train_gru_model(
        X_train, y_train, X_val, y_val,
        use_attention=True,
        time_step=time_step,
        epochs=epochs,
        batch_size=64,
        gru_units=64,
        att_units=32,
        dense_units=32
    )
    evaluate_and_record("8. Proposed GRU-Enhanced Attn", prop_model.predict(X_test, verbose=0), dur, prop_model.count_params())

    # Format Summary Table
    df_res = pd.DataFrame(results)
    print("\n\n==================================================================")
    print(f"COMPLETE RESEARCH BASELINE COMPARISON TABLE ({ticker})")
    print("==================================================================")
    print(df_res.to_string(index=False))
    print("==================================================================\n")

    # Persist summary CSV
    summary_path = os.path.join(results_dir, f"baseline_comparison_{ticker}.csv")
    df_res.to_csv(summary_path, index=False)
    print(f"Saved benchmark summary table to: {summary_path}")

    return df_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Research Baseline Benchmark")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker (e.g. AAPL, MSFT, RELIANCE.NS, _NSEI)")
    parser.add_argument("--feature_level", type=str, default="level2_returns", choices=["level1_price", "level2_returns", "level3_volatility", "level4_full"])
    parser.add_argument("--timestep", type=int, default=30, help="Lookback window")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs for neural models")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")

    args = parser.parse_args()
    benchmark_all_models(
        ticker=args.ticker,
        feature_level=args.feature_level,
        time_step=args.timestep,
        epochs=args.epochs,
        seed=args.seed
    )
