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
from src.features import engineer_financial_features, FEATURE_LEVELS
from src.preprocessing import prepare_datasets, inverse_transform_target
from src.models.attention_gru import train_gru_model
from src.evaluation import compute_forecasting_metrics, log_experiment


def set_seed(seed=42):
    import random
    import numpy as np
    import tensorflow as tf
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_single_experiment(ticker: str, feature_level: str = "level4_full",
                           use_attention: bool = True, time_step: int = 60,
                           epochs: int = 25, seed: int = 42, results_dir: str = "results"):
    set_seed(seed)
    model_name = "Proposed_Attention_GRU" if use_attention else "Ablation_GAP_GRU"

    print(f"\n==================================================================")
    print(f"RUNNING: Ticker={ticker} | Features={feature_level} | Model={model_name} | Seed={seed}")
    print(f"==================================================================")

    # 1. Load frozen data
    raw_df = load_frozen_dataset(ticker)

    # 2. Engineer features
    feat_df, feature_cols = engineer_financial_features(raw_df, level=feature_level)
    print(f"Feature set size: {len(feature_cols)} features ({feature_level})")

    # 3. Leakage-free preparation (70% train, 15% val, 15% test)
    data_dict = prepare_datasets(
        df=feat_df,
        feature_cols=feature_cols,
        target_col='Close',
        time_step=time_step,
        train_ratio=0.70,
        val_ratio=0.15
    )

    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']
    scaler = data_dict['scaler']
    target_idx = data_dict['target_idx']

    print(f"Data windows: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # 4. Train model (Early stopping on X_val, blind X_test)
    model, duration, history = train_gru_model(
        X_train, y_train, X_val, y_val,
        use_attention=use_attention,
        time_step=time_step,
        epochs=epochs,
        batch_size=64
    )
    print(f"Training completed in {duration:.2f}s (Epochs: {len(history.history['loss'])})")

    # 5. Evaluate out-of-sample
    num_feat = X_train.shape[2]
    y_train_pred = inverse_transform_target(scaler, model.predict(X_train, verbose=0), target_idx, num_feat)
    y_val_pred   = inverse_transform_target(scaler, model.predict(X_val, verbose=0), target_idx, num_feat)
    y_test_pred  = inverse_transform_target(scaler, model.predict(X_test, verbose=0), target_idx, num_feat)

    y_train_act = inverse_transform_target(scaler, y_train, target_idx, num_feat)
    y_val_act   = inverse_transform_target(scaler, y_val, target_idx, num_feat)
    y_test_act  = inverse_transform_target(scaler, y_test, target_idx, num_feat)

    train_m = compute_forecasting_metrics(y_train_act, y_train_pred)
    val_m   = compute_forecasting_metrics(y_val_act, y_val_pred)
    test_m  = compute_forecasting_metrics(y_test_act, y_test_pred)

    run_meta = {
        'run_id': f"{ticker}_{feature_level}_{'att' if use_attention else 'noatt'}_{seed}_{int(time.time())}",
        'timestamp': datetime.now().isoformat(),
        'ticker': ticker,
        'feature_level': feature_level,
        'num_features': len(feature_cols),
        'model': model_name,
        'seed': seed,
        'time_step': time_step,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'training_seconds': round(duration, 2)
    }

    csv_path = log_experiment(results_dir, run_meta, train_m, val_m, test_m)

    print(f"Test RMSE: {test_m['rmse']} | MAE: {test_m['mae']} | MAPE: {test_m['mape']}% | R2: {test_m['r2']} | MDA: {test_m['directional_acc_pct']}%")
    return run_meta, test_m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Research Experiment Runner")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker (e.g. AAPL, MSFT, RELIANCE.NS, _NSEI)")
    parser.add_argument("--feature_level", type=str, default="level4_full", choices=list(FEATURE_LEVELS.keys()), help="Feature engineering tier")
    parser.add_argument("--no_attention", action="store_true", help="Run GAP ablation without attention layer")
    parser.add_argument("--timestep", type=int, default=60, help="Lookback window")
    parser.add_argument("--epochs", type=int, default=25, help="Max training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    parser.add_argument("--run_feature_ablation", action="store_true", help="Run all 4 feature levels consecutively")

    args = parser.parse_args()

    if args.run_feature_ablation:
        print(f"\n========================================================")
        print(f"STARTING FEATURE ABLATION STUDY: {args.ticker}")
        print(f"Testing Level 1 -> Level 2 -> Level 3 -> Level 4")
        print(f"========================================================")
        results = []
        for fl in ["level1_price", "level2_returns", "level3_volatility", "level4_full"]:
            meta, test_m = run_single_experiment(
                ticker=args.ticker,
                feature_level=fl,
                use_attention=not args.no_attention,
                time_step=args.timestep,
                epochs=args.epochs,
                seed=args.seed
            )
            results.append({
                "Feature Tier": fl,
                "Features": meta['num_features'],
                "Test RMSE": test_m['rmse'],
                "Test MAE": test_m['mae'],
                "Test MAPE (%)": test_m['mape'],
                "Test R²": test_m['r2'],
                "MDA (%)": test_m['directional_acc_pct']
            })

        print("\n\n=================== FEATURE ABLATION SUMMARY ===================")
        res_df = pd.DataFrame(results)
        print(res_df.to_string(index=False))
        print("================================================================\n")
    else:
        run_single_experiment(
            ticker=args.ticker,
            feature_level=args.feature_level,
            use_attention=not args.no_attention,
            time_step=args.timestep,
            epochs=args.epochs,
            seed=args.seed
        )
