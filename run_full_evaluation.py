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
from src.regime_detector import MarketRegimeDetector
from src.preprocessing import prepare_datasets, inverse_transform_target
from src.evaluation import compute_forecasting_metrics, log_experiment
from src.statistical_tests import diebold_mariano_test, aggregate_seed_metrics
from src.backtest import FinancialBacktester
from src.walk_forward import run_walk_forward_validation

# Models
from src.models.regime_attention_gru import train_regime_adaptive_model
from src.models.attention_gru import train_gru_model
from src.models.dl_baselines import build_standard_attention_gru, build_vanilla_gru, train_dl_baseline


def set_seed(seed=42):
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_comprehensive_evaluation(ticker: str = "AAPL", feature_level: str = "level2_returns",
                                 time_step: int = 30, epochs: int = 10,
                                 seeds: list = [42, 101, 777], results_dir: str = "results"):
    print(f"\n==================================================================")
    print(f"COMPREHENSIVE RESEARCH EVALUATION ENGINE: {ticker}")
    print(f"Features: {feature_level} | Time Step: {time_step} | Epochs: {epochs} | Seeds: {seeds}")
    print(f"==================================================================")

    # 1. Load Data & Extract Features
    raw_df = load_frozen_dataset(ticker)
    feat_df, feature_cols = engineer_financial_features(raw_df, level=feature_level)
    target_idx = feature_cols.index('Close')
    num_features = len(feature_cols)

    # Fit Regime Detector on first 70%
    train_end = int(len(feat_df) * 0.70)
    reg_det = MarketRegimeDetector(n_regimes=4, random_state=42)
    reg_det.fit(raw_df.iloc[:train_end])
    reg_labels, reg_probs, _ = reg_det.predict_regimes(raw_df)

    feat_dates = feat_df.index
    d_map = {d: i for i, d in enumerate(raw_df.index)}
    aligned_probs = reg_probs[[d_map[d] for d in feat_dates]]

    datasets = prepare_datasets(
        df=feat_df, feature_cols=feature_cols,
        target_col='Close', time_step=time_step,
        train_ratio=0.70, val_ratio=0.15
    )

    X_train_seq = datasets['X_train']
    y_train = datasets['y_train']
    X_val_seq = datasets['X_val']
    y_val = datasets['y_val']
    X_test_seq = datasets['X_test']
    y_test = datasets['y_test']
    scaler = datasets['scaler']

    f_map = {d: i for i, d in enumerate(feat_dates)}
    X_train_reg = np.array([aligned_probs[f_map[d]] for d in datasets['dates_train']], dtype=np.float32)
    X_val_reg = np.array([aligned_probs[f_map[d]] for d in datasets['dates_val']], dtype=np.float32)
    X_test_reg = np.array([aligned_probs[f_map[d]] for d in datasets['dates_test']], dtype=np.float32)

    y_test_act = inverse_transform_target(scaler, y_test, target_idx, num_features)

    # -------------------------------------------------------------
    # SECTION 1: MULTI-SEED STATISTICAL STABILITY (PHASE 15)
    # -------------------------------------------------------------
    print("\n--- SECTION 1: MULTI-SEED REPEATABILITY (SEEDS: {}) ---".format(seeds))
    proposed_seed_metrics = []
    std_att_seed_metrics = []
    last_preds_proposed = None
    last_preds_std_att = None
    last_preds_vgru = None

    for s in seeds:
        set_seed(s)
        # Train Proposed
        m_prop, _, _ = train_regime_adaptive_model(
            X_train_seq, X_train_reg, y_train,
            X_val_seq, X_val_reg, y_val,
            time_step=time_step, epochs=epochs
        )
        p_prop = inverse_transform_target(scaler, m_prop.predict([X_test_seq, X_test_reg], verbose=0), target_idx, num_features)
        last_preds_proposed = p_prop
        proposed_seed_metrics.append(compute_forecasting_metrics(y_test_act, p_prop))

        # Train Standard Attention Baseline
        std_m = build_standard_attention_gru(time_step, num_features)
        m_std, _, _ = train_dl_baseline(std_m, X_train_seq, y_train, X_val_seq, y_val, epochs=epochs)
        p_std = inverse_transform_target(scaler, m_std.predict(X_test_seq, verbose=0), target_idx, num_features)
        last_preds_std_att = p_std
        std_att_seed_metrics.append(compute_forecasting_metrics(y_test_act, p_std))

    # Single Vanilla GRU run for comparison
    v_m = build_vanilla_gru(time_step, num_features)
    m_vgru, _, _ = train_dl_baseline(v_m, X_train_seq, y_train, X_val_seq, y_val, epochs=epochs)
    last_preds_vgru = inverse_transform_target(scaler, m_vgru.predict(X_test_seq, verbose=0), target_idx, num_features)

    prop_agg = aggregate_seed_metrics(proposed_seed_metrics)
    std_agg = aggregate_seed_metrics(std_att_seed_metrics)

    multi_seed_df = pd.DataFrame([
        {
            "Architecture": "Proposed: Regime-Adaptive Attn GRU",
            "Test RMSE (Mean +/- Std)": prop_agg['rmse_mean_std'],
            "Test MAE (Mean +/- Std)": prop_agg['mae_mean_std'],
            "Test MAPE (%)": f"{prop_agg['mape_mean']:.2f}%",
            "Test R2": f"{prop_agg['r2_mean']:.4f}",
            "MDA (%)": f"{prop_agg['directional_acc_pct_mean']:.1f}%"
        },
        {
            "Architecture": "Baseline: Standard Attention GRU",
            "Test RMSE (Mean +/- Std)": std_agg['rmse_mean_std'],
            "Test MAE (Mean +/- Std)": std_agg['mae_mean_std'],
            "Test MAPE (%)": f"{std_agg['mape_mean']:.2f}%",
            "Test R2": f"{std_agg['r2_mean']:.4f}",
            "MDA (%)": f"{std_agg['directional_acc_pct_mean']:.1f}%"
        }
    ])
    print(multi_seed_df.to_string(index=False))

    # -------------------------------------------------------------
    # SECTION 2: DIEBOLD-MARIANO HYPOTHESIS TESTS (PHASE 15)
    # -------------------------------------------------------------
    print("\n--- SECTION 2: DIEBOLD-MARIANO STATISTICAL TESTS ---")
    dm_stat_std, p_val_std = diebold_mariano_test(y_test_act, last_preds_proposed, last_preds_std_att, criterion="MSE")
    dm_stat_vgru, p_val_vgru = diebold_mariano_test(y_test_act, last_preds_proposed, last_preds_vgru, criterion="MSE")

    dm_df = pd.DataFrame([
        {
            "Comparison": "Proposed vs Standard Attention GRU",
            "Loss Criterion": "MSE",
            "DM Statistic": dm_stat_std,
            "p-value": p_val_std,
            "Significance (alpha=0.05)": "SIGNIFICANT (p < 0.05)" if p_val_std < 0.05 else "Not Significant"
        },
        {
            "Comparison": "Proposed vs Vanilla GRU",
            "Loss Criterion": "MSE",
            "DM Statistic": dm_stat_vgru,
            "p-value": p_val_vgru,
            "Significance (alpha=0.05)": "SIGNIFICANT (p < 0.05)" if p_val_vgru < 0.05 else "Not Significant"
        }
    ])
    print(dm_df.to_string(index=False))

    # -------------------------------------------------------------
    # SECTION 3: WALK-FORWARD VALIDATION (PHASE 10)
    # -------------------------------------------------------------
    print("\n--- SECTION 3: 3-FOLD EXPANDING WALK-FORWARD VALIDATION ---")
    wf_df, wf_agg = run_walk_forward_validation(feat_df, feature_cols, time_step=time_step, epochs=epochs, n_folds=3, seed=42)
    print(wf_df.to_string(index=False))
    print(f"\nWalk-Forward Summary -> Mean RMSE: {wf_agg['mean_rmse']} +/- {wf_agg['std_rmse']} | Mean R2: {wf_agg['mean_r2']} | Mean MDA: {wf_agg['mean_mda']}%")

    # -------------------------------------------------------------
    # SECTION 4: FINANCIAL BACKTESTING SIMULATION (PHASE 17)
    # -------------------------------------------------------------
    print("\n--- SECTION 4: FINANCIAL STRATEGY BACKTESTING ---")
    backtester = FinancialBacktester(threshold=0.0005, transaction_cost_bps=5.0)
    bt_results = backtester.simulate(y_test_act, last_preds_proposed, dates=datasets['dates_test'])
    bt_m = bt_results['metrics']

    bt_df = pd.DataFrame([
        {"Metric": "Cumulative Return", "Proposed Strategy": f"{bt_m['Strategy Return (%)']}%", "Buy & Hold Benchmark": f"{bt_m['Buy & Hold Return (%)']}%"},
        {"Metric": "Annualized Return", "Proposed Strategy": f"{bt_m['Strategy Annualized (%)']}%", "Buy & Hold Benchmark": f"{bt_m['Buy & Hold Annualized (%)']}%"},
        {"Metric": "Annualized Sharpe Ratio", "Proposed Strategy": f"{bt_m['Strategy Sharpe']}", "Buy & Hold Benchmark": f"{bt_m['Buy & Hold Sharpe']}"},
        {"Metric": "Maximum Drawdown", "Proposed Strategy": f"{bt_m['Strategy Max Drawdown (%)']}%", "Buy & Hold Benchmark": f"{bt_m['Buy & Hold Max Drawdown (%)']}%"},
        {"Metric": "Strategy Win Rate", "Proposed Strategy": f"{bt_m['Win Rate (%)']}%", "Buy & Hold Benchmark": "N/A"},
        {"Metric": "Active Trading Days", "Proposed Strategy": f"{bt_m['Trade Days']}/{bt_m['Total Days']}", "Buy & Hold Benchmark": f"{bt_m['Total Days']}/{bt_m['Total Days']}"}
    ])
    print(bt_df.to_string(index=False))

    # Persist all 4 tables to results/
    os.makedirs(results_dir, exist_ok=True)
    multi_seed_df.to_csv(os.path.join(results_dir, f"table_multiseed_{ticker}.csv"), index=False)
    dm_df.to_csv(os.path.join(results_dir, f"table_diebold_mariano_{ticker}.csv"), index=False)
    wf_df.to_csv(os.path.join(results_dir, f"table_walk_forward_{ticker}.csv"), index=False)
    bt_df.to_csv(os.path.join(results_dir, f"table_backtesting_{ticker}.csv"), index=False)

    print("\n==================================================================")
    print("ALL 4 RESEARCH PUBLICATION TABLES GENERATED & PERSISTED TO results/")
    print("==================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Research Evaluation (Statistical, Walk-Forward, Backtest)")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--feature_level", type=str, default="level2_returns")
    parser.add_argument("--timestep", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=8)

    args = parser.parse_args()
    run_comprehensive_evaluation(
        ticker=args.ticker,
        feature_level=args.feature_level,
        time_step=args.timestep,
        epochs=args.epochs,
        seeds=[42, 101, 777]
    )
