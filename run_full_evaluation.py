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
from src.statistical_tests import diebold_mariano_test, aggregate_seed_metrics
from src.backtest import FinancialBacktester
from src.walk_forward import run_walk_forward_validation

# Model Builders & Training Routines
from src.models.regime_attention_gru import (
    build_regime_adaptive_model,
    train_regime_adaptive_model
)
from src.models.attention_gru import (
    build_gru_model,
    train_gru_model
)
from src.models.dl_baselines import (
    build_vanilla_gru,
    build_standard_attention_gru,
    build_regime_feature_gru,
    train_dl_baseline
)


def set_seed(seed=42):
    """Enforces deterministic initialization across Python, NumPy, and TensorFlow."""
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_comprehensive_evaluation(ticker: str = "AAPL",
                                 feature_level: str = "level2_returns",
                                 time_step: int = 30,
                                 epochs: int = 15,
                                 seeds: list = [42, 101, 2024, 777, 999],
                                 results_dir: str = "results"):
    print(f"\n==================================================================")
    print(f"AUTHORITATIVE RESEARCH EVALUATION ENGINE: {ticker}")
    print(f"Features: {feature_level} | Time Step: {time_step} | Epochs: {epochs}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"==================================================================")

    # 1. Load Data & Extract Features
    raw_df = load_frozen_dataset(ticker)
    feat_df, feature_cols = engineer_financial_features(raw_df, level=feature_level)
    target_idx = feature_cols.index('Close')
    num_features = len(feature_cols)

    # 2. Strict Train-Only Regime Detector Fit (70% Partition)
    train_end = int(len(feat_df) * 0.70)
    reg_det = MarketRegimeDetector(n_regimes=4, random_state=42)
    reg_det.fit(raw_df.iloc[:train_end])
    reg_labels_all, reg_probs_all, _ = reg_det.predict_regimes(raw_df)

    feat_dates = feat_df.index
    d_map = {d: i for i, d in enumerate(raw_df.index)}
    aligned_probs = reg_probs_all[[d_map[d] for d in feat_dates]]
    aligned_labels = reg_labels_all[[d_map[d] for d in feat_dates]]

    # -------------------------------------------------------------
    # SECTION 0: REGIME CHARACTERIZATION VALIDATION (RESEARCH AUDIT)
    # -------------------------------------------------------------
    print("\n--- SECTION 0: REGIME CHARACTERIZATION VALIDATION ---")
    regime_char_df = reg_det.compute_regime_characterization(raw_df, reg_labels_all)
    print(regime_char_df.to_string(index=False))

    # 3. Chronological Train / Val / Test Partition
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
    test_regimes = np.array([aligned_labels[f_map[d]] for d in datasets['dates_test']])

    y_test_act = inverse_transform_target(scaler, y_test, target_idx, num_features)

    # -------------------------------------------------------------
    # SECTION 1: SYMMETRICAL 5-MODEL MULTI-SEED ABLATION
    # -------------------------------------------------------------
    print(f"\n--- SECTION 1: SYMMETRICAL 5-MODEL ABLATION ({len(seeds)} SEEDS) ---")
    model_keys = [
        "Model A: Vanilla GRU",
        "Model B: Standard Attention GRU",
        "Model C: GRU + Enhanced Attention (Static)",
        "Model D: Regime-Feature GRU (No Attention)",
        "Model E: Proposed Regime-Adaptive Attention GRU"
    ]

    seed_metrics = {k: [] for k in model_keys}
    seed_preds = {k: None for k in model_keys}

    for s_idx, s in enumerate(seeds):
        print(f"\n>> Running Evaluation Seed {s} ({s_idx + 1}/{len(seeds)})...")

        # Model A: Vanilla GRU
        set_seed(s)
        m_a = build_vanilla_gru(time_step, num_features)
        m_a, _, _ = train_dl_baseline(m_a, X_train_seq, y_train, X_val_seq, y_val, epochs=epochs, ckpt_name=f"v_gru_s{s}")
        p_a = inverse_transform_target(scaler, m_a.predict(X_test_seq, verbose=0), target_idx, num_features)
        seed_metrics["Model A: Vanilla GRU"].append(compute_forecasting_metrics(y_test_act, p_a))
        seed_preds["Model A: Vanilla GRU"] = p_a

        # Model B: Standard Attention GRU
        set_seed(s)
        m_b = build_standard_attention_gru(time_step, num_features)
        m_b, _, _ = train_dl_baseline(m_b, X_train_seq, y_train, X_val_seq, y_val, epochs=epochs, ckpt_name=f"std_att_s{s}")
        p_b = inverse_transform_target(scaler, m_b.predict(X_test_seq, verbose=0), target_idx, num_features)
        seed_metrics["Model B: Standard Attention GRU"].append(compute_forecasting_metrics(y_test_act, p_b))
        seed_preds["Model B: Standard Attention GRU"] = p_b

        # Model C: GRU + Enhanced Attention (Static)
        set_seed(s)
        m_c, _, _ = train_gru_model(X_train_seq, y_train, X_val_seq, y_val, use_attention=True,
                                    time_step=time_step, epochs=epochs, ckpt_name=f"enh_att_s{s}")
        p_c = inverse_transform_target(scaler, m_c.predict(X_test_seq, verbose=0), target_idx, num_features)
        seed_metrics["Model C: GRU + Enhanced Attention (Static)"].append(compute_forecasting_metrics(y_test_act, p_c))
        seed_preds["Model C: GRU + Enhanced Attention (Static)"] = p_c

        # Model D: Regime-Feature GRU (Control: direct features, no attention)
        set_seed(s)
        m_d = build_regime_feature_gru(time_step, num_features, num_regimes=4)
        m_d, _, _ = train_dl_baseline(m_d, [X_train_seq, X_train_reg], y_train,
                                      [X_val_seq, X_val_reg], y_val,
                                      epochs=epochs, ckpt_name=f"reg_feat_s{s}")
        p_d = inverse_transform_target(scaler, m_d.predict([X_test_seq, X_test_reg], verbose=0), target_idx, num_features)
        seed_metrics["Model D: Regime-Feature GRU (No Attention)"].append(compute_forecasting_metrics(y_test_act, p_d))
        seed_preds["Model D: Regime-Feature GRU (No Attention)"] = p_d

        # Model E: Proposed Regime-Adaptive Attention GRU
        set_seed(s)
        m_e, _, _ = train_regime_adaptive_model(
            X_train_seq, X_train_reg, y_train,
            X_val_seq, X_val_reg, y_val,
            time_step=time_step, epochs=epochs, ckpt_name=f"reg_adapt_s{s}"
        )
        p_e = inverse_transform_target(scaler, m_e.predict([X_test_seq, X_test_reg], verbose=0), target_idx, num_features)
        seed_metrics["Model E: Proposed Regime-Adaptive Attention GRU"].append(compute_forecasting_metrics(y_test_act, p_e))
        seed_preds["Model E: Proposed Regime-Adaptive Attention GRU"] = p_e

    # Compile Multi-Seed Symmetrical Summary
    multi_seed_rows = []
    for k in model_keys:
        agg = aggregate_seed_metrics(seed_metrics[k])
        multi_seed_rows.append({
            "Architecture": k,
            "Test RMSE (Mean +/- Std)": agg['rmse_mean_std'],
            "Test MAE (Mean +/- Std)": agg['mae_mean_std'],
            "Test MAPE (%)": f"{agg['mape_mean']:.2f}%",
            "Test R2": f"{agg['r2_mean']:.4f}",
            "MDA (%)": f"{agg['directional_acc_pct_mean']:.1f}%"
        })
    multi_seed_df = pd.DataFrame(multi_seed_rows)
    print("\n--- MULTI-SEED SUMMARY TABLE ---")
    print(multi_seed_df.to_string(index=False))

    # -------------------------------------------------------------
    # SECTION 2: REGIME-SPECIFIC BREAKDOWN TABLE
    # -------------------------------------------------------------
    print("\n--- SECTION 2: TEST PERFORMANCE BY DETECTED REGIME (RMSE) ---")
    regime_breakdown_rows = []
    for r_name in REGIME_NAMES:
        mask = (test_regimes == r_name)
        n_samples = int(np.sum(mask))
        if n_samples == 0:
            continue
        row = {"Regime": r_name, "Test Samples": n_samples}
        for k in model_keys:
            pred = seed_preds[k]
            r_rmse = float(np.sqrt(np.mean((y_test_act[mask] - pred[mask]) ** 2)))
            short_name = k.split(":")[0].strip()
            row[f"{short_name} RMSE"] = round(r_rmse, 3)
        regime_breakdown_rows.append(row)
    regime_breakdown_df = pd.DataFrame(regime_breakdown_rows)
    print(regime_breakdown_df.to_string(index=False))

    # -------------------------------------------------------------
    # SECTION 3: DIEBOLD-MARIANO HYPOTHESIS TESTS (NEWEY-WEST + HLN)
    # -------------------------------------------------------------
    print("\n--- SECTION 3: DIEBOLD-MARIANO HYPOTHESIS TESTS (PROPOSED VS BASELINES) ---")
    dm_rows = []
    pred_proposed = seed_preds["Model E: Proposed Regime-Adaptive Attention GRU"]

    comparisons = [
        ("Proposed (Model E) vs Vanilla GRU (Model A)", seed_preds["Model A: Vanilla GRU"]),
        ("Proposed (Model E) vs Standard Attention GRU (Model B)", seed_preds["Model B: Standard Attention GRU"]),
        ("Proposed (Model E) vs Enhanced Attention GRU (Model C)", seed_preds["Model C: GRU + Enhanced Attention (Static)"]),
        ("Proposed (Model E) vs Regime-Feature GRU (Model D)", seed_preds["Model D: Regime-Feature GRU (No Attention)"])
    ]

    for comp_name, baseline_pred in comparisons:
        dm_stat, p_val_raw, p_val_str = diebold_mariano_test(y_test_act, pred_proposed, baseline_pred, h=1, criterion="MSE")
        dm_rows.append({
            "Pairwise Comparison": comp_name,
            "Loss Criterion": "MSE",
            "HLN-Adjusted DM Stat": dm_stat,
            "p-value": p_val_str,
            "Significance (alpha=0.05)": "SIGNIFICANT (p < 0.05)" if p_val_raw < 0.05 else "Not Significant"
        })
    dm_df = pd.DataFrame(dm_rows)
    print(dm_df.to_string(index=False))

    # -------------------------------------------------------------
    # SECTION 4: 3-FOLD EXPANDING WALK-FORWARD VALIDATION
    # -------------------------------------------------------------
    print("\n--- SECTION 4: 3-FOLD EXPANDING WALK-FORWARD VALIDATION ---")
    wf_df, wf_agg = run_walk_forward_validation(feat_df, feature_cols, time_step=time_step, epochs=epochs, seed=42)
    print(wf_df.to_string(index=False))
    print(f"\nWalk-Forward Summary -> Mean RMSE: {wf_agg['mean_rmse']} +/- {wf_agg['std_rmse']} | Mean R2: {wf_agg['mean_r2']} | Mean MDA: {wf_agg['mean_mda']}%")

    # -------------------------------------------------------------
    # SECTION 5: FINANCIAL BACKTESTING & RISK PROFILE (PHASE 17)
    # -------------------------------------------------------------
    print("\n--- SECTION 5: FINANCIAL STRATEGY BACKTESTING ---")
    backtester = FinancialBacktester(threshold=0.0005, transaction_cost_bps=5.0)
    
    bt_prop = backtester.simulate(y_test_act, pred_proposed, dates=datasets['dates_test'])['metrics']
    bt_vgru = backtester.simulate(y_test_act, seed_preds["Model A: Vanilla GRU"], dates=datasets['dates_test'])['metrics']
    bt_regfeat = backtester.simulate(y_test_act, seed_preds["Model D: Regime-Feature GRU (No Attention)"], dates=datasets['dates_test'])['metrics']

    bt_df = pd.DataFrame([
        {"Metric": "Cumulative Return", "Proposed (Model E)": f"{bt_prop['Strategy Return (%)']}%", "Vanilla GRU (Model A)": f"{bt_vgru['Strategy Return (%)']}%", "Regime-Feature (Model D)": f"{bt_regfeat['Strategy Return (%)']}%", "Buy & Hold Benchmark": f"{bt_prop['Buy & Hold Return (%)']}%"},
        {"Metric": "Annualized Return", "Proposed (Model E)": f"{bt_prop['Strategy Annualized (%)']}%", "Vanilla GRU (Model A)": f"{bt_vgru['Strategy Annualized (%)']}%", "Regime-Feature (Model D)": f"{bt_regfeat['Strategy Annualized (%)']}%", "Buy & Hold Benchmark": f"{bt_prop['Buy & Hold Annualized (%)']}%"},
        {"Metric": "Annualized Sharpe", "Proposed (Model E)": f"{bt_prop['Strategy Sharpe']}", "Vanilla GRU (Model A)": f"{bt_vgru['Strategy Sharpe']}", "Regime-Feature (Model D)": f"{bt_regfeat['Strategy Sharpe']}", "Buy & Hold Benchmark": f"{bt_prop['Buy & Hold Sharpe']}"},
        {"Metric": "Maximum Drawdown", "Proposed (Model E)": f"{bt_prop['Strategy Max Drawdown (%)']}%", "Vanilla GRU (Model A)": f"{bt_vgru['Strategy Max Drawdown (%)']}%", "Regime-Feature (Model D)": f"{bt_regfeat['Strategy Max Drawdown (%)']}%", "Buy & Hold Benchmark": f"{bt_prop['Buy & Hold Max Drawdown (%)']}%"},
        {"Metric": "Win Rate", "Proposed (Model E)": f"{bt_prop['Win Rate (%)']}%", "Vanilla GRU (Model A)": f"{bt_vgru['Win Rate (%)']}%", "Regime-Feature (Model D)": f"{bt_regfeat['Win Rate (%)']}%", "Buy & Hold Benchmark": "N/A"},
        {"Metric": "Active Trading Days", "Proposed (Model E)": f"{bt_prop['Trade Days']}/{bt_prop['Total Days']}", "Vanilla GRU (Model A)": f"{bt_vgru['Trade Days']}/{bt_vgru['Total Days']}", "Regime-Feature (Model D)": f"{bt_regfeat['Trade Days']}/{bt_regfeat['Total Days']}", "Buy & Hold Benchmark": f"{bt_prop['Total Days']}/{bt_prop['Total Days']}"}
    ])
    print(bt_df.to_string(index=False))

    # Persist all audited research tables to results/
    os.makedirs(results_dir, exist_ok=True)
    regime_char_df.to_csv(os.path.join(results_dir, f"table_regime_characterization_{ticker}.csv"), index=False)
    multi_seed_df.to_csv(os.path.join(results_dir, f"table_multiseed_{ticker}.csv"), index=False)
    regime_breakdown_df.to_csv(os.path.join(results_dir, f"table_regime_breakdown_{ticker}.csv"), index=False)
    dm_df.to_csv(os.path.join(results_dir, f"table_diebold_mariano_{ticker}.csv"), index=False)
    wf_df.to_csv(os.path.join(results_dir, f"table_walk_forward_{ticker}.csv"), index=False)
    bt_df.to_csv(os.path.join(results_dir, f"table_backtesting_{ticker}.csv"), index=False)

    print("\n==================================================================")
    print("ALL 6 AUDITED RESEARCH PUBLICATION TABLES GENERATED & PERSISTED:")
    print(f"  1. results/table_regime_characterization_{ticker}.csv")
    print(f"  2. results/table_multiseed_{ticker}.csv")
    print(f"  3. results/table_regime_breakdown_{ticker}.csv")
    print(f"  4. results/table_diebold_mariano_{ticker}.csv")
    print(f"  5. results/table_walk_forward_{ticker}.csv")
    print(f"  6. results/table_backtesting_{ticker}.csv")
    print("==================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Research Evaluation (Statistical, Walk-Forward, Backtest)")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--feature_level", type=str, default="level2_returns")
    parser.add_argument("--timestep", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024, 777, 999])

    args = parser.parse_args()
    run_comprehensive_evaluation(
        ticker=args.ticker,
        feature_level=args.feature_level,
        time_step=args.timestep,
        epochs=args.epochs,
        seeds=args.seeds
    )
