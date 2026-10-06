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
                                 target_type: str = "price",
                                 feature_level: str = "level2_returns",
                                 time_step: int = 30,
                                 epochs: int = 15,
                                 seeds: list = [42, 101, 2024],
                                 results_dir: str = "results"):
    target_col = "Close" if target_type.lower() == "price" else "Return"
    is_return = (target_type.lower() == "return")

    print(f"\n==================================================================")
    print(f"AUTHORITATIVE RESEARCH EVALUATION ENGINE: {ticker}")
    print(f"Target: {target_col} ({target_type.upper()}) | Features: {feature_level} | Time Step: {time_step}")
    print(f"Epochs: {epochs} | Seeds ({len(seeds)}): {seeds}")
    print(f"==================================================================")

    # 1. Load Data & Extract Features
    raw_df = load_frozen_dataset(ticker)
    feat_df, feature_cols = engineer_financial_features(raw_df, level=feature_level)
    target_idx = feature_cols.index(target_col)
    num_features = len(feature_cols)

    # 2. Single Chronological Timestamp Cutoff for ALL Components
    train_end_idx = int(len(feat_df) * 0.70)
    train_cutoff_date = feat_df.index[train_end_idx - 1]

    # Fit GMM Regime Detector strictly up to the exact train_cutoff_date
    reg_det = MarketRegimeDetector(n_regimes=4, random_state=42)
    reg_det.fit(raw_df.loc[:train_cutoff_date])
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
        target_col=target_col, time_step=time_step,
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
    # Strictly sample regime probabilities at decision time t (end of lookback window X)
    X_train_reg = np.array([aligned_probs[f_map[d]] for d in datasets['dates_train_decision']], dtype=np.float32)
    X_val_reg = np.array([aligned_probs[f_map[d]] for d in datasets['dates_val_decision']], dtype=np.float32)
    X_test_reg = np.array([aligned_probs[f_map[d]] for d in datasets['dates_test_decision']], dtype=np.float32)
    test_regimes = np.array([aligned_labels[f_map[d]] for d in datasets['dates_test_decision']])

    y_test_act = inverse_transform_target(scaler, y_test, target_idx, num_features)

    # Also track actual prices for financial backtesting if predicting returns
    if is_return:
        close_idx = feature_cols.index('Close')
        actual_test_prices = inverse_transform_target(scaler, X_test_seq[:, -1, close_idx], close_idx, num_features)
    else:
        actual_test_prices = y_test_act

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

    eval_seed = 2024 if 2024 in seeds else seeds[-1]
    seed_metrics = {k: [] for k in model_keys}
    eval_preds = {k: None for k in model_keys}

    for s_idx, s in enumerate(seeds):
        print(f"\n>> Running Evaluation Seed {s} ({s_idx + 1}/{len(seeds)})...")

        # Model A: Vanilla GRU
        set_seed(s)
        m_a = build_vanilla_gru(time_step, num_features)
        m_a, _, _ = train_dl_baseline(m_a, X_train_seq, y_train, X_val_seq, y_val, epochs=epochs, ckpt_name=f"v_gru_s{s}")
        p_a = inverse_transform_target(scaler, m_a.predict(X_test_seq, verbose=0), target_idx, num_features)
        seed_metrics["Model A: Vanilla GRU"].append(compute_forecasting_metrics(y_test_act, p_a, is_return=is_return))
        if s == eval_seed:
            eval_preds["Model A: Vanilla GRU"] = p_a

        # Model B: Standard Attention GRU
        set_seed(s)
        m_b = build_standard_attention_gru(time_step, num_features)
        m_b, _, _ = train_dl_baseline(m_b, X_train_seq, y_train, X_val_seq, y_val, epochs=epochs, ckpt_name=f"std_att_s{s}")
        p_b = inverse_transform_target(scaler, m_b.predict(X_test_seq, verbose=0), target_idx, num_features)
        seed_metrics["Model B: Standard Attention GRU"].append(compute_forecasting_metrics(y_test_act, p_b, is_return=is_return))
        if s == eval_seed:
            eval_preds["Model B: Standard Attention GRU"] = p_b

        # Model C: GRU + Enhanced Attention (Static)
        set_seed(s)
        m_c, _, _ = train_gru_model(X_train_seq, y_train, X_val_seq, y_val, use_attention=True,
                                    time_step=time_step, epochs=epochs, ckpt_name=f"enh_att_s{s}")
        p_c = inverse_transform_target(scaler, m_c.predict(X_test_seq, verbose=0), target_idx, num_features)
        seed_metrics["Model C: GRU + Enhanced Attention (Static)"].append(compute_forecasting_metrics(y_test_act, p_c, is_return=is_return))
        if s == eval_seed:
            eval_preds["Model C: GRU + Enhanced Attention (Static)"] = p_c

        # Model D: Regime-Feature GRU (Control: direct features, no attention)
        set_seed(s)
        m_d = build_regime_feature_gru(time_step, num_features, num_regimes=4)
        m_d, _, _ = train_dl_baseline(m_d, [X_train_seq, X_train_reg], y_train,
                                      [X_val_seq, X_val_reg], y_val,
                                      epochs=epochs, ckpt_name=f"reg_feat_s{s}")
        p_d = inverse_transform_target(scaler, m_d.predict([X_test_seq, X_test_reg], verbose=0), target_idx, num_features)
        seed_metrics["Model D: Regime-Feature GRU (No Attention)"].append(compute_forecasting_metrics(y_test_act, p_d, is_return=is_return))
        if s == eval_seed:
            eval_preds["Model D: Regime-Feature GRU (No Attention)"] = p_d

        # Model E: Proposed Regime-Adaptive Attention GRU
        set_seed(s)
        m_e, _, _ = train_regime_adaptive_model(
            X_train_seq, X_train_reg, y_train,
            X_val_seq, X_val_reg, y_val,
            time_step=time_step, epochs=epochs, ckpt_name=f"reg_adapt_s{s}"
        )
        p_e = inverse_transform_target(scaler, m_e.predict([X_test_seq, X_test_reg], verbose=0), target_idx, num_features)
        seed_metrics["Model E: Proposed Regime-Adaptive Attention GRU"].append(compute_forecasting_metrics(y_test_act, p_e, is_return=is_return))
        if s == eval_seed:
            eval_preds["Model E: Proposed Regime-Adaptive Attention GRU"] = p_e

    # Compile Multi-Seed Symmetrical Summary
    multi_seed_rows = []
    for k in model_keys:
        agg = aggregate_seed_metrics(seed_metrics[k])
        row = {
            "Architecture": k,
            "Test RMSE (Mean +/- Std)": agg['rmse_mean_std'],
            "Test MAE (Mean +/- Std)": agg['mae_mean_std']
        }
        if not is_return and 'mape_mean' in agg:
            row["Test MAPE (%)"] = f"{agg['mape_mean']:.2f}%"
        row["Test R2"] = f"{agg['r2_mean']:.4f}"
        row["MDA (%)"] = f"{agg['directional_acc_pct_mean']:.1f}%"
        multi_seed_rows.append(row)
    multi_seed_df = pd.DataFrame(multi_seed_rows)
    print("\n--- MULTI-SEED SUMMARY TABLE ---")
    print(multi_seed_df.to_string(index=False))

    print(f"\n>> Note: Multi-seed forecasting metrics are aggregated across {len(seeds)} random seeds ({seeds}).")
    print(f">> Regime-specific breakdown, Diebold-Mariano hypothesis tests, and trading simulations use the pre-specified evaluation seed ({eval_seed}) to evaluate a deterministic model instance rather than averaging forecasts across independent models.")

    # -------------------------------------------------------------
    # SECTION 2: REGIME-SPECIFIC BREAKDOWN TABLE
    # -------------------------------------------------------------
    print(f"\n--- SECTION 2: TEST PERFORMANCE BY DETECTED REGIME (RMSE, SEED {eval_seed}) ---")
    regime_breakdown_rows = []
    for r_name in REGIME_NAMES:
        mask = (test_regimes == r_name)
        n_samples = int(np.sum(mask))
        if n_samples == 0:
            continue
        row = {"Regime": r_name, "Test Samples": n_samples}
        for k in model_keys:
            pred = eval_preds[k]
            r_rmse = float(np.sqrt(np.mean((y_test_act[mask] - pred[mask]) ** 2)))
            short_name = k.split(":")[0].strip()
            row[f"{short_name} RMSE"] = round(r_rmse, 3)
        regime_breakdown_rows.append(row)
    regime_breakdown_df = pd.DataFrame(regime_breakdown_rows)
    print(regime_breakdown_df.to_string(index=False))
    n_high_vol = int(np.sum(test_regimes == "HIGH_VOLATILITY"))
    if n_high_vol < 10:
        print(f">> Note: High-Volatility state contains only n={n_high_vol} test samples; findings for this state are preliminary and descriptive.")
    else:
        print(f">> Regime test partition distribution: BULL={int(np.sum(test_regimes=='BULL'))}, BEAR={int(np.sum(test_regimes=='BEAR'))}, SIDEWAYS={int(np.sum(test_regimes=='SIDEWAYS'))}, HIGH_VOL={n_high_vol}.")

    # -------------------------------------------------------------
    # SECTION 3: DIEBOLD-MARIANO HYPOTHESIS TESTS (NEWEY-WEST + HLN)
    # -------------------------------------------------------------
    print(f"\n--- SECTION 3: DIEBOLD-MARIANO HYPOTHESIS TESTS (PROPOSED VS BASELINES, SEED {eval_seed}) ---")
    dm_rows = []
    pred_proposed = eval_preds["Model E: Proposed Regime-Adaptive Attention GRU"]

    comparisons = [
        ("Proposed (Model E) vs Vanilla GRU (Model A)", eval_preds["Model A: Vanilla GRU"]),
        ("Proposed (Model E) vs Standard Attention GRU (Model B)", eval_preds["Model B: Standard Attention GRU"]),
        ("Proposed (Model E) vs Enhanced Attention GRU (Model C)", eval_preds["Model C: GRU + Enhanced Attention (Static)"]),
        ("Proposed (Model E) vs Regime-Feature GRU (Model D)", eval_preds["Model D: Regime-Feature GRU (No Attention)"])
    ]

    for comp_name, baseline_pred in comparisons:
        dm_stat, p_val_raw, p_val_str, conclusion_str = diebold_mariano_test(
            y_test_act, pred_proposed, baseline_pred, h=1, criterion="MSE"
        )
        dm_rows.append({
            "Pairwise Comparison": comp_name,
            "Loss Criterion": "MSE",
            "HLN-Adjusted DM Stat": dm_stat,
            "p-value": p_val_str,
            "Statistical Result": conclusion_str
        })
    dm_df = pd.DataFrame(dm_rows)
    print(dm_df.to_string(index=False))

    # -------------------------------------------------------------
    # SECTION 4: 3-FOLD EXPANDING WALK-FORWARD VALIDATION
    # -------------------------------------------------------------
    print("\n--- SECTION 4: 3-FOLD EXPANDING WALK-FORWARD VALIDATION ---")
    wf_df, wf_agg = run_walk_forward_validation(
        feat_df, feature_cols, time_step=time_step, epochs=epochs, n_folds=3, seed=42,
        target_col=target_col, is_return=is_return
    )
    print(wf_df.to_string(index=False))
    print(f"\nWalk-Forward Summary -> Mean RMSE: {wf_agg['mean_rmse']} +/- {wf_agg['std_rmse']} | Mean R2: {wf_agg['mean_r2']} | Mean MDA: {wf_agg['mean_mda']}%")

    # -------------------------------------------------------------
    # SECTION 5: FINANCIAL BACKTESTING & RISK PROFILE (PHASE 17)
    # -------------------------------------------------------------
    print(f"\n--- SECTION 5: 5-BPS TRANSACTION-COST-ADJUSTED FINANCIAL BACKTESTING (SEED {eval_seed}) ---")
    backtester = FinancialBacktester(threshold=0.0005, transaction_cost_bps=5.0)
    
    cur_p = datasets['prices_test_decision']
    tar_p = datasets['prices_test_target']

    bt_prop = backtester.simulate(current_prices=cur_p, target_prices=tar_p,
                                  predicted_values=pred_proposed, is_return_forecast=is_return,
                                  dates_decision=datasets['dates_test_decision'],
                                  dates_target=datasets['dates_test'])['metrics']
    bt_vgru = backtester.simulate(current_prices=cur_p, target_prices=tar_p,
                                  predicted_values=eval_preds["Model A: Vanilla GRU"], is_return_forecast=is_return,
                                  dates_decision=datasets['dates_test_decision'],
                                  dates_target=datasets['dates_test'])['metrics']
    bt_std_att = backtester.simulate(current_prices=cur_p, target_prices=tar_p,
                                     predicted_values=eval_preds["Model B: Standard Attention GRU"], is_return_forecast=is_return,
                                     dates_decision=datasets['dates_test_decision'],
                                     dates_target=datasets['dates_test'])['metrics']
    bt_enh_att = backtester.simulate(current_prices=cur_p, target_prices=tar_p,
                                     predicted_values=eval_preds["Model C: GRU + Enhanced Attention (Static)"], is_return_forecast=is_return,
                                     dates_decision=datasets['dates_test_decision'],
                                     dates_target=datasets['dates_test'])['metrics']
    bt_regfeat = backtester.simulate(current_prices=cur_p, target_prices=tar_p,
                                     predicted_values=eval_preds["Model D: Regime-Feature GRU (No Attention)"], is_return_forecast=is_return,
                                     dates_decision=datasets['dates_test_decision'],
                                     dates_target=datasets['dates_test'])['metrics']

    bt_df = pd.DataFrame([
        {"Metric": "Cumulative Return", "Vanilla GRU (A)": f"{bt_vgru['Strategy Return (%)']}%", "Standard Attn (B)": f"{bt_std_att['Strategy Return (%)']}%", "Enhanced Attn (C)": f"{bt_enh_att['Strategy Return (%)']}%", "Regime Feature (D)": f"{bt_regfeat['Strategy Return (%)']}%", "Proposed (E)": f"{bt_prop['Strategy Return (%)']}%", "Buy & Hold": f"{bt_prop['Buy & Hold Return (%)']}%"},
        {"Metric": "Annualized Return", "Vanilla GRU (A)": f"{bt_vgru['Strategy Annualized (%)']}%", "Standard Attn (B)": f"{bt_std_att['Strategy Annualized (%)']}%", "Enhanced Attn (C)": f"{bt_enh_att['Strategy Annualized (%)']}%", "Regime Feature (D)": f"{bt_regfeat['Strategy Annualized (%)']}%", "Proposed (E)": f"{bt_prop['Strategy Annualized (%)']}%", "Buy & Hold": f"{bt_prop['Buy & Hold Annualized (%)']}%"},
        {"Metric": "Annualized Sharpe", "Vanilla GRU (A)": f"{bt_vgru['Strategy Sharpe']}", "Standard Attn (B)": f"{bt_std_att['Strategy Sharpe']}", "Enhanced Attn (C)": f"{bt_enh_att['Strategy Sharpe']}", "Regime Feature (D)": f"{bt_regfeat['Strategy Sharpe']}", "Proposed (E)": f"{bt_prop['Strategy Sharpe']}", "Buy & Hold": f"{bt_prop['Buy & Hold Sharpe']}"},
        {"Metric": "Maximum Drawdown", "Vanilla GRU (A)": f"{bt_vgru['Strategy Max Drawdown (%)']}%", "Standard Attn (B)": f"{bt_std_att['Strategy Max Drawdown (%)']}%", "Enhanced Attn (C)": f"{bt_enh_att['Strategy Max Drawdown (%)']}%", "Regime Feature (D)": f"{bt_regfeat['Strategy Max Drawdown (%)']}%", "Proposed (E)": f"{bt_prop['Strategy Max Drawdown (%)']}%", "Buy & Hold": f"{bt_prop['Buy & Hold Max Drawdown (%)']}%"},
        {"Metric": "Win Rate", "Vanilla GRU (A)": f"{bt_vgru['Win Rate (%)']}%", "Standard Attn (B)": f"{bt_std_att['Win Rate (%)']}%", "Enhanced Attn (C)": f"{bt_enh_att['Win Rate (%)']}%", "Regime Feature (D)": f"{bt_regfeat['Win Rate (%)']}%", "Proposed (E)": f"{bt_prop['Win Rate (%)']}%", "Buy & Hold": "N/A"},
        {"Metric": "Active Trading Days", "Vanilla GRU (A)": f"{bt_vgru['Trade Days']}/{bt_vgru['Total Days']}", "Standard Attn (B)": f"{bt_std_att['Trade Days']}/{bt_std_att['Total Days']}", "Enhanced Attn (C)": f"{bt_enh_att['Trade Days']}/{bt_enh_att['Total Days']}", "Regime Feature (D)": f"{bt_regfeat['Trade Days']}/{bt_regfeat['Total Days']}", "Proposed (E)": f"{bt_prop['Trade Days']}/{bt_prop['Total Days']}", "Buy & Hold": f"{bt_prop['Total Days']}/{bt_prop['Total Days']}"}
    ])
    print(bt_df.to_string(index=False))

    # Persist all audited research tables to results/
    prefix = f"{ticker}_{target_type}" if target_type != "price" else ticker
    os.makedirs(results_dir, exist_ok=True)
    regime_char_df.to_csv(os.path.join(results_dir, f"table_regime_characterization_{prefix}.csv"), index=False)
    multi_seed_df.to_csv(os.path.join(results_dir, f"table_multiseed_{prefix}.csv"), index=False)
    regime_breakdown_df.to_csv(os.path.join(results_dir, f"table_regime_breakdown_{prefix}.csv"), index=False)
    dm_df.to_csv(os.path.join(results_dir, f"table_diebold_mariano_{prefix}.csv"), index=False)
    wf_df.to_csv(os.path.join(results_dir, f"table_walk_forward_{prefix}.csv"), index=False)
    bt_df.to_csv(os.path.join(results_dir, f"table_backtesting_{prefix}.csv"), index=False)

    print("\n==================================================================")
    print("ALL 6 AUDITED RESEARCH PUBLICATION TABLES GENERATED & PERSISTED:")
    print(f"  1. results/table_regime_characterization_{prefix}.csv")
    print(f"  2. results/table_multiseed_{prefix}.csv")
    print(f"  3. results/table_regime_breakdown_{prefix}.csv")
    print(f"  4. results/table_diebold_mariano_{prefix}.csv")
    print(f"  5. results/table_walk_forward_{prefix}.csv")
    print(f"  6. results/table_backtesting_{prefix}.csv")
    print("==================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Research Evaluation (Statistical, Walk-Forward, Backtest)")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--target_type", type=str, default="price", choices=["price", "return"],
                        help="Target variable: 'price' (Close level) or 'return' (1-day Return)")
    parser.add_argument("--feature_level", type=str, default="level2_returns")
    parser.add_argument("--timestep", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024])

    args = parser.parse_args()
    run_comprehensive_evaluation(
        ticker=args.ticker,
        target_type=args.target_type,
        feature_level=args.feature_level,
        time_step=args.timestep,
        epochs=args.epochs,
        seeds=args.seeds
    )
