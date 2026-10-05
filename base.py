import os
import sys
import time
import json
import argparse
from datetime import date, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_frozen_dataset, fetch_raw_ohlcv, BENCHMARK_TICKERS
from src.features import engineer_financial_features, FEATURE_LEVELS
from src.regime_detector import MarketRegimeDetector, REGIME_NAMES
from src.preprocessing import prepare_datasets, inverse_transform_target
from src.evaluation import compute_forecasting_metrics, log_experiment

# Models
from src.models.attention_gru import (
    GRU_Enhanced_Attention,
    GRU_Enhanced_Attention_Inspect,
    build_gru_model,
    train_gru_model
)
from src.models.regime_attention_gru import (
    RegimeAdaptive_GRU_Enhanced_Attention,
    RegimeAdaptive_GRU_Enhanced_Attention_Inspect,
    build_regime_adaptive_model,
    train_regime_adaptive_model
)
from src.models.dl_baselines import build_standard_attention_gru, build_vanilla_gru, train_dl_baseline


def set_seed(seed=42):
    """Enforces deterministic behavior across random, numpy, and tensorflow."""
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ==============================================================================
# CLI EXECUTION PIPELINE
# ==============================================================================

def run_headless_experiment(ticker="AAPL", start_date="2018-01-01", end_date="2024-01-01",
                            feature_level="level2_returns", time_step=30, epochs=15, seed=42,
                            model_choice="regime_adaptive", use_frozen=True):
    set_seed(seed)
    print(f"\n==================================================================")
    print(f"RUNNING RESEARCH EXPERIMENT: {ticker} (Model: {model_choice})")
    print(f"Features: {feature_level} | Time Step: {time_step} | Seed: {seed}")
    print(f"==================================================================")

    if use_frozen:
        try:
            data_df = load_frozen_dataset(ticker)
        except Exception:
            data_df = fetch_raw_ohlcv(ticker, start_date=start_date, end_date=end_date)
    else:
        data_df = fetch_raw_ohlcv(ticker, start_date=start_date, end_date=end_date)

    feat_df, feature_cols = engineer_financial_features(data_df, level=feature_level)
    target_idx = feature_cols.index('Close')
    num_features = len(feature_cols)

    # Chronological partition
    N = len(feat_df)
    train_end = int(N * 0.70)

    # Fit Regime Detector on train
    regime_detector = MarketRegimeDetector(n_regimes=4, random_state=seed)
    regime_detector.fit(data_df.iloc[:train_end])
    regime_labels_all, regime_probs_all, _ = regime_detector.predict_regimes(data_df)

    feat_dates = feat_df.index
    date_to_idx = {d: i for i, d in enumerate(data_df.index)}
    aligned_indices = [date_to_idx[d] for d in feat_dates]
    regime_probs = regime_probs_all[aligned_indices]

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

    date_to_feat_idx = {d: i for i, d in enumerate(feat_dates)}
    X_train_reg = np.array([regime_probs[date_to_feat_idx[d]] for d in datasets['dates_train']], dtype=np.float32)
    X_val_reg = np.array([regime_probs[date_to_feat_idx[d]] for d in datasets['dates_val']], dtype=np.float32)
    X_test_reg = np.array([regime_probs[date_to_feat_idx[d]] for d in datasets['dates_test']], dtype=np.float32)

    t0 = time.time()
    if model_choice == "regime_adaptive":
        model, dur, _ = train_regime_adaptive_model(
            X_train_seq, X_train_reg, y_train,
            X_val_seq, X_val_reg, y_val,
            time_step=time_step, epochs=epochs
        )
        test_pred_scaled = model.predict([X_test_seq, X_test_reg], verbose=0)
    elif model_choice == "static_attention":
        model, dur, _ = train_gru_model(
            X_train_seq, y_train, X_val_seq, y_val,
            use_attention=True, time_step=time_step, epochs=epochs
        )
        test_pred_scaled = model.predict(X_test_seq, verbose=0)
    else:
        v_gru = build_vanilla_gru(time_step, num_features)
        model, dur, _ = train_dl_baseline(v_gru, X_train_seq, y_train, X_val_seq, y_val, epochs=epochs)
        test_pred_scaled = model.predict(X_test_seq, verbose=0)

    y_test_act = inverse_transform_target(scaler, y_test, target_idx, num_features)
    y_test_pred = inverse_transform_target(scaler, test_pred_scaled, target_idx, num_features)

    test_metrics = compute_forecasting_metrics(y_test_act, y_test_pred)
    run_meta = {
        'run_id': f"{ticker}_{model_choice}_{int(time.time())}",
        'timestamp': datetime.now().isoformat(),
        'ticker': ticker,
        'feature_level': feature_level,
        'model': model_choice,
        'seed': seed,
        'time_step': time_step,
        'train_samples': len(X_train_seq),
        'val_samples': len(X_val_seq),
        'test_samples': len(X_test_seq),
        'training_seconds': round(dur, 2)
    }
    csv_path = log_experiment("results", run_meta, {}, {}, test_metrics)

    print("\n--- RESULTS SUMMARY ---")
    print(f"Test RMSE: {test_metrics['rmse']} | MAE: {test_metrics['mae']} | MAPE: {test_metrics['mape']}% | R2: {test_metrics['r2']} | MDA: {test_metrics['directional_acc_pct']}%")
    print(f"Persisted experiment log to: {csv_path}\n")

# ==============================================================================
# INTERACTIVE STREAMLIT DASHBOARD
# ==============================================================================

def render_streamlit_dashboard():
    import streamlit as st
    set_seed(42)

    st.set_page_config(page_title="Regime-Adaptive Attention Research", layout="wide")
    st.title("🔬 Regime-Adaptive Temporal Attention for Financial Forecasting")
    st.caption("Phase 8 & 9: Unsupervised Market-Regime Detection, Regime-Conditioned Temporal Attention, & Held-Out Out-of-Sample Evaluation")

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.title("1. Research Dataset")
    data_source = st.sidebar.radio("Data Source Mode", ["Frozen Benchmark (Reproducible)", "Live Yahoo Finance Fetch"])

    all_benchmarks = ["AAPL", "MSFT", "NVDA", "RELIANCE.NS", "TCS.NS", "^NSEI", "^GSPC"]
    if data_source == "Frozen Benchmark (Reproducible)":
        ticker = st.sidebar.selectbox("Select Benchmark Asset", all_benchmarks, index=0)
        start_date, end_date = "2015-01-01", "2024-01-01"
    else:
        ticker = st.sidebar.text_input("Enter Asset Ticker", "AAPL").upper()
        default_start = date(2018, 1, 1)
        today = date.today()
        start_date = st.sidebar.date_input("Start Date", default_start, max_value=today)
        end_date = st.sidebar.date_input("End Date", today, min_value=start_date, max_value=today)

    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Model Architecture")
    model_choice = st.sidebar.selectbox(
        "Select Forecasting Architecture",
        [
            "Proposed: Regime-Adaptive Attention GRU",
            "Proposed: GRU-Enhanced Attention (Static)",
            "Baseline: Standard Attention GRU (Bahdanau)",
            "Baseline: Vanilla GRU"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Feature Engineering Tier")
    feature_level = st.sidebar.selectbox(
        "Feature Level (Ablation)",
        options=list(FEATURE_LEVELS.keys()),
        index=1,
        format_func=lambda x: {
            "level1_price": "Level 1: Price & Volume (5 features)",
            "level2_returns": "Level 2: Price + Returns (9 features)",
            "level3_volatility": "Level 3: Returns + Volatility + ATR (13 features)",
            "level4_full": "Level 4: Full Suite + RSI, MACD, MAs (20 features)"
        }[x]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("4. Hyperparameters")
    time_step = st.sidebar.slider("Sequence Lookback Window", 15, 60, 30, step=5)
    epochs = st.sidebar.slider("Training Epochs", 5, 50, 15, step=5)
    random_seed = st.sidebar.number_input("Random Seed (Reproducibility)", value=42, step=1)
    demo_mode = st.sidebar.checkbox("Fast Demo Mode", value=True)

    if st.sidebar.button("📥 Load & Prepare Pipeline", key="btn_load"):
        with st.spinner(f"Preparing data, extracting features, and detecting regimes for {ticker}..."):
            try:
                if data_source == "Frozen Benchmark (Reproducible)":
                    raw_df = load_frozen_dataset(ticker)
                else:
                    raw_df = fetch_raw_ohlcv(ticker, start_date=str(start_date), end_date=str(end_date))

                feat_df, cols = engineer_financial_features(raw_df, level=feature_level)
                
                # Fit regime detector on train
                train_end = int(len(feat_df) * 0.70)
                reg_det = MarketRegimeDetector(n_regimes=4, random_state=int(random_seed))
                reg_det.fit(raw_df.iloc[:train_end])
                labels_all, probs_all, _ = reg_det.predict_regimes(raw_df)

                feat_dates = feat_df.index
                date_to_idx = {d: i for i, d in enumerate(raw_df.index)}
                aligned_indices = [date_to_idx[d] for d in feat_dates]
                regime_probs = probs_all[aligned_indices]
                regime_labels = labels_all[aligned_indices]

                st.session_state['feat_df'] = feat_df
                st.session_state['feature_cols'] = cols
                st.session_state['ticker'] = ticker
                st.session_state['feature_level'] = feature_level
                st.session_state['regime_probs'] = regime_probs
                st.session_state['regime_labels'] = regime_labels
                st.session_state.pop('trained_model', None)
                st.sidebar.success(f"Loaded {len(feat_df)} rows with {len(cols)} features & 4 regimes!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    if 'feat_df' not in st.session_state:
        st.info("👈 Select options and click **'Load & Prepare Pipeline'** to begin.")
        st.markdown("""
        ### Phase 8 & 9 Methodology Highlights
        - **Unsupervised Regime Detection (GMM):** Classifies market conditions into **Bull**, **Bear**, **Sideways**, and **High-Volatility**.
        - **Regime-Adaptive Attention:** Conditions temporal scores $\\alpha_t = \\text{softmax}(f(h_t, r_t))$ directly on the market regime vector $r_t$.
        - **Zero Future Information Leakage:** Both Scaler and GMM are trained exclusively on historical train data.
        """)
        return

    feat_df = st.session_state['feat_df']
    feature_cols = st.session_state['feature_cols']
    ticker = st.session_state['ticker']
    regime_probs = st.session_state['regime_probs']
    regime_labels = st.session_state['regime_labels']
    target_idx = feature_cols.index('Close')
    num_features = len(feature_cols)

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

    feat_dates = feat_df.index
    date_to_feat_idx = {d: i for i, d in enumerate(feat_dates)}
    X_train_reg = np.array([regime_probs[date_to_feat_idx[d]] for d in datasets['dates_train']], dtype=np.float32)
    X_val_reg = np.array([regime_probs[date_to_feat_idx[d]] for d in datasets['dates_val']], dtype=np.float32)
    X_test_reg = np.array([regime_probs[date_to_feat_idx[d]] for d in datasets['dates_test']], dtype=np.float32)
    test_regime_labels = np.array([regime_labels[date_to_feat_idx[d]] for d in datasets['dates_test']])

    tab_regimes, tab_train, tab_inspector, tab_registry = st.tabs([
        "📈 Market Regimes Timeline",
        "🚀 Train & Evaluate Architecture",
        "🔍 Regime-Adaptive Attention Inspector",
        "📁 Experiment Logs"
    ])

    # --- TAB 1: REGIME TIMELINE ---
    with tab_regimes:
        st.subheader("Unsupervised Market Regime Segmentation (GMM)")
        import collections
        dist = collections.Counter(regime_labels)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Bull Market", f"{dist['BULL']} days")
        c2.metric("🔴 Bear Market", f"{dist['BEAR']} days")
        c3.metric("🟡 Sideways", f"{dist['SIDEWAYS']} days")
        c4.metric("⚡ High-Volatility", f"{dist['HIGH_VOLATILITY']} days")

        fig_reg, ax_reg = plt.subplots(figsize=(14, 5))
        colors = {"BULL": "#2ca02c", "BEAR": "#d62728", "SIDEWAYS": "#bcbd22", "HIGH_VOLATILITY": "#9467bd"}
        for r_name, col in colors.items():
            mask = (regime_labels == r_name)
            ax_reg.scatter(feat_dates[mask], feat_df.loc[mask, 'Close'], label=r_name, color=col, s=8, alpha=0.6)
        ax_reg.plot(feat_dates, feat_df['Close'], color='black', alpha=0.25, linewidth=1)
        ax_reg.set_title(f"Unsupervised Market Regime Dynamics for {ticker}")
        ax_reg.set_ylabel("Price")
        ax_reg.legend(); ax_reg.grid(True, alpha=0.3)
        st.pyplot(fig_reg)

    # --- TAB 2: TRAINING ---
    with tab_train:
        st.subheader(f"Model Training: {model_choice}")
        if st.button("🚀 Train Selected Model", key="btn_train"):
            set_seed(int(random_seed))
            actual_epochs = 8 if demo_mode else epochs
            gru_units = 32 if demo_mode else 64
            att_units = 16 if demo_mode else 32

            with st.spinner(f"Training {model_choice} with EarlyStopping on validation split..."):
                if "Regime-Adaptive" in model_choice:
                    model, dur, history = train_regime_adaptive_model(
                        X_train_seq, X_train_reg, y_train,
                        X_val_seq, X_val_reg, y_val,
                        time_step=time_step, epochs=actual_epochs,
                        gru_units=gru_units, att_units=att_units
                    )
                    test_pred_scaled = model.predict([X_test_seq, X_test_reg], verbose=0)
                elif "Static" in model_choice:
                    model, dur, history = train_gru_model(
                        X_train_seq, y_train, X_val_seq, y_val,
                        use_attention=True, time_step=time_step, epochs=actual_epochs,
                        gru_units=gru_units, att_units=att_units
                    )
                    test_pred_scaled = model.predict(X_test_seq, verbose=0)
                elif "Standard Attention" in model_choice:
                    std_m = build_standard_attention_gru(time_step, num_features, units=gru_units, att_units=att_units)
                    model, dur, history = train_dl_baseline(std_m, X_train_seq, y_train, X_val_seq, y_val, epochs=actual_epochs)
                    test_pred_scaled = model.predict(X_test_seq, verbose=0)
                else:
                    v_m = build_vanilla_gru(time_step, num_features, units=gru_units)
                    model, dur, history = train_dl_baseline(v_m, X_train_seq, y_train, X_val_seq, y_val, epochs=actual_epochs)
                    test_pred_scaled = model.predict(X_test_seq, verbose=0)

            y_test_act = inverse_transform_target(scaler, y_test, target_idx, num_features)
            y_test_pred = inverse_transform_target(scaler, test_pred_scaled, target_idx, num_features)
            test_metrics = compute_forecasting_metrics(y_test_act, y_test_pred)

            st.session_state['eval_results'] = {
                'metrics': test_metrics,
                'y_test_act': y_test_act,
                'y_test_pred': y_test_pred,
                'dates_test': datasets['dates_test'],
                'test_regimes': test_regime_labels,
                'history': history.history
            }
            st.session_state['trained_model'] = model
            st.session_state['trained_model_name'] = model_choice
            st.success(f"Training completed in {dur:.1f}s!")

        if 'eval_results' in st.session_state:
            res = st.session_state['eval_results']
            st.markdown(f"### 🏆 Out-of-Sample Performance ({st.session_state.get('trained_model_name')})")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Test RMSE", f"{res['metrics']['rmse']:.4f}")
            m2.metric("Test MAE", f"{res['metrics']['mae']:.4f}")
            m3.metric("Test MAPE", f"{res['metrics']['mape']:.2f}%")
            m4.metric("Test R²", f"{res['metrics']['r2']:.4f}")
            m5.metric("Directional Acc.", f"{res['metrics']['directional_acc_pct']:.1f}%")

            col_a, col_b = st.columns([2, 1])
            with col_a:
                fig_res, ax_res = plt.subplots(figsize=(12, 5))
                ax_res.plot(res['dates_test'], res['y_test_act'], label="Actual Test Price", color='black', linewidth=1.5)
                ax_res.plot(res['dates_test'], res['y_test_pred'], label="Forecast", color='#d62728', linestyle='--', linewidth=1.5)
                ax_res.set_title(f"Test Set Evaluation ({ticker})")
                ax_res.set_ylabel("Price")
                ax_res.legend(); ax_res.grid(True, alpha=0.3)
                st.pyplot(fig_res)
            with col_b:
                # Regime-wise breakdown on test set
                reg_breakdown = []
                for r in ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY"]:
                    m = (res['test_regimes'] == r)
                    if np.sum(m) > 0:
                        r_rmse = float(np.sqrt(np.mean((res['y_test_act'][m] - res['y_test_pred'][m])**2)))
                        reg_breakdown.append({"Regime": r, "RMSE": round(r_rmse, 3), "Samples": int(np.sum(m))})
                st.write("**Regime-Specific Breakdown:**")
                st.dataframe(pd.DataFrame(reg_breakdown), use_container_width=True)

    # --- TAB 3: ATTENTION INSPECTOR ---
    with tab_inspector:
        st.subheader("Proof of Mechanism: Regime-Conditioned Attention Inspector")
        if 'trained_model' not in st.session_state:
            st.warning("Please train a model in Tab 2 to inspect temporal attention weights.")
        else:
            sample_idx = st.slider("Select Test Window to Inspect:", 0, len(X_test_seq) - 1, 0)
            target_regime = test_regime_labels[sample_idx]
            reg_prob_sample = X_test_reg[sample_idx]

            st.info(f"Window Target Date: `{datasets['dates_test'][sample_idx].strftime('%Y-%m-%d')}` | Detected Market Regime: **{target_regime}** (BULL: {reg_prob_sample[0]:.2f}, BEAR: {reg_prob_sample[1]:.2f}, SIDEWAYS: {reg_prob_sample[2]:.2f}, HIGH_VOL: {reg_prob_sample[3]:.2f})")

            if st.button("🔍 Extract Attention Weights", key="btn_inspect"):
                model = st.session_state['trained_model']
                sample_seq = X_test_seq[sample_idx:sample_idx+1]
                sample_reg = X_test_reg[sample_idx:sample_idx+1]

                if "Regime-Adaptive" in st.session_state.get('trained_model_name'):
                    pre_att_layer = model.get_layer("dropout_1").output
                    extractor = tf.keras.models.Model(inputs=model.input, outputs=pre_att_layer)
                    pre_att_features = extractor.predict([sample_seq, sample_reg], verbose=0)

                    inspect_layer = RegimeAdaptive_GRU_Enhanced_Attention_Inspect(att_units=16, name="inspector")
                    d_seq = tf.keras.layers.Input(shape=(pre_att_features.shape[1], pre_att_features.shape[2]))
                    d_reg = tf.keras.layers.Input(shape=(sample_reg.shape[1],))
                    out_inspect = inspect_layer([d_seq, d_reg])
                    inspector_model = tf.keras.models.Model(inputs=[d_seq, d_reg], outputs=out_inspect)

                    try:
                        inspect_layer.set_weights(model.get_layer("regime_adaptive_attention").get_weights())
                    except Exception:
                        pass
                    _, _, att_weights = inspector_model.predict([pre_att_features, sample_reg], verbose=0)
                    weights = att_weights.squeeze()
                else:
                    weights = np.ones(time_step) / time_step

                raw_sample = inverse_transform_target(scaler, sample_seq[0, :, target_idx], target_idx, num_features)
                top_idx = np.argsort(weights)[-5:]

                fig_att, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
                ax1.plot(range(time_step), raw_sample, label="Price Sequence in Lookback Horizon", color='#1f77b4', linewidth=1.8)
                ax1.scatter(top_idx, raw_sample[top_idx], color='red', s=120, zorder=5, label="Top Attended Timesteps")
                ax1.set_title(f"Regime-Conditioned Attention ({target_regime} Regime)")
                ax1.set_ylabel("Price")
                ax1.legend(); ax1.grid(True, alpha=0.3)

                ax2.bar(range(time_step), weights, color='#2ca02c', alpha=0.7, label=r"Regime-Modulated Attention $\alpha_t$")
                ax2.set_xlabel("Relative Lookback Step (t - window to t)")
                ax2.set_ylabel("Attention Weight")
                ax2.legend(); ax2.grid(True, alpha=0.3)
                st.pyplot(fig_att)

    # --- TAB 4: EXPERIMENT LOGS ---
    with tab_registry:
        st.subheader("Persistent Research Registry (results/experiments_log.csv)")
        csv_file = os.path.join("results", "experiments_log.csv")
        if os.path.exists(csv_file):
            st.dataframe(pd.read_csv(csv_file).tail(20), use_container_width=True)
            if st.button("Refresh Experiment Logs"):
                st.rerun()

# ==============================================================================
# SCRIPT ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Time-Series Forecasting Research Framework")
    parser.add_argument("--cli", action="store_true", help="Run experiment in headless CLI mode instead of Streamlit")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--feature_level", type=str, default="level2_returns", choices=list(FEATURE_LEVELS.keys()))
    parser.add_argument("--model", type=str, default="regime_adaptive", choices=["regime_adaptive", "static_attention", "vanilla_gru"])
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timestep", type=int, default=30, help="Lookback window length")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    if len(sys.argv) > 1 and "--cli" in sys.argv:
        args = parser.parse_args()
        run_headless_experiment(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            feature_level=args.feature_level,
            time_step=args.timestep,
            epochs=args.epochs,
            seed=args.seed,
            model_choice=args.model,
            use_frozen=True
        )
    else:
        render_streamlit_dashboard()
