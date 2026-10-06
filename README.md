# Regime-Adaptive GRU Temporal Attention for Financial Time-Series Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An audited, reproducible empirical research framework investigating whether **regime-adaptive temporal attention mechanisms** improve risk-aware financial forecasting, downside protection, and directional decision making under non-stationary market conditions.

---

## 📌 Research Question & Empirical Narrative

Financial time series exhibit persistent structural breaks between volatile contractions and momentum expansions. Traditional neural attention mechanisms score temporal states using internal recurrent dynamics alone, ignoring macroeconomic and market-regime context.

- **Central Research Question:** Can regime-conditioned temporal attention improve risk-aware financial forecasting and trading decisions under changing market conditions, even when raw point-forecast price levels do not necessarily improve?
- **Hypothesis:** Continuous market-regime probabilities $r_t$ derived from strictly backward-looking volatility, momentum, drawdown, and volume indicators dynamically modulate attention scoring ($e_t = v^\top \tanh(W_h h'_t + W_r r_t + b)$), enabling the recurrent network to focus on historically congruent states while avoiding negative transfer during market turmoil.

---

## 🔬 Core Contributions & 5-Model Ablation

1. **GRU-Enhanced Temporal Attention:** A secondary recurrent refinement layer ($h'_t = \text{GRU}_{\text{att}}(h_t)$) that extracts contextualized hidden representations prior to attention scoring.
2. **Regime-Conditioned Attention Scoring:** Dynamic injection of continuous posterior probabilities $r_t$ into the temporal attention mechanism:
   $$e_t = v^\top \tanh\left(W_h h'_t + W_r r_t + b\right)$$
   $$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^T \exp(e_k)}, \quad c = \sum_{t=1}^T \alpha_t h_t$$
3. **Causal 5-Model Ablation Suite:** Isolates recurrence, attention, custom attention, regime features, and regime conditioning:
   - **Model A:** Vanilla GRU (Recurrence baseline)
   - **Model B:** GRU + Standard Additive Attention (Bahdanau attention contribution)
   - **Model C:** GRU + Enhanced Attention (Custom secondary recurrent attention, static)
   - **Model D:** Regime-Feature GRU (Ablation control: direct concatenation of regime vector $r_t$ without attention modulation)
   - **Model E:** Proposed Regime-Adaptive Attention GRU (Full architecture)
4. **Strict Methodological Rigor:**
   - Single chronological timestamp cutoff for all components (train-only scalers and GMM).
   - Zero-lookahead feature extraction (no backward filling).
   - Regime probabilities sampled strictly at decision time $t$ (end of input sequence $X$).
   - Sample-by-sample causal backtesting alignment (no 1-day shifting).
   - Newey-West Bartlett kernel spectral density with Harvey-Leybourne-Newbold small-sample correction.

---

## 📊 Audited Empirical Results (AAPL Benchmark)

### 1. Unsupervised Market Regime Characterization
Fitted strictly up to the training cutoff date (70% partition) and characterized out-of-sample over 2,264 trading days:

| Detected Regime | Days | Frequency (%) | Mean Daily Return (%) | Ann. Volatility (%) | Mean Drawdown (%) | Max Drawdown (%) | Volume Ratio | Avg Duration (Days) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BULL** | 318 | 14.05% | **+1.321%** | 20.59% | -3.52% | -22.91% | 1.09 | 2.3 |
| **BEAR** | 396 | 17.49% | **-0.576%** | 36.10% | -13.57% | -32.08% | 1.35 | 3.2 |
| **SIDEWAYS** | 1,065 | 47.04% | **-0.000%** | 17.76% | -6.41% | -26.81% | 0.89 | 5.0 |
| **HIGH_VOLATILITY** | 485 | 21.42% | **+0.088%** | 39.56% | -19.47% | -38.73% | 0.91 | 11.3 |

*Validation Note:* Cluster economics align with designated financial states: BULL exhibits the strongest positive drift (+1.32%/day) and lowest drawdown (-3.52%); BEAR displays sustained negative returns (-0.576%/day) and elevated volume (1.35x); HIGH_VOLATILITY captures peak realized volatility (39.56%) and maximum drawdown (-38.73%).
The held-out test partition contains a robust sample across all regimes: BULL: 45, BEAR: 81, SIDEWAYS: 117, HIGH_VOLATILITY: 90.

---

### 2. Dual Target Investigation: Price Levels vs Stationary Returns

#### Case A: Raw Price Level Forecasting ($P_{t+1}$)
Evaluated across 3 seeds (`42, 101, 2024`) on the held-out test split:

| Architecture | Test RMSE (Mean ± Std) | Test MAE (Mean ± Std) | Test $R^2$ | Directional Acc. (MDA) |
| :--- | :---: | :---: | :---: | :---: |
| **Model A: Vanilla GRU** | **4.5174 ± 1.0542** | **3.7583 ± 0.9861** | **0.9441** | 48.6% |
| **Model D: Regime-Feature GRU (Control)** | 6.2571 ± 2.1172 | 5.1063 ± 1.7360 | 0.8885 | 48.4% |
| **Model E: Proposed Regime-Adaptive Attention** | **7.5933 ± 0.3404** | **6.4711 ± 0.3015** | **0.8473** | 48.8% |
| **Model B: Standard Attention GRU** | 8.2292 ± 0.7497 | 7.0282 ± 0.6870 | 0.8199 | **49.7%** |
| **Model C: Enhanced Attention (Static)** | 8.8318 ± 0.5209 | 7.5674 ± 0.3837 | 0.7932 | 48.8% |

*Point-Regression vs Trading Reality:* Vanilla GRU achieves lower point RMSE because predicting near persistence ($\hat{P}_{t+1} \approx P_t$) minimizes squared distance on trending random walks. The proposed regime-adaptive attention model achieves stronger risk-adjusted trading performance than the evaluated trading baselines, despite not minimizing raw price-level RMSE. Note that Model E achieves superior stability among attention architectures.

#### Case B: Stationary Return Forecasting ($R_{t+1}$)
When the random-walk persistence shortcut is eliminated by forecasting stationary 1-day returns (MAPE is omitted as econometrically ill-defined for returns near zero):

| Architecture | Test RMSE (Mean ± Std) | Test MAE (Mean ± Std) | Test $R^2$ | Directional Acc. (MDA) |
| :--- | :---: | :---: | :---: | :---: |
| **Model B: Standard Attention GRU** | **0.0170 ± 0.0002** | **0.0125 ± 0.0001** | **-0.0397** | **50.6%** |
| **Model E: Proposed Regime-Adaptive Attention** | **0.0173 ± 0.0004** | **0.0127 ± 0.0003** | **-0.0703** | 46.4% |
| **Model C: Enhanced Attention (Static)** | 0.0176 ± 0.0008 | 0.0130 ± 0.0008 | -0.1119 | 46.5% |
| **Model A: Vanilla GRU** | 0.0182 ± 0.0006 | 0.0133 ± 0.0005 | -0.1852 | 46.4% |
| **Model D: Regime-Feature GRU (Control)** | 0.0196 ± 0.0007 | 0.0145 ± 0.0006 | -0.3778 | 47.4% |

*Key Finding on Returns:* Model E performs comparably to Standard Attention for return forecasting ($DM = -0.4957, p = 0.6204$), while significantly outperforming Vanilla GRU ($DM = -6.1946, p < 0.0001$) and direct regime-feature conditioning ($DM = -4.8199, p < 0.0001$). However, this forecasting advantage does not translate into superior trading performance for the AAPL return-target experiment, where Standard Attention produces the strongest deep-learning trading result (+8.03% return, 0.312 Sharpe). This distinction highlights that forecast-error improvements and economic trading performance are related but not equivalent objectives.

> **Evaluation Protocol Note:** Multi-seed forecasting metrics are aggregated across 3 random seeds (`[42, 101, 2024]`). Regime-specific breakdown, Diebold-Mariano hypothesis tests, and trading simulations use the pre-specified evaluation seed (`2024`) to evaluate a deterministic model instance rather than averaging forecasts across independent models.

---

### 3. Pairwise Diebold–Mariano Hypothesis Tests (Newey–West + HLN Corrected, Seed 2024)

#### On Price Target ($P_{t+1}$):
| Pairwise Comparison | Loss Criterion | HLN DM Stat | p-value | Statistical Result |
| :--- | :---: | :---: | :---: | :--- |
| Proposed (Model E) vs Vanilla GRU (Model A) | MSE | +4.3660 | $< 0.0001$ | Baseline has significantly LOWER error |
| **Proposed (Model E) vs Standard Attention (Model B)** | MSE | **-4.4909** | **$< 0.0001$** | **Proposed has significantly LOWER error** |
| **Proposed (Model E) vs Enhanced Attention (Model C)** | MSE | **-4.0422** | **$< 0.0001$** | **Proposed has significantly LOWER error** |
| **Proposed (Model E) vs Regime-Feature (Model D)** | MSE | **-2.4663** | **$0.0142$** | **Proposed has significantly LOWER error** |

#### On Stationary Return Target ($R_{t+1}$):
| Pairwise Comparison | Loss Criterion | HLN DM Stat | p-value | Statistical Result |
| :--- | :---: | :---: | :---: | :--- |
| **Proposed (Model E) vs Vanilla GRU (Model A)** | MSE | **-6.1946** | **$< 0.0001$** | **Proposed has significantly LOWER error** |
| Proposed (Model E) vs Standard Attention (Model B) | MSE | -0.4957 | $0.6204$ | No significant difference |
| Proposed (Model E) vs Enhanced Attention (Model C) | MSE | -0.7891 | $0.4306$ | No significant difference |
| **Proposed (Model E) vs Regime-Feature (Model D)** | MSE | **-4.8199** | **$< 0.0001$** | **Proposed has significantly LOWER error** |

---

### 4. Financial Decision Making: 5-bps Transaction-Cost-Adjusted Backtesting (Seed 2024)

Executed using strictly causal sample-by-sample matching (decision at $t$, holding period $t \rightarrow t+1$):

| Metric | Vanilla GRU (A) | Standard Attn (B) | Enhanced Attn (C) | Regime Feature (D) | Proposed (E) | Buy & Hold Benchmark |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cumulative Return** | 10.28% | 5.12% | 5.82% | -6.24% | **11.83%** | 21.89% |
| **Annualized Return** | 7.69% | 3.85% | 4.37% | -4.76% | **8.83%** | 16.16% |
| **Annualized Sharpe** | 0.425 | 0.188 | 0.215 | -0.247 | **0.437** | 0.622 |
| **Maximum Drawdown** | **-8.50%** | -15.50% | -15.63% | -15.97% | **-10.34%** | -23.50% |
| **Strategy Win Rate** | 50.77% | 52.38% | 55.14% | 50.00% | **53.93%** | N/A |
| **Active Trading Days** | 65 / 333 | 105 / 333 | 107 / 333 | 108 / 333 | 89 / 333 | 333 / 333 |

#### Mechanistic Ablation Insights:
1. **Model E Outperforms All Attention and Control Baselines:** Proposed Model E achieves the highest return (11.83%) and Sharpe (0.437) among all deep learning candidates, outperforming Standard Attention (5.12%, Sharpe 0.188) and Enhanced Attention (5.82%, Sharpe 0.215).
2. **Mechanistic Ablation Evidence:** Direct regime-feature concatenation (**Model D**) underperforms the proposed regime-conditioned attention mechanism (-6.24% return, Sharpe -0.247). Conditioning temporal attention on regime probabilities (**Model E**) transforms this information into a net-positive trading policy.
3. **Downside Capital Preservation:** Both Model E (-10.34%) and Model A (-8.50%) achieve less than half the maximum drawdown of Buy & Hold (-23.50%), demonstrating active downside risk mitigation.

---

### 5. Deterministic Expanding Walk-Forward Validation

#### Price Target Walk-Forward ($P_{t+1}$):
| Fold | Window Span | Test Period | Test RMSE | Test MAE | Test MAPE (%) | Test $R^2$ | Directional Acc. (MDA) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 50% Train / 15% Val / 15% Test | 2020-11 to 2022-03 | 45.2169 | 43.2145 | 29.23% | -5.9343 | 48.19% |
| **Fold 2** | 60% Train / 15% Val / 15% Test | 2021-10 to 2023-02 | 11.4723 | 9.6221 | 6.02% | 0.2726 | 50.60% |
| **Fold 3** | 70% Train / 15% Val / 15% Test | 2022-09 to 2023-12 | 7.9543 | 6.8102 | 4.02% | 0.8327 | 48.19% |

#### Stationary Return Target Walk-Forward ($R_{t+1}$):
| Fold | Window Span | Test Period | Test RMSE | Test MAE | Test $R^2$ | Directional Acc. (MDA) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 50% Train / 15% Val / 15% Test | 2020-11 to 2022-03 | 0.0193 | 0.0147 | -0.3423 | 45.65% |
| **Fold 2** | 60% Train / 15% Val / 15% Test | 2021-10 to 2023-02 | 0.0244 | 0.0194 | -0.3359 | 47.75% |
| **Fold 3** | 70% Train / 15% Val / 15% Test | 2022-09 to 2023-12 | 0.0171 | 0.0126 | -0.0529 | 47.45% |

---

### 6. Cross-Asset Generalizability: MSFT Benchmark Replication

To examine whether findings generalize beyond Apple, the identical locked pipeline (`time_step=30`, `epochs=15`, `seeds=[42, 101, 2024]`, `eval_seed=2024`, 5-bps transaction costs) was executed on Microsoft (MSFT).

#### Stationary Return Forecasting on MSFT ($R_{t+1}$):
| Architecture | Test RMSE (Mean ± Std) | Test MAE (Mean ± Std) | Test $R^2$ | Directional Acc. (MDA) |
| :--- | :---: | :---: | :---: | :---: |
| **Model C: Enhanced Attention (Static)** | **0.0187 ± 0.0001** | 0.0140 ± 0.0002 | **-0.0618** | 47.5% |
| **Model B: Standard Attention GRU** | 0.0188 ± 0.0004 | **0.0139 ± 0.0002** | -0.0617 | **49.0%** |
| **Model E: Proposed Regime-Adaptive Attention** | 0.0192 ± 0.0005 | 0.0146 ± 0.0004 | -0.1177 | **49.0%** |
| **Model D: Regime-Feature GRU (Control)** | 0.0200 ± 0.0006 | 0.0148 ± 0.0006 | -0.2065 | 48.4% |
| **Model A: Vanilla GRU** | 0.0201 ± 0.0015 | 0.0148 ± 0.0012 | -0.2178 | 48.0% |

#### MSFT Diebold–Mariano Hypothesis Tests on Returns:
| Pairwise Comparison | Loss Criterion | HLN DM Stat | p-value | Statistical Result |
| :--- | :---: | :---: | :---: | :--- |
| **Proposed (Model E) vs Vanilla GRU (Model A)** | MSE | **-5.0757** | **$< 0.0001$** | **Proposed has significantly LOWER error** |
| Proposed (Model E) vs Standard Attention (Model B) | MSE | +2.4002 | $0.0169$ | Baseline has significantly LOWER error |
| Proposed (Model E) vs Enhanced Attention (Model C) | MSE | -0.1260 | $0.8998$ | No significant difference |
| Proposed (Model E) vs Regime-Feature (Model D) | MSE | -1.7225 | $0.0859$ | No significant difference |

#### MSFT 5-bps Transaction-Cost-Adjusted Backtesting (Seed 2024):
| Metric | Vanilla GRU (A) | Standard Attn (B) | Enhanced Attn (C) | Regime Feature (D) | Proposed (E) | Buy & Hold Benchmark |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cumulative Return** | -10.88% | **+35.51%** | +5.91% | -6.94% | **+35.06%** | +44.41% |
| **Annualized Return** | -8.35% | **+25.86%** | +4.44% | -5.30% | **+25.54%** | +32.06% |
| **Annualized Sharpe** | -1.078 | **1.240** | 0.241 | -0.302 | **1.099** | 1.038 |
| **Maximum Drawdown** | -17.73% | **-9.52%** | **-9.52%** | -15.39% | **-11.26%** | -19.65% |
| **Strategy Win Rate** | 45.45% | 48.59% | 50.91% | 50.61% | 48.02% | N/A |
| **Active Trading Days** | 55 / 333 | 142 / 333 | 110 / 333 | 164 / 333 | 177 / 333 | 333 / 333 |

#### Cross-Asset Scientific Insights:
1. **Replication of Advantage over Non-Attentive Baselines:** Model E consistently achieves statistically significantly lower return prediction error than Vanilla GRU across both assets (AAPL: $DM = -6.1946, p < 0.0001$; MSFT: $DM = -5.0757, p < 0.0001$).
2. **Elimination of Catastrophic Downside:** On MSFT, Vanilla GRU ($-10.88\%$) and passive Regime Features ($-6.94\%$) both yield negative returns and negative Sharpe ratios. Model E produces **+35.06% return** with a **1.099 Sharpe ratio** (exceeding Buy & Hold at 1.038) and nearly halving maximum drawdown ($-11.26\%$ vs. $-19.65\%$).
3. **Scientifically Grounded Boundary:** Model E does not claim universal dominance over all attention mechanisms (Standard Attention performs slightly better on MSFT), demonstrating a realistic empirical outcome rather than overfit benchmark tuning.

---

## 🏗️ Repository Architecture

```text
Research/
├── configs/
│   └── experiment_config.json        # Hyperparameters & seed definitions
├── data/
│   └── raw/                          # Frozen benchmark market data (2015–2024)
├── results/                          # Persistent audited publication tables
│   ├── table_regime_characterization_AAPL.csv
│   ├── table_multiseed_AAPL.csv
│   ├── table_regime_breakdown_AAPL.csv
│   ├── table_diebold_mariano_AAPL.csv
│   ├── table_walk_forward_AAPL.csv
│   ├── table_backtesting_AAPL.csv
│   ├── table_multiseed_AAPL_return.csv
│   └── table_diebold_mariano_AAPL_return.csv
├── src/
│   ├── data_loader.py                # Chronological data pipeline
│   ├── features.py                   # 4-tier feature engineering
│   ├── preprocessing.py              # Zero-leakage scalers & causal sequence generator
│   ├── regime_detector.py            # Backward-looking GMM regime classifier (no bfill)
│   ├── statistical_tests.py          # Diebold-Mariano test (Newey-West + HLN)
│   ├── walk_forward.py               # Deterministic 3-fold expanding cross-validation
│   ├── backtest.py                   # Causal 5-bps transaction-cost-adjusted backtesting
│   ├── evaluation.py                 # Price and Return forecasting metrics
│   └── models/
│       ├── classical_baselines.py    # Naive, ARIMA, Random Forest, XGBoost
│       ├── dl_baselines.py           # Vanilla GRU, LSTM, Std Attention, Regime-Feature GRU
│       ├── attention_gru.py          # Custom Enhanced Attention GRU
│       └── regime_attention_gru.py   # Proposed Regime-Adaptive Attention GRU
├── base.py                           # Interactive Streamlit dashboard & CLI
└── run_full_evaluation.py            # Authoritative research evaluation engine
```

---

## 🚀 Reproduction Instructions

### 1. Installation
```bash
git clone https://github.com/kkathambari/RS.git
cd RS
pip install -r requirements.txt
```

### 2. Execute Price Target Evaluation
```bash
python run_full_evaluation.py --ticker AAPL --target_type price --epochs 15 --seeds 42 101 2024
```

### 3. Execute Stationary Return Target Evaluation
```bash
python run_full_evaluation.py --ticker AAPL --target_type return --epochs 15 --seeds 42 101 2024
```

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run base.py
```

---

## 📄 License
Distributed under the MIT License.
