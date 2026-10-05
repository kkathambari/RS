import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

REGIME_NAMES = ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY"]


class MarketRegimeDetector:
    """
    Unsupervised Market Regime Detector using Gaussian Mixture Models (GMM).
    Identifies 4 economic market states:
      0: Bull Market (Positive drift, low-to-moderate volatility)
      1: Bear Market (Negative drift, elevated volatility)
      2: Sideways / Consolidation (Near-zero drift, low volatility)
      3: High-Volatility / Shock (Extreme realized volatility, fat tails)
    
    Guarantees:
      - Scaler and GMM are fitted STRICTLY on the training split to avoid lookahead bias.
      - Provides both discrete regime classification and continuous soft probability vectors r_t.
    """
    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            random_state=random_state,
            n_init=5
        )
        self.component_mapping = {}

    def extract_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes statistical regime indicators from price and volume."""
        features = pd.DataFrame(index=df.index)
        
        # 1. 20-day cumulative momentum
        features['momentum_20d'] = df['Close'].pct_change(20)
        
        # 2. 20-day annualized realized volatility
        log_ret = np.log(df['Close'] / df['Close'].shift(1))
        features['realized_vol_20d'] = log_ret.rolling(20).std() * np.sqrt(252)
        
        # 3. 60-day maximum drawdown
        rolling_max = df['Close'].rolling(60, min_periods=20).max()
        features['drawdown_60d'] = (df['Close'] - rolling_max) / (rolling_max + 1e-8)
        
        # 4. Volume surge ratio
        vol_ma = df['Volume'].rolling(20).mean()
        features['volume_ratio_20d'] = df['Volume'] / (vol_ma + 1e-8)
        
        return features

    def fit(self, train_df: pd.DataFrame):
        """Fits GMM strictly on the training partition."""
        feat_df = self.extract_regime_features(train_df).dropna()
        scaled = self.scaler.fit_transform(feat_df.values)
        self.gmm.fit(scaled)
        
        # Interpret clusters deterministically based on (mean return, mean volatility)
        means = self.gmm.means_
        # Feature 0: momentum, Feature 1: volatility
        volatilities = means[:, 1]
        returns = means[:, 0]
        
        high_vol_idx = int(np.argmax(volatilities))
        remaining = [i for i in range(self.n_regimes) if i != high_vol_idx]
        
        bull_idx = remaining[int(np.argmax(returns[remaining]))]
        remaining = [i for i in remaining if i != bull_idx]
        
        bear_idx = remaining[int(np.argmin(returns[remaining]))]
        sideways_idx = [i for i in remaining if i != bear_idx][0]
        
        self.component_mapping = {
            bull_idx: "BULL",
            bear_idx: "BEAR",
            sideways_idx: "SIDEWAYS",
            high_vol_idx: "HIGH_VOLATILITY"
        }
        return self

    def predict_regimes(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """
        Infers market regime for each timestep.
        Returns:
          - discrete_labels (N,): String labels (BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY)
          - soft_probabilities (N, 4): Continuous posterior probabilities P(Regime | Features)
          - feature_df (N, 4): Extracted indicators
        """
        feat_df = self.extract_regime_features(df).bfill().ffill()
        scaled = self.scaler.transform(feat_df.values)
        probs = self.gmm.predict_proba(scaled)
        raw_clusters = np.argmax(probs, axis=1)
        
        # Reorder probabilities to match [BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY]
        ordered_keys = []
        for name in REGIME_NAMES:
            for cluster_id, cname in self.component_mapping.items():
                if cname == name:
                    ordered_keys.append(cluster_id)
                    break
        
        ordered_probs = probs[:, ordered_keys]
        named_labels = np.array([self.component_mapping.get(c, "SIDEWAYS") for c in raw_clusters])
        
        return named_labels, ordered_probs, feat_df
