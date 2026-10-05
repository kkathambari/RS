import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, Any, Tuple


class NaiveForecaster:
    """
    Persistence / Random Walk Baseline:
    Forecasts tomorrow's price as today's observed price: \hat{P}_{t+1} = P_t.
    """
    def __init__(self):
        self.name = "Baseline_Naive_Persistence"

    def predict(self, X: np.ndarray, target_idx: int) -> np.ndarray:
        # X shape: (samples, time_step, num_features)
        # Prediction is simply the last observed target price in the lookback window
        return X[:, -1, target_idx]


class ARIMAForecaster:
    """
    Classical Autoregressive Integrated Moving Average Baseline.
    Operates on the unscaled 1D target price series.
    """
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.name = f"Baseline_ARIMA_{order[0]}_{order[1]}_{order[2]}"
        self.fitted_model = None

    def fit(self, train_series: np.ndarray):
        model = ARIMA(train_series, order=self.order)
        self.fitted_model = model.fit()
        return self

    def predict_test(self, train_series: np.ndarray, test_series: np.ndarray) -> np.ndarray:
        """
        One-step-ahead rolling forecast across the test period.
        """
        history = list(train_series)
        predictions = []
        # Fit once and do rolling 1-step forecasts
        for t in range(len(test_series)):
            model = ARIMA(history, order=self.order)
            fitted = model.fit()
            yhat = fitted.forecast()[0]
            predictions.append(yhat)
            history.append(test_series[t])
        return np.array(predictions)


class RandomForestForecaster:
    """
    Random Forest Regressor baseline on flattened temporal windows.
    """
    def __init__(self, n_estimators=100, max_depth=12, random_state=42):
        self.name = "Baseline_RandomForest"
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # Flatten (N, time_step, num_features) -> (N, time_step * num_features)
        X_flat = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_flat, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)


class XGBoostForecaster:
    """
    XGBoost Regressor baseline on flattened temporal windows.
    """
    def __init__(self, n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42):
        self.name = "Baseline_XGBoost"
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_flat = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_flat, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
