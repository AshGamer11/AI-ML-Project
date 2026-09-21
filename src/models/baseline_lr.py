"""
Baseline Linear Model (Ridge Regression with L2 Regularization).
Establishes the minimal predictive performance floor on tabular and lag features.
"""

import joblib
import numpy as np
from sklearn.linear_model import Ridge

class RidgeBaselineModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha, random_state=42)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")
        preds = self.model.predict(X)
        # Concentration cannot be physically negative
        return np.clip(preds, 0.0, None)

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        self.model = joblib.load(filepath)
        self.is_fitted = True
        return self

