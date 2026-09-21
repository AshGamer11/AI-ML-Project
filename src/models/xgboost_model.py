"""
XGBoost Gradient Boosted Decision Tree Model for Tabular & Time-Series Lags.
Applies second-order Taylor expansion loss optimization with L1/L2 tree regularization.
"""

import os
import joblib
import numpy as np

class XGBoostAirQualityModel:
    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": -1
        }
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(**self.params)
            self.backend = "xgboost"
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                subsample=self.params["subsample"],
                random_state=self.params["random_state"]
            )
            self.backend = "sklearn_gbr"

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: list = None):
        if self.backend == "xgboost" and eval_set is not None:
            self.model.fit(
                X, y,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, None)

    def get_feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        self.model = joblib.load(filepath)
        return self

