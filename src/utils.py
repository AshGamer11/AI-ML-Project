import os
import random
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across numpy, python, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics and atmospheric chemistry benchmarking metrics.
    Includes R², RMSE, MAE, Index of Agreement (d), and Normalized Mean Bias (NMB).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Clean any NaNs or infinities
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {"r2": 0.0, "rmse": 0.0, "mae": 0.0, "d": 0.0, "nmb": 0.0}

    # Mean Absolute Error
    mae = float(np.mean(np.abs(y_pred - y_true)))
    
    # Root Mean Squared Error
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    
    # R-squared (Coefficient of Determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
    
    # Willmott's Index of Agreement (d)
    y_mean = np.mean(y_true)
    d_numerator = np.sum((y_true - y_pred) ** 2)
    d_denominator = np.sum((np.abs(y_pred - y_mean) + np.abs(y_true - y_mean)) ** 2)
    d = float(1 - (d_numerator / (d_denominator + 1e-8))) if d_denominator > 0 else 0.0
    
    # Normalized Mean Bias (NMB %) - Atmospheric Science Standard
    nmb = float((np.sum(y_pred - y_true) / (np.sum(y_true) + 1e-8)) * 100.0)

    return {
        "r2": round(r2, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "index_of_agreement_d": round(d, 4),
        "nmb_percent": round(nmb, 2)
    }

def categorize_aqi(pm25_val: float) -> Tuple[str, str]:
    """
    Categorizes PM2.5 (µg/m³) into CPCB National Air Quality Index (NAQI) categories and colors.
    Breakpoints (CPCB India standard):
    0-30: Good
    31-60: Satisfactory
    61-90: Moderate
    91-120: Poor
    121-250: Very Poor
    250+: Severe
    """
    if pm25_val <= 30:
        return "Good", "#00b050"
    elif pm25_val <= 60:
        return "Satisfactory", "#92d050"
    elif pm25_val <= 90:
        return "Moderate", "#ffc000"
    elif pm25_val <= 120:
        return "Poor", "#e26b00"
    elif pm25_val <= 250:
        return "Very Poor", "#c00000"
    else:
        return "Severe", "#7030a0"

def create_sequences(data: np.ndarray, target: np.ndarray, lookback: int = 24, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates temporal window sequences for PyTorch models [Samples, Lookback, Features].
    data: 2D array of features [N, num_features]
    target: 1D or 2D array of target variable (PM2.5) [N]
    lookback: historical window size (default 24 hours)
    horizon: forecasting horizon ahead (1 for t+1, 24 for t+24)
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback)])
        y.append(target[i + lookback + horizon - 1])
    return np.array(X), np.array(y)

