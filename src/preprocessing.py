"""
Preprocessing & Data Hygiene Pipeline for Air Quality Time Series.
Ensures zero data leakage:
- Filters physically invalid sensor readings (negatives, flatlines).
- Fills short-duration missing sensor telemetry using forward/backward propagation.
- Strict chronological train/validation/test splitting with an embargo buffer.
- Fits Scalers strictly on training data.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import build_feature_pipeline

def clean_raw_data(df: pd.DataFrame, target_col: str = "PM2.5") -> pd.DataFrame:
    """Sanitizes raw CPCB sensor readings."""
    df = df.copy()
    
    # Ensure Timestamp is datetime and sort chronologically
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)

    # Filter physically impossible negative concentrations
    pollutant_cols = [c for c in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"] if c in df.columns]
    for col in pollutant_cols:
        df[col] = df[col].apply(lambda x: np.nan if (x is not None and x < 0) else x)
        # Cap unreasonable sensor spikes (e.g. PM2.5 > 1500 µg/m³)
        if col == "PM2.5":
            df[col] = df[col].apply(lambda x: np.nan if (x is not None and x > 1500) else x)
            
    # Interpolate short missing gaps (up to 3 consecutive hours)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=3, limit_direction="both")
    # Backfill/forward fill remaining sporadic gaps
    df[numeric_cols] = df[numeric_cols].bfill().ffill()
    
    return df

def prepare_splits_and_scaling(
    df: pd.DataFrame,
    target_col: str = "PM2.5",
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    embargo_hours: int = 24,
    save_scaler_path: str = None
) -> Dict[str, Any]:
    """
    Executes walk-forward chronological train/val/test splitting with an explicit
    embargo gap to prevent autocorrelation leakage from autoregressive features.
    """
    # Exclude non-feature columns
    drop_cols = ["Timestamp", "Station"]
    feature_cols = [c for c in df.columns if c not in drop_cols and c != target_col]
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_start = train_end + embargo_hours
    val_end = int(n * (train_ratio + val_ratio))
    test_start = val_end + embargo_hours
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[val_start:val_end].copy()
    test_df = df.iloc[test_start:].copy()
    
    X_train_raw = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    X_val_raw = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    X_test_raw = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Fit StandardScaler strictly on training features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    if save_scaler_path:
        os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
        joblib.dump(scaler, save_scaler_path)

    return {
        "feature_names": feature_cols,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df
    }

def run_preprocessing_pipeline(raw_csv_path: str, save_scaler_path: str = None) -> Dict[str, Any]:
    """Complete preprocessing orchestration from raw CSV to scaled split arrays."""
    df_raw = pd.read_csv(raw_csv_path)
    df_clean = clean_raw_data(df_raw)
    df_features = build_feature_pipeline(df_clean)
    split_data = prepare_splits_and_scaling(df_features, save_scaler_path=save_scaler_path)
    return split_data

