"""
Feature Engineering Module for PM2.5 Forecasting.
Transforms raw meteorological and pollutant time series into physically grounded feature representations:
- Trigonometric cyclical encodings for diurnal and annual rhythms.
- Cartesian decomposition of wind vectors (u and v components).
- Backward-looking autoregressive lags (no look-ahead target leakage).
- Backward-looking rolling multi-scale statistics and change rates.
- Domain-specific co-pollutant chemical ratios.
"""

import numpy as np
import pandas as pd
from typing import List

def add_cyclical_features(df: pd.DataFrame, time_col: str = "Timestamp") -> pd.DataFrame:
    """Adds harmonic sine/cosine encodings for hour, day of week, and day of year."""
    df = df.copy()
    dt = pd.to_datetime(df[time_col])
    
    # Hour of day (24-hour cycle)
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
    
    # Month of year (12-month cycle)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12.0)
    
    # Day of week (7-day cycle)
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)
    
    # Binary episode flags
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    # North India stubble burning / extreme smog window (mid-Oct to mid-Nov: day 288 to 325)
    doy = dt.dt.dayofyear
    df["is_stubble_season"] = ((doy >= 288) & (doy <= 325)).astype(int)
    df["is_monsoon"] = ((dt.dt.month >= 7) & (dt.dt.month <= 8)).astype(int)
    return df

def add_meteorological_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decomposes polar wind vectors into Cartesian components and calculates atmospheric ventilation.
    u = -ws * sin(theta) [East-West]
    v = -ws * cos(theta) [North-South]
    """
    df = df.copy()
    if "Wind_Speed" in df.columns and "Wind_Direction" in df.columns:
        rad = np.radians(df["Wind_Direction"])
        df["wind_u"] = -df["Wind_Speed"] * np.sin(rad)
        df["wind_v"] = -df["Wind_Speed"] * np.cos(rad)
    
    if "Wind_Speed" in df.columns and "PBLH" in df.columns:
        # Atmospheric Ventilation Index = Wind Speed * Boundary Layer Height
        df["ventilation_index"] = (df["Wind_Speed"] * df["PBLH"]) / 1000.0

    return df

def add_copollutant_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates diagnostic chemical ratios:
    - PM2.5 / PM10 ratio (>0.7 indicates combustion/secondary aerosols, <0.4 indicates coarse crustal dust)
    - Photochemical proxy (NO2 * O3)
    """
    df = df.copy()
    if "PM2.5" in df.columns and "PM10" in df.columns:
        df["pm_ratio"] = np.clip(df["PM2.5"] / (df["PM10"] + 1e-4), 0.05, 1.2)
        
    if "NO2" in df.columns and "O3" in df.columns:
        df["photochemical_index"] = (df["NO2"] * df["O3"]) / 100.0

    return df

def add_lag_features(df: pd.DataFrame, target_col: str = "PM2.5", lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
    """
    Constructs autoregressive lag values strictly from past observations.
    Avoids lookahead data leakage by shifting forward.
    """
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)
    
    # Also add 1h lags for primary co-pollutants if available
    for col in ["PM10", "NO2", "CO"]:
        if col in df.columns and col != target_col:
            df[f"{col}_lag_1h"] = df[col].shift(1)
            df[f"{col}_lag_24h"] = df[col].shift(24)

    return df

def add_rolling_statistics(df: pd.DataFrame, target_col: str = "PM2.5", windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
    """
    Computes backward-looking rolling statistics (center=False).
    - Rolling Mean: short, medium, and regulatory 24h exposure
    - Rolling Std: volatility indicator
    - Rate of Change: difference between short-term (3h) and mid-term (6h) trends
    """
    df = df.copy()
    # Note: Shift by 1 first so rolling features for time t do NOT include target at time t
    shifted_target = df[target_col].shift(1)
    
    for w in windows:
        df[f"{target_col}_roll_mean_{w}h"] = shifted_target.rolling(window=w, min_periods=max(1, w // 2), center=False).mean()
        df[f"{target_col}_roll_std_{w}h"] = shifted_target.rolling(window=w, min_periods=max(2, w // 2), center=False).std()
        df[f"{target_col}_roll_max_{w}h"] = shifted_target.rolling(window=w, min_periods=max(1, w // 2), center=False).max()
        df[f"{target_col}_roll_min_{w}h"] = shifted_target.rolling(window=w, min_periods=max(1, w // 2), center=False).min()

    # Rate of change / sudden plume arrival
    df[f"{target_col}_plume_delta"] = df[f"{target_col}_roll_mean_3h"] - df[f"{target_col}_roll_mean_6h"]
    return df

def build_feature_pipeline(df: pd.DataFrame, target_col: str = "PM2.5") -> pd.DataFrame:
    """Executes the complete physical and temporal feature engineering pipeline."""
    df = add_cyclical_features(df)
    df = add_meteorological_physics_features(df)
    df = add_copollutant_ratios(df)
    df = add_lag_features(df, target_col=target_col)
    df = add_rolling_statistics(df, target_col=target_col)
    
    # Drop initial rows containing NaNs introduced by the 24h lags
    df = df.dropna().reset_index(drop=True)
    return df

