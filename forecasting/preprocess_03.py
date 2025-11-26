"""
PREPROCESS WITH EXTENDED FEATURES (VERSION 03)
===============================================
- 48-hour lookback instead of 24
- More rolling statistics
- Trend features
- Differencing (captures changes, not absolute values)
- Better for volatile zones like DE, IE
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

BASE_DIR   = Path(__file__).parent
RAW        = BASE_DIR /"forecasting"/ "data" / "raw_data_02"
PROCESSED  = BASE_DIR / "data" / "processed_365d_03"
SCALERS    = BASE_DIR / "data" / "scalers_365d_03"

PROCESSED.mkdir(parents=True, exist_ok=True)
SCALERS.mkdir(parents=True, exist_ok=True)

zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]

def safe_col(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    else:
        return pd.Series([default] * len(df), index=df.index)

print("="*70)
print("PREPROCESSING 03: ENHANCED FEATURES FOR 48H LOOKBACK")
print("="*70)

for zone in zones:
    print(f"\nProcessing {zone} ...")

    carbon_path = RAW / f"{zone}_carbon_365d_02.json"
    power_path  = RAW / f"{zone}_power_365d_02.json"

    if not carbon_path.exists() or not power_path.exists():
        print(f"  ❌ Missing files, skipping")
        continue

    with open(carbon_path) as f:
        carbon_json = json.load(f)["data"]
    with open(power_path) as f:
        power_json = json.load(f)["data"]

    dfc = pd.DataFrame(carbon_json)
    dfp = pd.DataFrame(power_json)

    dfc["datetime"] = pd.to_datetime(dfc["datetime"])
    dfp["datetime"] = pd.to_datetime(dfp["datetime"])

    df = pd.merge(dfc, dfp, on="datetime", how="inner", suffixes=("_c", "_p")).reset_index(drop=True)
    df = df.sort_values("datetime")

    print(f"  Merged rows: {len(df)}")

    # Carbon
    df["carbon"] = pd.to_numeric(df["carbonIntensity"], errors="coerce")

    # Time features
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)

    # Power
    df["renewable_pct"] = safe_col(df, "renewablePercentage", 0.0)
    df["wind"]          = safe_col(df, "wind", 0.0)
    df["solar"]         = safe_col(df, "solar", 0.0)
    df["coal"]          = safe_col(df, "coal", 0.0)
    df["gas"]           = safe_col(df, "gas", 0.0)
    df["fossil_pct"]    = df["coal"] + df["gas"]

    # ===== NEW: EXTENDED LAG FEATURES =====
    df["carbon_lag_1"]   = df["carbon"].shift(1)
    df["carbon_lag_6"]   = df["carbon"].shift(6)
    df["carbon_lag_12"]  = df["carbon"].shift(12)
    df["carbon_lag_24"]  = df["carbon"].shift(24)
    df["carbon_lag_48"]  = df["carbon"].shift(48)

    df["renewable_lag_1"]   = df["renewable_pct"].shift(1)
    df["renewable_lag_24"]  = df["renewable_pct"].shift(24)

    df["wind_lag_1"]    = df["wind"].shift(1)
    df["wind_lag_6"]    = df["wind"].shift(6)
    df["wind_lag_12"]   = df["wind"].shift(12)
    df["wind_lag_24"]   = df["wind"].shift(24)

    df["solar_lag_1"]   = df["solar"].shift(1)
    df["solar_lag_12"]  = df["solar"].shift(12)
    df["solar_lag_24"]  = df["solar"].shift(24)

    # ===== NEW: ROLLING STATISTICS (TREND) =====
    df["carbon_rolling_mean_6"]   = df["carbon"].rolling(6, min_periods=1).mean()
    df["carbon_rolling_mean_24"]  = df["carbon"].rolling(24, min_periods=1).mean()
    df["carbon_rolling_std_24"]   = df["carbon"].rolling(24, min_periods=1).std().fillna(0)

    df["wind_rolling_mean_12"]    = df["wind"].rolling(12, min_periods=1).mean()
    df["wind_rolling_std_12"]     = df["wind"].rolling(12, min_periods=1).std().fillna(0)

    df["renewable_rolling_mean_12"] = df["renewable_pct"].rolling(12, min_periods=1).mean()

    # ===== NEW: DIFFERENCING (Captures change rate) =====
    df["carbon_diff_1"]   = df["carbon"].diff().fillna(0)
    df["carbon_diff_24"]  = df["carbon"].diff(24).fillna(0)
    df["wind_diff_1"]     = df["wind"].diff().fillna(0)

    # ===== NEW: TREND DIRECTION =====
    df["carbon_trend_up_24"] = (df["carbon"] > df["carbon_rolling_mean_24"]).astype(int)

    df = df.ffill().bfill()

    # Normalization
    scalers = {}
    cols_to_norm = [
        "carbon", "carbon_lag_1", "carbon_lag_6", "carbon_lag_12", "carbon_lag_24", "carbon_lag_48",
        "renewable_pct", "renewable_lag_1", "renewable_lag_24",
        "wind", "wind_lag_1", "wind_lag_6", "wind_lag_12", "wind_lag_24",
        "solar", "solar_lag_1", "solar_lag_12", "solar_lag_24",
        "coal", "gas", "fossil_pct",
        "carbon_rolling_mean_6", "carbon_rolling_mean_24", "carbon_rolling_std_24",
        "wind_rolling_mean_12", "wind_rolling_std_12", "renewable_rolling_mean_12",
        "carbon_diff_1", "carbon_diff_24", "wind_diff_1",
    ]

    for col in cols_to_norm:
        s = MinMaxScaler((0, 1))
        df[col + "_norm"] = s.fit_transform(df[[col]])
        scalers[col] = s

    print(f"  Normalized {len(cols_to_norm)} features")

    # Final feature set (33 features)
    features = [
        "carbon_norm",
        "hour_sin", "hour_cos", "day_of_week", "is_weekend",
        "renewable_pct_norm", "renewable_lag_1_norm", "renewable_lag_24_norm",
        "wind_norm", "wind_lag_1_norm", "wind_lag_6_norm", "wind_lag_12_norm", "wind_lag_24_norm",
        "solar_norm", "solar_lag_1_norm", "solar_lag_12_norm", "solar_lag_24_norm",
        "coal_norm", "gas_norm", "fossil_pct_norm",
        "carbon_lag_1_norm", "carbon_lag_6_norm", "carbon_lag_12_norm", "carbon_lag_24_norm", "carbon_lag_48_norm",
        "carbon_rolling_mean_6_norm", "carbon_rolling_mean_24_norm", "carbon_rolling_std_24_norm",
        "wind_rolling_mean_12_norm", "wind_rolling_std_12_norm", "renewable_rolling_mean_12_norm",
        "carbon_diff_1_norm", "carbon_diff_24_norm", "wind_diff_1_norm",
        "carbon_trend_up_24",
    ]

    out = df[["datetime"] + features].copy()

    out_path = PROCESSED / f"{zone}_processed_03.csv"
    out.to_csv(out_path, index=False)

    scaler_path = SCALERS / f"{zone}_scalers_03.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scalers, f)

    print(f"  ✅ Saved: {len(out)} rows, {len(features)} features")

print("\n" + "="*70)
print("✅ PREPROCESSING 03 COMPLETE")
print("="*70)
