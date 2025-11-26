"""
PREPROCESS 365-DAY DATA (VERSION 02)
====================================
- Uses carbon + power data for 8 zones
- Adds time features, lags, simple power features
- Normalizes key numeric columns
- Saves processed CSVs + scalers with _02 suffix
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
RAW        = BASE_DIR / "forecasting" / "data" / "raw_data_02"
PROCESSED  = BASE_DIR / "forecasting" / "data" / "processed_365d_02"
SCALERS    = BASE_DIR / "forecasting" / "data" / "scalers_365d_02"

PROCESSED.mkdir(parents=True, exist_ok=True)
SCALERS.mkdir(parents=True, exist_ok=True)

zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]

print("RAW:       ", RAW)
print("PROCESSED: ", PROCESSED)
print("SCALERS:   ", SCALERS)

# -------------------------------------------------------------------
# HELPER: safe numeric extraction for power columns
# -------------------------------------------------------------------
def safe_col(df, col, default=0.0):
    """
    If column exists -> convert to numeric and fill NaNs.
    If not -> return Series of 'default' with same index length.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    else:
        return pd.Series([default] * len(df), index=df.index)

# -------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------
for zone in zones:
    print("\n" + "="*70)
    print(f"Processing {zone} ...")

    carbon_path = RAW / f"{zone}_carbon_365d_02.json"
    power_path  = RAW / f"{zone}_power_365d_02.json"

    if not carbon_path.exists() or not power_path.exists():
        print(f"❌ Missing files for {zone}, skipping")
        continue

    # ----------------- LOAD JSON -----------------
    with open(carbon_path) as f:
        carbon_json = json.load(f)["data"]
    with open(power_path) as f:
        power_json = json.load(f)["data"]

    dfc = pd.DataFrame(carbon_json)
    dfp = pd.DataFrame(power_json)

    if dfc.empty or dfp.empty:
        print(f"❌ Empty data for {zone}, skipping")
        continue

    dfc["datetime"] = pd.to_datetime(dfc["datetime"])
    dfp["datetime"] = pd.to_datetime(dfp["datetime"])

    dfc = dfc.sort_values("datetime")
    dfp = dfp.sort_values("datetime")

    # ----------------- MERGE -----------------
    df = pd.merge(
        dfc,
        dfp,
        on="datetime",
        how="inner",
        suffixes=("_c", "_p")
    ).reset_index(drop=True)

    print(f"   Merged rows: {len(df)}")

    # ----------------- CARBON -----------------
    df["carbon"] = pd.to_numeric(df["carbonIntensity"], errors="coerce")
    print(f"   Carbon min/max: {df['carbon'].min():.1f} / {df['carbon'].max():.1f} gCO2/kWh")

    # ----------------- TIME FEATURES -----------------
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)

    # ----------------- POWER FEATURES (SAFE) -----------------
    df["renewable_pct"] = safe_col(df, "renewablePercentage", 0.0)
    df["wind"]          = safe_col(df, "wind", 0.0)
    df["solar"]         = safe_col(df, "solar", 0.0)
    df["coal"]          = safe_col(df, "coal", 0.0)
    df["gas"]           = safe_col(df, "gas", 0.0)

    df["fossil_pct"] = df["coal"] + df["gas"]

    print(f"   Renewable mean: {df['renewable_pct'].mean():.1f}%")

    # ----------------- LAG FEATURES -----------------
    df["carbon_lag_1"]   = df["carbon"].shift(1)
    df["carbon_lag_24"]  = df["carbon"].shift(24)
    df["renewable_lag_1"]= df["renewable_pct"].shift(1)

    # Fill NaNs from shifts
    df = df.ffill().bfill()

    # ----------------- NORMALIZATION -----------------
    scalers = {}
    cols_to_norm = [
        "carbon",
        "carbon_lag_1",
        "carbon_lag_24",
        "renewable_pct",
        "renewable_lag_1",
        "wind",
        "solar",
        "coal",
        "gas",
        "fossil_pct",
    ]

    for col in cols_to_norm:
        s = MinMaxScaler(feature_range=(0, 1))
        df[col + "_norm"] = s.fit_transform(df[[col]])
        scalers[col] = s

    print(f"   Normalized {len(cols_to_norm)} columns")

    # ----------------- FINAL FEATURE SET -----------------
    features = [
        "carbon_norm",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        "renewable_pct_norm",
        "renewable_lag_1_norm",
        "wind_norm",
        "solar_norm",
        "coal_norm",
        "gas_norm",
        "fossil_pct_norm",
        "carbon_lag_1_norm",
        "carbon_lag_24_norm",
    ]

    out = df[["datetime"] + features].copy()

    # ----------------- SAVE -----------------
    out_path = PROCESSED / f"{zone}_processed_02.csv"
    out.to_csv(out_path, index=False)

    scaler_path = SCALERS / f"{zone}_scalers_02.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scalers, f)

    print(f"   ✅ Saved processed: {out_path.name} ({len(out)} rows, {len(features)} features)")
    print(f"   ✅ Saved scalers:   {scaler_path.name}")

print("\n" + "="*70)
print("✅ PREPROCESSING 02 COMPLETE")
