import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
SCALERS = Path("data/scalers")

PROCESSED.mkdir(parents=True, exist_ok=True)
SCALERS.mkdir(parents=True, exist_ok=True)

zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]

print("🔄 PREPROCESSING DATA WITH CLEANING\n")
print("="*70)

for zone in zones:
    print(f"\n📊 Processing {zone}...")
    
    # Load carbon data
    with open(RAW / f"{zone}_carbon_180d.json") as f:
        carbon_data = json.load(f)
    
    df = pd.DataFrame(carbon_data['data'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    
    # Extract carbon intensity
    df['carbon'] = pd.to_numeric(df['carbonIntensity'], errors='coerce')
    
    # Add time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for hour (24h cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day/Night flag
    df['is_night'] = ((df['hour'] >= 18) | (df['hour'] < 6)).astype(int)
    
    # Drop missing values
    df = df.dropna(subset=['carbon'])
    
    # ===== CLEANING STEP =====
    # Remove extreme outliers (near 0 and near max)
    # Keep only values between reasonable bounds
    
    carbon_min = df['carbon'].quantile(0.05)  # 5th percentile
    carbon_max = df['carbon'].quantile(0.95)  # 95th percentile
    
    # Filter to keep middle 90%
    df_clean = df[(df['carbon'] >= carbon_min) & (df['carbon'] <= carbon_max)].copy()
    
    removed = len(df) - len(df_clean)
    percent_removed = (removed / len(df)) * 100
    
    print(f"   🧹 Original records: {len(df)}")
    print(f"   🧹 Removed outliers: {removed} ({percent_removed:.1f}%)")
    print(f"   ✅ Clean records: {len(df_clean)}")
    print(f"   📊 Carbon range: {df_clean['carbon'].min():.3f} - {df_clean['carbon'].max():.3f}")
    
    # Select features for modeling
    features = ['carbon', 'hour_sin', 'hour_cos', 'is_weekend', 'is_night', 'day_of_week']
    df_features = df_clean[['datetime'] + features].copy()
    
    # Normalize only carbon
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_features['carbon'] = scaler.fit_transform(df_features[['carbon']])
    
    # Save scaler for later inverse transformation
    with open(SCALERS / f"{zone}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save processed data
    df_features.to_csv(PROCESSED / f"{zone}_processed.csv", index=False)
    
    print(f"   ✅ Features: {features}")
    print(f"   ✅ Saved: {PROCESSED}/{zone}_processed.csv")

print("\n" + "="*70)
print("✅ Preprocessing complete! Data cleaned and ready for training.")
