import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

RAW = Path("data/raw_data")
PROCESSED = Path("data/processed_365d")
SCALERS = Path("data/scalers_365d")

PROCESSED.mkdir(parents=True, exist_ok=True)
SCALERS.mkdir(parents=True, exist_ok=True)

zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]

print("🔄 PREPROCESSING DATA FOR TRAINING...\n")

for zone in zones:
    print(f"📊 Processing {zone}...")
    
    # Load carbon data
    with open(RAW / f"{zone}_carbon_365d.json") as f:
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
    
    # Select features for modeling
    features = ['carbon', 'hour_sin', 'hour_cos', 'is_weekend', 'is_night', 'day_of_week']
    df_features = df[['datetime'] + features].copy()
    
    # Normalize only carbon (others are already normalized)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_features['carbon'] = scaler.fit_transform(df_features[['carbon']])
    
    # Save scaler for later inverse transformation
    with open(SCALERS / f"{zone}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save processed data
    df_features.to_csv(PROCESSED / f"{zone}_processed.csv", index=False)
    
    print(f"   ✅ {len(df_features)} records, Features: {features}")
    print(f"   📊 Carbon range: {df['carbon'].min():.1f} - {df['carbon'].max():.1f}")

print("\n✅ Preprocessing complete!")
exit