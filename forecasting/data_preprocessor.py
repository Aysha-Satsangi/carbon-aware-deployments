# forecasting/data_preprocessor.py
import json, os, pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
SCALERS = Path("data/scalers")

PROCESSED.mkdir(parents=True, exist_ok=True)
SCALERS.mkdir(parents=True, exist_ok=True)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def process_zone(zone):
    """Process zone with normalization and additional features"""
    
    print(f"\n🔄 Processing {zone}...")
    
    # 1. Load carbon history JSON
    carbon_file = next(RAW.glob(f"{zone}_past_*h.json"), None)
    if not carbon_file:
        print(f"❌ No carbon file for {zone}")
        return
    
    carbon_json = load_json(carbon_file)
    carbon_records = carbon_json.get("history", [])
    
    # Build DataFrame for carbon
    carbon_df = pd.DataFrame(carbon_records)
    if "datetime" not in carbon_df.columns or "carbonIntensity" not in carbon_df.columns:
        print(f"❌ Carbon JSON format unexpected for {zone}")
        return
    
    carbon_df = carbon_df.rename(columns={"carbonIntensity": "carbon"})
    carbon_df["datetime"] = pd.to_datetime(carbon_df["datetime"])
    carbon_df = carbon_df.set_index("datetime").sort_index()
    
    # 2. Load power breakdown JSON (if exists)
    power_file = next(RAW.glob(f"{zone}_power_*d.json"), None)

    
    # if power_file:
    #     print(f"   Found power file: {power_file.name}")
    #     power_json = load_json(power_file)
    #     power_records = power_json.get("data", [])
        
    #     if power_records:
    #         # power_df = pd.DataFrame(power_records)
    #         print(f"   Power records found: {len(power_records)}")
            
    #         if "datetime" in power_df.columns:
    #             # Extract renewable and fossil-free percentages if available
    #             for rec in power_records:
    #                 if "fossilFreePercentage" in rec:
    #                     power_df["fossilFree"] = [r.get("fossilFreePercentage", np.nan) for r in power_records]
    #                 if "renewablePercentage" in rec:
    #                     power_df["renewable"] = [r.get("renewablePercentage", np.nan) for r in power_records]
                
    #             power_df["datetime"] = pd.to_datetime(power_df["datetime"])
    #             power_df = power_df.set_index("datetime").sort_index()
                
    #             # Join with carbon data
    #             df = carbon_df.join(power_df[["fossilFree", "renewable"]], how="left")
    #         else:
    #             df = carbon_df.copy()
    #     else:
    #         df = carbon_df.copy()
    # else:
    #     df = carbon_df.copy()

    if power_file:
        print(f"   Found power file: {power_file.name}")
        power_json = load_json(power_file)
        power_records = power_json.get("data", [])
        
        if power_records:
            print(f"   Power records found: {len(power_records)}")
            
            # Extract percentages directly from records
            power_data = []
            for rec in power_records:
                power_data.append({
                    'datetime': rec.get('datetime'),
                    'fossilFree': rec.get('fossilFreePercentage'),
                    'renewable': rec.get('renewablePercentage')
                })
            
            power_df = pd.DataFrame(power_data)
            power_df["datetime"] = pd.to_datetime(power_df["datetime"])
            power_df = power_df.set_index("datetime").sort_index()
            
            print(f"   Fossil-free available: {power_df['fossilFree'].notna().sum()} records")
            print(f"   Renewable available: {power_df['renewable'].notna().sum()} records")
            
            # Join with carbon data
            df = carbon_df.join(power_df, how="left")
        else:
            print(f"    No power records in file")
            df = carbon_df.copy()
    else:
        print(f"   No power breakdown file found")
        df = carbon_df.copy()
    
    # 3. Feature engineering - time-based features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    
    # Add cyclical encoding for hour (captures daily patterns better)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # 4. Handle missing values
    # Forward fill for power data (use last known value)
    if "fossilFree" in df.columns:
        df["fossilFree"] = df["fossilFree"].fillna(method='ffill').fillna(0)
    if "renewable" in df.columns:
        df["renewable"] = df["renewable"].fillna(method='ffill').fillna(0)
    
    # Drop rows with missing carbon intensity
    df = df.dropna(subset=["carbon"])
    
    # 5. Normalization - Create and save scalers
    features_to_scale = ["carbon"]
    if "fossilFree" in df.columns:
        features_to_scale.append("fossilFree")
    if "renewable" in df.columns:
        features_to_scale.append("renewable")
    
    # Store original values before scaling
    df_original = df.copy()
    
    # Create scaler for each feature
    scalers = {}
    for feature in features_to_scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[feature] = scaler.fit_transform(df[[feature]])
        scalers[feature] = scaler
    
    # Save scalers for inverse transformation during inference
    with open(SCALERS / f"{zone}_scalers.pkl", 'wb') as f:
        pickle.dump(scalers, f)
    
    print(f"✅ Saved scalers for {zone}: {list(scalers.keys())}")
    
    # 6. Save processed data (both scaled and original)
    # Scaled version for training
    scaled_file = PROCESSED / f"{zone}_scaled.csv"
    df.to_csv(scaled_file)
    print(f"✅ Saved scaled: {scaled_file} ({len(df)} records)")
    
    # Original version for validation/plotting
    original_file = PROCESSED / f"{zone}_original.csv"
    df_original.to_csv(original_file)
    print(f"✅ Saved original: {original_file}")
    
    # 7. Print statistics
    print(f"📊 Statistics for {zone}:")
    print(f"   Carbon range: {df_original['carbon'].min():.1f} - {df_original['carbon'].max():.1f} gCO₂eq/kWh")
    print(f"   Mean carbon: {df_original['carbon'].mean():.1f} gCO₂eq/kWh")
    if "fossilFree" in df_original.columns:
        print(f"   Fossil-free: {df_original['fossilFree'].mean():.1f}%")
    if "renewable" in df_original.columns:
        print(f"   Renewable: {df_original['renewable'].mean():.1f}%")
    print(f"   Features: {df.shape[1]} columns")

if __name__ == "__main__":
    zones = {p.name.split("_")[0] for p in RAW.glob("*_past_*h.json")}
    print(f"🚀 Processing {len(zones)} zones with normalization and feature engineering...\n")
    
    for zone in sorted(zones):
        process_zone(zone)
    
    print("\n🎉 Preprocessing complete!")
    print(f"📁 Outputs:")
    print(f"   - Scaled CSVs: data/processed/*_scaled.csv")
    print(f"   - Original CSVs: data/processed/*_original.csv")
    print(f"   - Scalers: data/scalers/*_scalers.pkl")
