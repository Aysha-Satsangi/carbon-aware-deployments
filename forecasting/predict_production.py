"""
ECODEPLOY PRODUCTION PREDICTION SYSTEM
=======================================
Collects real 120h data from API → Preprocesses → Predicts → Visualizes

Usage:
    python forecasting/predict_production.py --api-key YOUR_KEY
    python forecasting/predict_production.py --dummy  (for testing without API)
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import requests
import json
import warnings
warnings.filterwarnings('ignore')

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "forecasting" / "data" / "models_365d_03"
OUTPUT_DIR = BASE_DIR / "forecasting" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

LOOKBACK = 48
HORIZON = 24

ZONES = {
    "DE": "Germany",
    "US-MIDA-PJM": "US Mid-Atlantic",
    "US-NW-PACW": "US West Oregon",
    "IE": "Ireland",
    "SG": "Singapore",
    "BE": "Belgium",
    "US-MIDW-MISO": "US Midwest",
    "JP-TK": "Japan Tokyo"
}

CARBON_RANGES = {
    "DE": (50, 800),
    "US-MIDA-PJM": (100, 700),
    "US-NW-PACW": (50, 600),
    "IE": (100, 900),
    "SG": (300, 700),
    "BE": (50, 800),
    "US-MIDW-MISO": (150, 850),
    "JP-TK": (200, 700)
}

BASE_URL = "https://api.electricitymaps.com/v3"


# ============================================================================
# STEP 1: DATA COLLECTION FROM API
# ============================================================================

def fetch_real_data(zone, api_key, hours=120):
    """
    Fetch real 120h data from Electricity Maps API.
    Returns merged DataFrame with carbon + power data.
    """
    print(f"      Fetching {hours}h real data from API...")
    
    headers = {"auth-token": api_key}
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)
    
    start_iso = start_time.isoformat().replace("+00:00", "Z")
    end_iso = end_time.isoformat().replace("+00:00", "Z")
    
    try:
        # Fetch carbon intensity
        print(f"        Fetching carbon data...")
        carbon_resp = requests.get(
            f"{BASE_URL}/carbon-intensity/past-range",
            params={
                "zone": zone,
                "start": start_iso,
                "end": end_iso,
                "temporalGranularity": "hourly"
            },
            headers=headers,
            timeout=30
        )
        
        if carbon_resp.status_code != 200:
            print(f"        ✗ Carbon API error: {carbon_resp.status_code}")
            return None
        
        carbon_data = carbon_resp.json().get("data", [])
        print(f"        ✓ Carbon: {len(carbon_data)} records")
        
        # Fetch power breakdown
        print(f"        Fetching power data...")
        power_resp = requests.get(
            f"{BASE_URL}/power-breakdown/past-range",
            params={
                "zone": zone,
                "start": start_iso,
                "end": end_iso,
                "temporalGranularity": "hourly"
            },
            headers=headers,
            timeout=30
        )
        
        if power_resp.status_code != 200:
            print(f"        ✗ Power API error: {power_resp.status_code}")
            return None
        
        power_data = power_resp.json().get("data", [])
        print(f"        ✓ Power: {len(power_data)} records")
        
        # Merge by datetime
        carbon_by_dt = {r["datetime"]: r for r in carbon_data if "datetime" in r}
        power_by_dt = {r["datetime"]: r for r in power_data if "datetime" in r}
        
        merged_records = []
        for dt in sorted(carbon_by_dt.keys()):
            carbon = carbon_by_dt[dt]
            power = power_by_dt.get(dt, {})
            
            # Extract fields
            power_breakdown = power.get('powerProductionBreakdown', {})
            
            record = {
                'datetime': dt,
                'carbon_intensity': carbon.get('carbonIntensity', 0),
                'renewable_percentage': power.get('renewablePercentage', 0),
                'fossil_percentage': 100 - power.get('renewablePercentage', 0),
                'wind_power': power_breakdown.get('wind', 0),
                'solar_power': power_breakdown.get('solar', 0),
            }
            merged_records.append(record)
        
        df = pd.DataFrame(merged_records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.fillna(0)
        
        print(f"        ✓ Merged {len(df)} total records")
        return df
        
    except Exception as e:
        print(f"        ✗ API Error: {e}")
        return None


def generate_dummy_data(zone, hours=120):
    """Generate realistic dummy data for testing"""
    print(f"      Generating {hours}h dummy data...")
    
    end_time = datetime.now(timezone.utc)
    dates = [end_time - timedelta(hours=i) for i in range(hours-1, -1, -1)]
    
    # Realistic patterns
    time_array = np.linspace(0, 5*2*np.pi, hours)
    
    base_carbon = np.random.randint(250, 350)
    carbon = base_carbon + 150 * np.sin(time_array)
    carbon += np.random.normal(0, 15, hours)
    carbon = np.clip(carbon, 50, 900)
    
    renewable = 50 + 30 * np.sin(time_array + np.pi/2)
    renewable += np.random.normal(0, 5, hours)
    renewable = np.clip(renewable, 15, 85)
    
    fossil = 100 - renewable
    
    data = {
        'datetime': dates,
        'carbon_intensity': carbon,
        'fossil_percentage': fossil,
        'renewable_percentage': renewable,
        'wind_power': np.random.uniform(100, 800, hours),
        'solar_power': np.maximum(0, 600 * np.sin(time_array + np.pi/2)) + np.random.normal(0, 50, hours)
    }
    
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.fillna(0)
    
    print(f"        ✓ Generated {len(df)} dummy records")
    return df


# ============================================================================
# STEP 2: EXACT FEATURE ENGINEERING (35 FEATURES)
# ============================================================================

def engineer_features_exact_35(df):
    """
    EXACT feature engineering matching preprocess_03.py (35 features).
    Replicates training preprocessing exactly.
    """
    df = df.copy()
    
    # Rename to match training
    df = df.rename(columns={
        'carbon_intensity': 'carbon',
        'renewable_percentage': 'renewable_pct',
        'fossil_percentage': 'fossil_pct',
        'wind_power': 'wind',
        'solar_power': 'solar'
    })
    
    # Add coal and gas (derive from fossil_pct)
    df['coal'] = df['fossil_pct'] * 0.6
    df['gas'] = df['fossil_pct'] * 0.4
    
    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # LAG FEATURES - Carbon
    df['carbon_lag_1'] = df['carbon'].shift(1)
    df['carbon_lag_6'] = df['carbon'].shift(6)
    df['carbon_lag_12'] = df['carbon'].shift(12)
    df['carbon_lag_24'] = df['carbon'].shift(24)
    df['carbon_lag_48'] = df['carbon'].shift(48)
    
    # LAG FEATURES - Renewable
    df['renewable_lag_1'] = df['renewable_pct'].shift(1)
    df['renewable_lag_24'] = df['renewable_pct'].shift(24)
    
    # LAG FEATURES - Wind
    df['wind_lag_1'] = df['wind'].shift(1)
    df['wind_lag_6'] = df['wind'].shift(6)
    df['wind_lag_12'] = df['wind'].shift(12)
    df['wind_lag_24'] = df['wind'].shift(24)
    
    # LAG FEATURES - Solar
    df['solar_lag_1'] = df['solar'].shift(1)
    df['solar_lag_12'] = df['solar'].shift(12)
    df['solar_lag_24'] = df['solar'].shift(24)
    
    # ROLLING STATISTICS - Carbon
    df['carbon_rolling_mean_6'] = df['carbon'].rolling(6, min_periods=1).mean()
    df['carbon_rolling_mean_24'] = df['carbon'].rolling(24, min_periods=1).mean()
    df['carbon_rolling_std_24'] = df['carbon'].rolling(24, min_periods=1).std().fillna(0)
    
    # ROLLING STATISTICS - Wind
    df['wind_rolling_mean_12'] = df['wind'].rolling(12, min_periods=1).mean()
    df['wind_rolling_std_12'] = df['wind'].rolling(12, min_periods=1).std().fillna(0)
    
    # ROLLING STATISTICS - Renewable
    df['renewable_rolling_mean_12'] = df['renewable_pct'].rolling(12, min_periods=1).mean()
    
    # DIFFERENCING
    df['carbon_diff_1'] = df['carbon'].diff().fillna(0)
    df['carbon_diff_24'] = df['carbon'].diff(24).fillna(0)
    df['wind_diff_1'] = df['wind'].diff().fillna(0)
    
    # TREND
    df['carbon_trend_up_24'] = (df['carbon'] > df['carbon_rolling_mean_24']).astype(int)
    
    # Fill missing
    df = df.ffill().bfill()
    
    # NORMALIZATION (MinMax to [0,1])
    cols_to_norm = [
        'carbon', 'carbon_lag_1', 'carbon_lag_6', 'carbon_lag_12', 'carbon_lag_24', 'carbon_lag_48',
        'renewable_pct', 'renewable_lag_1', 'renewable_lag_24',
        'wind', 'wind_lag_1', 'wind_lag_6', 'wind_lag_12', 'wind_lag_24',
        'solar', 'solar_lag_1', 'solar_lag_12', 'solar_lag_24',
        'coal', 'gas', 'fossil_pct',
        'carbon_rolling_mean_6', 'carbon_rolling_mean_24', 'carbon_rolling_std_24',
        'wind_rolling_mean_12', 'wind_rolling_std_12', 'renewable_rolling_mean_12',
        'carbon_diff_1', 'carbon_diff_24', 'wind_diff_1',
    ]
    
    for col in cols_to_norm:
        scaler = MinMaxScaler((0, 1))
        df[col + '_norm'] = scaler.fit_transform(df[[col]])
    
    # FINAL 35 FEATURES (exact order from training)
    features = [
        'carbon_norm',
        'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend',
        'renewable_pct_norm', 'renewable_lag_1_norm', 'renewable_lag_24_norm',
        'wind_norm', 'wind_lag_1_norm', 'wind_lag_6_norm', 'wind_lag_12_norm', 'wind_lag_24_norm',
        'solar_norm', 'solar_lag_1_norm', 'solar_lag_12_norm', 'solar_lag_24_norm',
        'coal_norm', 'gas_norm', 'fossil_pct_norm',
        'carbon_lag_1_norm', 'carbon_lag_6_norm', 'carbon_lag_12_norm', 'carbon_lag_24_norm', 'carbon_lag_48_norm',
        'carbon_rolling_mean_6_norm', 'carbon_rolling_mean_24_norm', 'carbon_rolling_std_24_norm',
        'wind_rolling_mean_12_norm', 'wind_rolling_std_12_norm', 'renewable_rolling_mean_12_norm',
        'carbon_diff_1_norm', 'carbon_diff_24_norm', 'wind_diff_1_norm',
        'carbon_trend_up_24',
    ]
    
    df_final = df[['datetime'] + features].copy()
    
    print(f"      ✓ Engineered {len(features)} features")
    
    return df_final


def prepare_input(df):
    """Prepare last 48 rows as model input"""
    df_tail = df.tail(48).copy()
    
    if len(df_tail) < 48:
        return None
    
    X = df_tail.drop(columns=['datetime']).values
    X = np.expand_dims(X, axis=0)  # Shape: (1, 48, 35)
    
    return X


# ============================================================================
# STEP 3: MODEL LOADING & PREDICTION
# ============================================================================

def load_models(zone):
    """Load 3 trained ensemble models"""
    models = []
    for i in range(1, 4):
        path = MODELS_DIR / f"{zone}_model_{i}_03.keras"
        if not path.exists():
            print(f"        ✗ Model not found: {path}")
            return None
        try:
            model = keras.models.load_model(path)
            models.append(model)
        except Exception as e:
            print(f"        ✗ Load error: {e}")
            return None
    return models


def predict_ensemble(models, X, zone):
    """Make ensemble prediction and denormalize"""
    try:
        preds = []
        for model in models:
            pred = model.predict(X, verbose=0)[0]
            preds.append(pred)
        
        # Average predictions
        ensemble = np.mean(preds, axis=0)
        
        # Denormalize from [0,1] to real carbon values
        min_c, max_c = CARBON_RANGES[zone]
        ensemble_real = ensemble * (max_c - min_c) + min_c
        
        return ensemble_real
        
    except Exception as e:
        print(f"        ✗ Prediction error: {e}")
        return None


# ============================================================================
# STEP 4: VISUALIZATION
# ============================================================================

def plot_all_forecasts(all_forecasts):
    """Plot 24h forecasts for all regions"""
    try:
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        hours = np.arange(1, 25)
        
        for idx, (zone, forecast) in enumerate(sorted(all_forecasts.items())):
            ax = axes[idx]
            ax.plot(hours, forecast, 'b-', linewidth=2, marker='o', markersize=3)
            ax.fill_between(hours, forecast, alpha=0.2, color='blue')
            
            # Highlight green hour
            min_idx = np.argmin(forecast)
            ax.axvline(min_idx + 1, color='green', linestyle='--', alpha=0.5, linewidth=2)
            
            ax.set_title(f"{ZONES[zone]} ({zone})", fontweight='bold', fontsize=11)
            ax.set_xlabel("Hours Ahead", fontsize=9)
            ax.set_ylabel("Carbon (gCO₂/kWh)", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 25)
        
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "forecasts_all_regions.png"
        plt.savefig(plot_path, dpi=250, bbox_inches='tight')
        print(f"\n  ✓ Saved plot: {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"\n  ✗ Plot error: {e}")


def plot_comparison(all_forecasts):
    """Bar chart comparing regions"""
    try:
        regions = []
        avg_carbon = []
        min_carbon = []
        
        for zone in sorted(ZONES.keys()):
            if zone in all_forecasts:
                regions.append(ZONES[zone])
                avg_carbon.append(np.mean(all_forecasts[zone]))
                min_carbon.append(np.min(all_forecasts[zone]))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(regions))
        width = 0.35
        
        ax.bar(x - width/2, avg_carbon, width, label='24h Average', alpha=0.8)
        ax.bar(x + width/2, min_carbon, width, label='24h Minimum (Green)', alpha=0.8, color='green')
        
        ax.set_xlabel('Region', fontweight='bold')
        ax.set_ylabel('Carbon Intensity (gCO₂/kWh)', fontweight='bold')
        ax.set_title('24-Hour Carbon Forecast Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        comp_path = OUTPUT_DIR / "comparison_regions.png"
        plt.savefig(comp_path, dpi=250, bbox_inches='tight')
        print(f"  ✓ Saved comparison: {comp_path}")
        plt.close()
        
    except Exception as e:
        print(f"  ✗ Comparison plot error: {e}")


# ============================================================================
# STEP 5: RECOMMENDATIONS
# ============================================================================

def print_recommendations(all_forecasts):
    """Print deployment recommendations"""
    print("\n" + "="*80)
    print("🌱 DEPLOYMENT RECOMMENDATIONS")
    print("="*80)
    
    # Find best overall
    best_zone = None
    best_hour = None
    best_carbon = float('inf')
    
    for zone, forecast in all_forecasts.items():
        min_idx = np.argmin(forecast)
        min_val = forecast[min_idx]
        if min_val < best_carbon:
            best_carbon = min_val
            best_zone = zone
            best_hour = min_idx + 1
    
    if best_zone:
        now = datetime.now(timezone.utc)
        deploy_time = now + timedelta(hours=int(best_hour))
        
        print(f"\n🏆 BEST OVERALL DEPLOYMENT:")
        print(f"   Region: {ZONES[best_zone]} ({best_zone})")
        print(f"   Deploy at: Hour {best_hour} (in {best_hour}h from now)")
        print(f"   Time: {deploy_time.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"   Expected Carbon: {best_carbon:.1f} gCO₂/kWh")
    
    # Top 3 per region
    print(f"\n📍 GREEN HOURS PER REGION (Top 3):")
    print("-" * 80)
    
    for zone in sorted(ZONES.keys()):
        if zone not in all_forecasts:
            continue
        
        forecast = all_forecasts[zone]
        top3_idx = np.argsort(forecast)[:3]
        
        print(f"\n{ZONES[zone]:20s}: ", end="")
        for idx in top3_idx:
            print(f"H{idx+1:2d}({forecast[idx]:5.0f}) ", end="")
    
    print("\n\n" + "="*80 + "\n")


def save_csv_results(all_forecasts):
    """Save forecast results to CSV"""
    try:
        all_data = []
        now = datetime.now(timezone.utc)
        
        for zone, forecast in all_forecasts.items():
            for hour, carbon in enumerate(forecast, 1):
                all_data.append({
                    'zone': zone,
                    'region_name': ZONES[zone],
                    'hour_ahead': hour,
                    'forecast_time_utc': (now + timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M'),
                    'carbon_intensity_gco2_kwh': round(carbon, 2)
                })
        
        df = pd.DataFrame(all_data)
        csv_path = OUTPUT_DIR / f"forecasts_{now.strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved CSV: {csv_path}")
        
    except Exception as e:
        print(f"  ✗ CSV save error: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='EcoDeploy Production Prediction')
    parser.add_argument('--api-key', default=None, help='Electricity Maps API key')
    parser.add_argument('--dummy', action='store_true', help='Use dummy data (no API)')
    
    args = parser.parse_args()
    
    # Check API key
    if not args.dummy and not args.api_key:
        # Try environment variable
        args.api_key = os.getenv('ELECTRICITY_MAP_TOKEN')
        if not args.api_key:
            print("\n✗ ERROR: No API key provided!")
            print("   Use: --api-key YOUR_KEY")
            print("   Or:  --dummy (for testing)")
            print("   Or:  Set ELECTRICITY_MAP_TOKEN environment variable")
            return
    
    print("\n" + "="*80)
    print("ECODEPLOY PRODUCTION PREDICTION SYSTEM")
    print("="*80)
    print(f"Mode: {'DUMMY DATA' if args.dummy else 'REAL API DATA'}")
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
    
    all_forecasts = {}
    
    for zone in sorted(ZONES.keys()):
        print(f"\n{zone} - {ZONES[zone]}:")
        print("-" * 40)
        
        try:
            # 1. Fetch data
            print(f"  1. Data collection")
            if args.dummy:
                df_raw = generate_dummy_data(zone, hours=120)
            else:
                df_raw = fetch_real_data(zone, args.api_key, hours=120)
            
            if df_raw is None or len(df_raw) < 72:
                print(f"      ✗ Insufficient data")
                continue
            
            print(f"      ✓ Collected {len(df_raw)} rows")
            
            # 2. Engineer features
            print(f"  2. Feature engineering")
            df_eng = engineer_features_exact_35(df_raw)
            
            if df_eng is None or len(df_eng) < 48:
                print(f"      ✗ Insufficient engineered data")
                continue
            
            # 3. Prepare input
            print(f"  3. Prepare input")
            X = prepare_input(df_eng)
            if X is None:
                print(f"      ✗ Input preparation failed")
                continue
            print(f"      ✓ Input shape: {X.shape}")
            
            # 4. Load models
            print(f"  4. Load models")
            models = load_models(zone)
            if models is None:
                continue
            print(f"      ✓ Loaded 3 models")
            
            # 5. Predict
            print(f"  5. Predict 24h forecast")
            forecast = predict_ensemble(models, X, zone)
            if forecast is None:
                continue
            
            print(f"      ✓ Success!")
            print(f"        Avg: {np.mean(forecast):.1f} gCO₂/kWh")
            print(f"        Min: {np.min(forecast):.1f} (Hour {np.argmin(forecast)+1})")
            print(f"        Max: {np.max(forecast):.1f} (Hour {np.argmax(forecast)+1})")
            
            all_forecasts[zone] = forecast
            
        except Exception as e:
            print(f"      ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Results
    if all_forecasts:
        print("\n" + "="*80)
        print("GENERATING OUTPUTS")
        print("="*80)
        print(f"✓ Forecasts generated for {len(all_forecasts)}/{len(ZONES)} regions\n")
        
        plot_all_forecasts(all_forecasts)
        plot_comparison(all_forecasts)
        save_csv_results(all_forecasts)
        print_recommendations(all_forecasts)
        
        print("✅ COMPLETE!")
        print(f"\nOutputs saved to: {OUTPUT_DIR.absolute()}")
        
    else:
        print("\n✗ No forecasts generated!")
        print("\nTroubleshooting:")
        print("  1. Check API key is valid")
        print("  2. Check models exist: forecasting/data/models_365d_03/")
        print("  3. Check internet connection")
        print("  4. Try --dummy mode for testing")


if __name__ == "__main__":
    main()
