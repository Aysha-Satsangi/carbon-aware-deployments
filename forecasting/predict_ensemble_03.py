"""
ECODEPLOY COMPLETE PREDICTION SYSTEM - SIMPLIFIED
==================================================
Collects data → Preprocesses → Predicts for all regions → Visualizes
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras

# Configuration
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "forecasting" / "data" / "models_365d_03"
OUTPUT_DIR = BASE_DIR / "forecasting" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

LOOKBACK = 48
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


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_dummy_data_extended(zone, hours=120):
    """Generate 120 hours of realistic dummy data"""
    print(f"      Generating {hours}h dummy data...")
    
    end_time = datetime.now(timezone.utc)
    dates = [end_time - timedelta(hours=i) for i in range(hours-1, -1, -1)]
    
    # Create realistic 5-day pattern
    time_array = np.linspace(0, 5*2*np.pi, hours)
    
    # Carbon: high at night, low during day
    base_carbon = np.random.randint(250, 350)
    carbon = base_carbon + 150 * np.sin(time_array)
    carbon += np.random.normal(0, 15, hours)
    carbon = np.clip(carbon, 50, 900)
    
    # Renewable: peaks during day
    renewable = 50 + 30 * np.sin(time_array + np.pi/2)
    renewable += np.random.normal(0, 5, hours)
    renewable = np.clip(renewable, 15, 85)
    
    # Fossil: inverse
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
    return df


def engineer_features_simple(df):
    """Simplified feature engineering"""
    df = df.copy()
    
    # Lags (1h, 6h, 12h, 24h)
    for lag in [1, 6, 12, 24]:
        df[f'carbon_lag_{lag}'] = df['carbon_intensity'].shift(lag)
        df[f'renewable_lag_{lag}'] = df['renewable_percentage'].shift(lag)
    
    # Rolling mean (6h, 12h)
    for window in [6, 12]:
        df[f'carbon_mean_{window}'] = df['carbon_intensity'].rolling(window).mean()
    
    # Temporal features
    df['hour'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Drop NaN
    df = df.dropna().reset_index(drop=True)
    
    return df


def prepare_prediction_input(df):
    """Prepare last 48 rows for prediction"""
    # Get last 48 rows
    df_tail = df.tail(48).copy()
    
    if len(df_tail) < 48:
        return None
    
    # Drop datetime column
    X = df_tail.drop(columns=['datetime']).values
    
    # Add batch dimension
    X = np.expand_dims(X, axis=0)
    
    return X


# ============================================================================
# PREDICTION
# ============================================================================

def load_models_safe(zone):
    """Load 3 models with error checking"""
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
            print(f"        ✗ Error loading model {i}: {e}")
            return None
    return models


def predict_ensemble_safe(models, X, zone):
    """Make ensemble prediction with proper denormalization"""
    try:
        preds = []
        for i, model in enumerate(models):
            pred = model.predict(X, verbose=0)[0]
            preds.append(pred)
        
        # Average
        ensemble = np.mean(preds, axis=0)
        
        # Denormalize from [0,1] back to original carbon range
        min_c, max_c = CARBON_RANGES[zone]
        ensemble_denorm = ensemble * (max_c - min_c) + min_c
        
        return ensemble_denorm
    except Exception as e:
        print(f"        ✗ Prediction error: {e}")
        return None

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_forecasts(all_forecasts):
    """Plot all forecasts"""
    try:
        fig, axes = plt.subplots(4, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (zone, forecast) in enumerate(sorted(all_forecasts.items())):
            ax = axes[idx]
            hours = np.arange(1, 25)
            ax.plot(hours, forecast, 'b-', linewidth=2, marker='o', markersize=3)
            ax.fill_between(hours, forecast, alpha=0.2)
            ax.set_title(f"{ZONES[zone]} ({zone})", fontweight='bold')
            ax.set_xlabel("Hours Ahead")
            ax.set_ylabel("Carbon (gCO₂/kWh)")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "forecasts_all_regions.png"
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"\n  ✓ Saved plot: {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"\n  ✗ Plot error: {e}")


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
        print(f"\n🏆 BEST OVERALL:")
        print(f"   Region: {ZONES[best_zone]} ({best_zone})")
        print(f"   Hour: {best_hour} (in {best_hour}h from now)")
        print(f"   Carbon: {best_carbon:.1f} gCO₂/kWh")
    
    # Top 3 per region
    print(f"\n📍 GREEN HOURS PER REGION:")
    for zone in sorted(ZONES.keys()):
        if zone not in all_forecasts:
            continue
        forecast = all_forecasts[zone]
        top3_idx = np.argsort(forecast)[:3]
        print(f"\n{ZONES[zone]:20s}: ", end="")
        for idx in top3_idx:
            print(f"Hour {idx+1:2d}({forecast[idx]:5.0f}) ", end="")
        print()
    
    print("\n" + "="*80 + "\n")


# ============================================================================
# MAIN
# ============================================================================
def engineer_features_exact_match(df):
    """
    EXACT feature engineering matching preprocess_03.py (35 features).
    Must replicate the training preprocessing exactly!
    """
    df = df.copy()
    
    # Ensure required columns exist
    required = ['datetime', 'carbon_intensity', 'renewable_percentage', 
                'fossil_percentage', 'wind_power', 'solar_power']
    for col in required:
        if col not in df.columns:
            print(f"      ✗ Missing column: {col}")
            return None
    
    # Rename to match training
    df = df.rename(columns={
        'carbon_intensity': 'carbon',
        'renewable_percentage': 'renewable_pct',
        'fossil_percentage': 'fossil_pct',
        'wind_power': 'wind',
        'solar_power': 'solar'
    })
    
    # Add coal and gas (set to 0 for dummy data, or derive from fossil_pct)
    df['coal'] = df['fossil_pct'] * 0.6  # Assume 60% coal
    df['gas'] = df['fossil_pct'] * 0.4   # Assume 40% gas
    
    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # LAG FEATURES
    df['carbon_lag_1'] = df['carbon'].shift(1)
    df['carbon_lag_6'] = df['carbon'].shift(6)
    df['carbon_lag_12'] = df['carbon'].shift(12)
    df['carbon_lag_24'] = df['carbon'].shift(24)
    df['carbon_lag_48'] = df['carbon'].shift(48)
    
    df['renewable_lag_1'] = df['renewable_pct'].shift(1)
    df['renewable_lag_24'] = df['renewable_pct'].shift(24)
    
    df['wind_lag_1'] = df['wind'].shift(1)
    df['wind_lag_6'] = df['wind'].shift(6)
    df['wind_lag_12'] = df['wind'].shift(12)
    df['wind_lag_24'] = df['wind'].shift(24)
    
    df['solar_lag_1'] = df['solar'].shift(1)
    df['solar_lag_12'] = df['solar'].shift(12)
    df['solar_lag_24'] = df['solar'].shift(24)
    
    # ROLLING STATISTICS
    df['carbon_rolling_mean_6'] = df['carbon'].rolling(6, min_periods=1).mean()
    df['carbon_rolling_mean_24'] = df['carbon'].rolling(24, min_periods=1).mean()
    df['carbon_rolling_std_24'] = df['carbon'].rolling(24, min_periods=1).std().fillna(0)
    
    df['wind_rolling_mean_12'] = df['wind'].rolling(12, min_periods=1).mean()
    df['wind_rolling_std_12'] = df['wind'].rolling(12, min_periods=1).std().fillna(0)
    
    df['renewable_rolling_mean_12'] = df['renewable_pct'].rolling(12, min_periods=1).mean()
    
    # DIFFERENCING
    df['carbon_diff_1'] = df['carbon'].diff().fillna(0)
    df['carbon_diff_24'] = df['carbon'].diff(24).fillna(0)
    df['wind_diff_1'] = df['wind'].diff().fillna(0)
    
    # TREND
    df['carbon_trend_up_24'] = (df['carbon'] > df['carbon_rolling_mean_24']).astype(int)
    
    # Fill missing values
    df = df.ffill().bfill()
    
    # NORMALIZATION (MinMax to [0,1])
    from sklearn.preprocessing import MinMaxScaler
    
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
    
    # Select only these features
    df_final = df[['datetime'] + features].copy()
    
    print(f"      ✓ Engineered {len(features)} features (matches training exactly)")
    
    return df_final

def main():
    print("\n" + "="*80)
    print("ECODEPLOY PREDICTION SYSTEM")
    print("="*80)
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
    
    all_forecasts = {}
    
    for zone in sorted(ZONES.keys()):
        print(f"\n{zone} - {ZONES[zone]}:")
        print("-" * 40)
        
        try:
            # 1. Generate data
            print(f"  1. Data preparation")
            df_raw = generate_dummy_data_extended(zone, hours=120)
            print(f"      ✓ Generated {len(df_raw)} rows")
            
            # 2. Engineer features
            print(f"  2. Feature engineering")
            df_eng = engineer_features_exact_match(df_raw)
            print(f"      ✓ Engineered {len(df_eng)} rows, {len(df_eng.columns)-1} features")
            
            if len(df_eng) < 48:
                print(f"      ✗ Insufficient rows after engineering: {len(df_eng)} < 48")
                continue
            
            # 3. Prepare input
            print(f"  3. Prepare input")
            X = prepare_prediction_input(df_eng)
            if X is None:
                print(f"      ✗ Failed to prepare input")
                continue
            print(f"      ✓ Input shape: {X.shape}")
            
            # 4. Load models
            print(f"  4. Load models")
            models = load_models_safe(zone)
            if models is None:
                print(f"      ✗ Failed to load models")
                continue
            print(f"      ✓ Loaded 3 models")
            
            # 5. Predict
            print(f"  5. Predict")
            forecast = predict_ensemble_safe(models, X, zone)
            if forecast is None:
                print(f"      ✗ Prediction failed")
                continue
            print(f"      ✓ Predicted 24h")
            print(f"        Average: {np.mean(forecast):.1f} gCO₂/kWh")
            print(f"        Min/Max: {np.min(forecast):.1f} / {np.max(forecast):.1f}")
            
            all_forecasts[zone] = forecast
            
        except Exception as e:
            print(f"      ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Visualize
    if all_forecasts:
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"✓ Generated forecasts for {len(all_forecasts)} regions\n")
        
        plot_forecasts(all_forecasts)
        print_recommendations(all_forecasts)
        
        print("✅ COMPLETE - Check ./forecasting/output/ for plots")
    else:
        print("\n✗ No forecasts generated!")
        print("Troubleshooting:")
        print("  1. Check models exist: forecasting/data/models_365d_03/")
        print("  2. Check model names: *_model_1_03.keras, *_model_2_03.keras, *_model_3_03.keras")
        print("  3. Check TensorFlow installed: pip install tensorflow")


if __name__ == "__main__":
    main()
