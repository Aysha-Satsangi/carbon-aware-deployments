import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
import pickle
from datetime import datetime, timedelta, timezone
import pytz
import matplotlib.pyplot as plt

PROCESSED = Path("data/processed")
MODELS = Path("data/models")
SCALERS = Path("data/scalers")
PLOTS = Path("data/plots")

zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
LOOKBACK = 24

# Timezone setup
IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.UTC

print("="*80)
print("GENERATING 24-HOUR CARBON FORECASTS (Indian Standard Time)")
print("="*80)

forecasts_summary = []
all_forecasts = {}

for zone in zones:
    try:
        print(f"\n{zone}:")
        
        # Load latest data
        df = pd.read_csv(PROCESSED / f"{zone}_processed.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Get last 24 hours
        latest_24h = df.tail(LOOKBACK).drop('datetime', axis=1).values
        
        if len(latest_24h) < LOOKBACK:
            print(f"   Not enough data (only {len(latest_24h)} hours)")
            continue
        
        # Load model and scaler
        model = tf.keras.models.load_model(MODELS / f"{zone}_best.keras")
        with open(SCALERS / f"{zone}_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        # Make prediction (normalized scale)
        X = latest_24h.reshape(1, LOOKBACK, latest_24h.shape[1])
        y_pred_scaled = model.predict(X, verbose=0)[0]
        
        # Inverse transform to original scale
        y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Generate timestamps for next 24h (in UTC first)
        last_time = df['datetime'].iloc[-1]
        forecast_times_utc = [last_time + timedelta(hours=i+1) for i in range(24)]
        
        # Convert to IST
        forecast_times_ist = [t.replace(tzinfo=UTC).astimezone(IST) for t in forecast_times_utc]
        
        # Current (latest) value
        current_carbon_scaled = df['carbon'].iloc[-1]
        current_carbon_original = scaler.inverse_transform([[current_carbon_scaled]])[0][0]
        
        # Statistics
        min_val = y_pred_original.min()
        max_val = y_pred_original.max()
        avg_val = y_pred_original.mean()
        
        # Best deployment time (lowest carbon)
        best_idx = y_pred_original.argmin()
        worst_idx = y_pred_original.argmax()
        
        best_time_ist = forecast_times_ist[best_idx]
        worst_time_ist = forecast_times_ist[worst_idx]
        best_time_utc = forecast_times_utc[best_idx]
        worst_time_utc = forecast_times_utc[worst_idx]
        
        # Improvement/worsening compared to now
        improvement = ((current_carbon_original - min_val) / current_carbon_original) * 100
        worsening = ((max_val - current_carbon_original) / current_carbon_original) * 100
        
        # Store
        forecasts_summary.append({
            'Zone': zone,
            'Current_Carbon': f"{current_carbon_original:.1f}",
            'Forecast_Min': f"{min_val:.1f}",
            'Forecast_Max': f"{max_val:.1f}",
            'Forecast_Avg': f"{avg_val:.1f}",
            'Best_Time_IST': best_time_ist.strftime('%H:%M IST'),
            'Best_Time_UTC': best_time_utc.strftime('%H:%M UTC'),
            'Best_Carbon': f"{min_val:.1f}",
            'Best_Improvement_%': f"{improvement:.1f}%",
            'Worst_Time_IST': worst_time_ist.strftime('%H:%M IST'),
            'Worst_Time_UTC': worst_time_utc.strftime('%H:%M UTC'),
            'Worst_Carbon': f"{max_val:.1f}",
            'Worst_Worsening_%': f"{worsening:.1f}%"
        })
        
        all_forecasts[zone] = {
            'times_ist': forecast_times_ist,
            'times_utc': forecast_times_utc,
            'carbon': y_pred_original,
            'current': current_carbon_original
        }
        
        # Display
        now_ist = datetime.now(IST)
        print(f"   Current: {current_carbon_original:.1f} gCO2/kWh")
        print(f"   Next 24h: {min_val:.1f} - {max_val:.1f} gCO2/kWh")
        print(f"   Average: {avg_val:.1f} gCO2/kWh")
        print(f"   BEST:    {best_time_ist.strftime('%H:%M IST')} ({best_time_utc.strftime('%H:%M UTC')}) = {min_val:.1f} gCO2/kWh ({improvement:.1f}% better)")
        print(f"   WORST:   {worst_time_ist.strftime('%H:%M IST')} ({worst_time_utc.strftime('%H:%M UTC')}) = {max_val:.1f} gCO2/kWh ({worsening:.1f}% worse)")
        
    except Exception as e:
        print(f"   Error: {e}")

# Summary table
print("\n" + "="*80)
print("24-HOUR FORECAST SUMMARY (Indian Standard Time):\n")

summary_df = pd.DataFrame(forecasts_summary)
print(summary_df.to_string(index=False))

# Save
summary_df.to_csv(MODELS / "forecasts_24h_summary_ist.csv", index=False)
print(f"\nSaved: {MODELS}/forecasts_24h_summary_ist.csv")

# CREATE VISUALIZATION with IST
print(f"\nCreating forecast visualization...")

fig, axes = plt.subplots(4, 2, figsize=(16, 14))
axes = axes.flatten()
fig.suptitle('24-Hour Carbon Intensity Forecasts - All Zones\n(Indian Standard Time - IST)', fontsize=16, fontweight='bold')

for idx, zone in enumerate(zones):
    if zone not in all_forecasts:
        continue
    
    forecast_data = all_forecasts[zone]
    times_ist = forecast_data['times_ist']
    carbon = forecast_data['carbon']
    current = forecast_data['current']
    
    hours = [t.strftime('%H:%M') for t in times_ist]
    
    # Plot
    axes[idx].plot(range(24), carbon, 'b-', linewidth=2, marker='o', markersize=4, label='Forecast')
    axes[idx].axhline(y=current, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Current: {current:.0f}')
    
    # Highlight best and worst
    best_idx = np.argmin(carbon)
    worst_idx = np.argmax(carbon)
    
    axes[idx].scatter([best_idx], [carbon[best_idx]], color='green', s=100, marker='*', zorder=5, label=f'Best: {carbon[best_idx]:.0f}')
    axes[idx].scatter([worst_idx], [carbon[worst_idx]], color='red', s=100, marker='*', zorder=5, label=f'Worst: {carbon[worst_idx]:.0f}')
    
    # Fill area
    axes[idx].fill_between(range(24), carbon, alpha=0.3, color='steelblue')
    
    axes[idx].set_title(f'{zone} - Next 24 Hours (IST)', fontweight='bold', fontsize=10)
    axes[idx].set_xlabel('Time (IST)', fontsize=9)
    axes[idx].set_ylabel('Carbon Intensity (gCO2/kWh)', fontsize=9)
    axes[idx].set_xticks(range(0, 24, 3))
    axes[idx].set_xticklabels([hours[i] for i in range(0, 24, 3)], rotation=45, ha='right', fontsize=8)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(fontsize=8, loc='best')

plt.tight_layout()
plt.savefig(PLOTS / 'forecasts_24h_all_zones_ist.png', dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS}/forecasts_24h_all_zones_ist.png")
plt.close()

# CREATE DEPLOYMENT RECOMMENDATION
print(f"\nCreating deployment recommendations...")

now_ist = datetime.now(IST)

recommendation = """
CARBON-AWARE CLOUD DEPLOYMENT RECOMMENDATIONS
===============================================

Generated: {} (IST)

NOTE: All times shown in Indian Standard Time (IST = UTC+5:30)
       UTC times also provided in parentheses for reference

DEPLOYMENT STRATEGY:
--------------------

1. IMMEDIATE (Best NOW - Next Hour):
""".format(now_ist.strftime('%Y-%m-%d %H:%M:%S'))

# Find best zones right now
current_times = []
for zone in zones:
    if zone in all_forecasts:
        current = all_forecasts[zone]['current']
        carbon = all_forecasts[zone]['carbon'][0]
        current_times.append((zone, current, carbon))

current_times.sort(key=lambda x: x[1])

for zone, current, forecast in current_times[:3]:
    recommendation += f"\n   {zone}: {current:.0f} gCO2/kWh (next hour forecast: {forecast:.0f})"

recommendation += "\n\n2. BEST IN NEXT 24H:\n"

# Find best times across all zones
best_opportunities = []
for zone in zones:
    if zone in all_forecasts:
        carbon = all_forecasts[zone]['carbon']
        times_ist = all_forecasts[zone]['times_ist']
        times_utc = all_forecasts[zone]['times_utc']
        best_idx = np.argmin(carbon)
        best_opportunities.append((zone, times_ist[best_idx], times_utc[best_idx], carbon[best_idx]))

best_opportunities.sort(key=lambda x: x[3])

for zone, best_time_ist, best_time_utc, best_carbon in best_opportunities[:5]:
    recommendation += f"\n   {zone}: {best_time_ist.strftime('%H:%M IST')} ({best_time_utc.strftime('%H:%M UTC')}) = {best_carbon:.0f} gCO2/kWh"

recommendation += "\n\n3. WORST TIMES TO AVOID:\n"

# Find worst times
worst_opportunities = []
for zone in zones:
    if zone in all_forecasts:
        carbon = all_forecasts[zone]['carbon']
        times_ist = all_forecasts[zone]['times_ist']
        times_utc = all_forecasts[zone]['times_utc']
        worst_idx = np.argmax(carbon)
        worst_opportunities.append((zone, times_ist[worst_idx], times_utc[worst_idx], carbon[worst_idx]))

worst_opportunities.sort(key=lambda x: x[3], reverse=True)

for zone, worst_time_ist, worst_time_utc, worst_carbon in worst_opportunities[:5]:
    recommendation += f"\n   {zone}: {worst_time_ist.strftime('%H:%M IST')} ({worst_time_utc.strftime('%H:%M UTC')}) = {worst_carbon:.0f} gCO2/kWh"

recommendation += "\n\n" + "="*60

with open(MODELS / "deployment_recommendations_ist.txt", 'w', encoding='utf-8') as f:
    f.write(recommendation)

print(recommendation)
print(f"\nSaved: {MODELS}/deployment_recommendations_ist.txt")

print("\n" + "="*80)
print("FORECASTING COMPLETE (IST)!")
print("="*80)
print("\nGenerated files:")
print("  1. forecasts_24h_summary_ist.csv")
print("  2. forecasts_24h_all_zones_ist.png")
print("  3. deployment_recommendations_ist.txt")
print("\nAll times in Indian Standard Time (IST)")
