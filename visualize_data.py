import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
# Adjust this path if running from project root:
raw_dir = Path("forecasting/data/raw_data")
plot_dir = Path("data/plots_365d/360d_visualizations")
plot_dir.mkdir(parents=True, exist_ok=True)

print("📊 Creating comprehensive visualizations for all zones...\n")

# 1. CARBON INTENSITY - 8 ZONES (8x1 grid)
fig, axes = plt.subplots(8, 1, figsize=(16, 14))
fig.suptitle('🌍 Carbon Intensity Trends (365 Days) - All 8 Zones', fontsize=16, fontweight='bold', y=0.995)

for idx, zone in enumerate(zones):
    try:
        with open(raw_dir / f"{zone}_carbon_365d.json") as f:
            data = json.load(f)
        df = pd.DataFrame(data['data'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        
        axes[idx].plot(df['datetime'], df['carbonIntensity'], linewidth=0.8, color='steelblue', alpha=0.8)
        axes[idx].fill_between(df['datetime'], df['carbonIntensity'], alpha=0.3, color='steelblue')
        
        mean_val = df['carbonIntensity'].mean()
        min_val = df['carbonIntensity'].min()
        max_val = df['carbonIntensity'].max()
        
        axes[idx].axhline(y=mean_val, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Avg: {mean_val:.0f}')
        axes[idx].set_title(f'{zone} | Min: {min_val:.0f} | Avg: {mean_val:.0f} | Max: {max_val:.0f}', fontweight='bold', fontsize=10)
        axes[idx].set_ylabel('gCO₂/kWh', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(loc='upper right', fontsize=8)
        
        print(f"✅ {zone}: {len(df)} records, Range: {min_val:.0f}-{max_val:.0f}")
        
    except Exception as e:
        print(f"❌ {zone}: {e}")

plt.tight_layout()
plt.savefig(plot_dir / 'visualization_1_carbon_all_zones.png', dpi=150, bbox_inches='tight')
print("\n📁 Saved: data/plots_365d/360d_visualizations/visualization_1_carbon_all_zones.png\n")
plt.close()

# 2. POWER BREAKDOWN - 8 ZONES (Carbon + Renewables)
fig, axes = plt.subplots(8, 2, figsize=(16, 14))
fig.suptitle('⚡ Carbon Intensity & Renewable % Comparison - All 8 Zones', fontsize=16, fontweight='bold', y=0.995)

for idx, zone in enumerate(zones):
    try:
        # Load carbon
        with open(raw_dir / f"{zone}_carbon_365d.json") as f:
            carbon_data = json.load(f)
        df_carbon = pd.DataFrame(carbon_data['data'])
        df_carbon['datetime'] = pd.to_datetime(df_carbon['datetime'])
        df_carbon = df_carbon.sort_values('datetime')
        
        axes[idx, 0].plot(df_carbon['datetime'], df_carbon['carbonIntensity'], linewidth=0.8, color='darkred')
        axes[idx, 0].fill_between(df_carbon['datetime'], df_carbon['carbonIntensity'], alpha=0.3, color='red')
        axes[idx, 0].set_title(f'{zone} - Carbon Intensity', fontweight='bold', fontsize=9)
        axes[idx, 0].set_ylabel('gCO₂/kWh', fontsize=8)
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Load power breakdown
        try:
            with open(raw_dir / f"{zone}_power_365d.json") as f:
                power_data = json.load(f)
            df_power = pd.DataFrame(power_data['data'])
            df_power['datetime'] = pd.to_datetime(df_power['datetime'])
            df_power = df_power.sort_values('datetime')
            
            if 'renewablePercentage' in df_power.columns:
                renewable = df_power['renewablePercentage'].dropna()
                if len(renewable) > 0:
                    axes[idx, 1].plot(df_power['datetime'], df_power['renewablePercentage'], linewidth=0.8, color='green')
                    axes[idx, 1].fill_between(df_power['datetime'], df_power['renewablePercentage'], alpha=0.3, color='green')
                    axes[idx, 1].set_title(f'{zone} - Renewable %', fontweight='bold', fontsize=9)
                    axes[idx, 1].set_ylabel('Renewable %', fontsize=8)
                    axes[idx, 1].set_ylim(0, 100)
                    axes[idx, 1].grid(True, alpha=0.3)
                    print(f"✅ {zone}: Renewable data found ({len(renewable)} records)")
                else:
                    axes[idx, 1].text(0.5, 0.5, f'No renewable data for {zone}', ha='center', va='center', transform=axes[idx, 1].transAxes)
                    print(f"⚠️ {zone}: No renewable percentage data")
            else:
                axes[idx, 1].text(0.5, 0.5, f'No renewable field in {zone}', ha='center', va='center', transform=axes[idx, 1].transAxes)
                print(f"⚠️ {zone}: No renewable field")
        except FileNotFoundError:
            axes[idx, 1].text(0.5, 0.5, f'Power file not found for {zone}', ha='center', va='center', transform=axes[idx, 1].transAxes)
            print(f"⚠️ {zone}: Power breakdown file not found")
        
    except Exception as e:
        print(f"❌ {zone}: {e}")

plt.tight_layout()
plt.savefig(plot_dir / 'visualization_2_carbon_vs_renewable.png', dpi=150, bbox_inches='tight')
print("\n📁 Saved: data/plots_365d/360d_visualizations/visualization_2_carbon_vs_renewable.png\n")
plt.close()

# 3. HOURLY PATTERN - BEST TIME TO DEPLOY (8 zones)
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
axes = axes.flatten()
fig.suptitle('🟢 Greenest Hours of Day - Average Carbon by Hour (UTC)', fontsize=16, fontweight='bold')

for idx, zone in enumerate(zones):
    try:
        with open(raw_dir / f"{zone}_carbon_365d.json") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['data'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        
        hourly_avg = df.groupby('hour')['carbonIntensity'].agg(['mean', 'std'])
        
        axes[idx].bar(hourly_avg.index, hourly_avg['mean'], color='steelblue', alpha=0.7, label='Mean')
        axes[idx].fill_between(hourly_avg.index, 
                              hourly_avg['mean'] - hourly_avg['std'],
                              hourly_avg['mean'] + hourly_avg['std'],
                              alpha=0.3, color='steelblue', label='±1 Std Dev')
        
        best_hour = hourly_avg['mean'].idxmin()
        worst_hour = hourly_avg['mean'].idxmax()
        
        axes[idx].axvline(x=best_hour, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best: {best_hour}h')
        axes[idx].axvline(x=worst_hour, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Worst: {worst_hour}h')
        
        axes[idx].set_title(f'{zone} - Greenest: {best_hour}:00 UTC ({hourly_avg["mean"].min():.0f} gCO₂/kWh)', fontweight='bold', fontsize=10)
        axes[idx].set_xlabel('Hour (UTC)', fontsize=9)
        axes[idx].set_ylabel('Avg Carbon (gCO₂/kWh)', fontsize=9)
        axes[idx].set_xlim(-0.5, 23.5)
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].legend(fontsize=7, loc='upper right')
        
        print(f"✅ {zone}: Greenest hour: {best_hour}:00 UTC")
        
    except Exception as e:
        print(f"❌ {zone}: {e}")

plt.tight_layout()
plt.savefig(plot_dir / 'visualization_3_hourly_patterns.png', dpi=150, bbox_inches='tight')
print("\n📁 Saved: data/plots_365d/360d_visualizations/visualization_3_hourly_patterns.png\n")
plt.close()

# 4. STATISTICS TABLE
print("\n" + "="*80)
print("📊 STATISTICS SUMMARY FOR ALL ZONES")
print("="*80)

stats_list = []
for zone in zones:
    try:
        with open(raw_dir / f"{zone}_carbon_365d.json") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['data'])
        df['carbonIntensity'] = pd.to_numeric(df['carbonIntensity'], errors='coerce')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        
        hourly_avg = df.groupby('hour')['carbonIntensity'].mean()
        
        stats_list.append({
            'Zone': zone,
            'Records': len(df),
            'Min': f"{df['carbonIntensity'].min():.0f}",
            'Max': f"{df['carbonIntensity'].max():.0f}",
            'Avg': f"{df['carbonIntensity'].mean():.0f}",
            'Std': f"{df['carbonIntensity'].std():.0f}",
            'Greenest Hour': f"{hourly_avg.idxmin()}:00",
            'Greenest Value': f"{hourly_avg.min():.0f}",
            'Dirtiest Hour': f"{hourly_avg.idxmax()}:00",
            'Dirtiest Value': f"{hourly_avg.max():.0f}"
        })
    except Exception as e:
        print(f"❌ {zone}: {e}")

stats_df = pd.DataFrame(stats_list)
print(stats_df.to_string(index=False))

# Save as CSV
stats_df.to_csv(plot_dir / 'statistics_summary.csv', index=False)
print(f"\n✅ Saved: data/plots_365d/360d_visualizations/statistics_summary.csv\n")

print("="*80)
print("🎉 All visualizations complete!")
print(f"📁 Output saved in: {plot_dir}")
print("="*80)
