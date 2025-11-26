import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
raw_dir = Path("data/raw")
plot_dir = Path("data/plots")
plot_dir.mkdir(exist_ok=True)

print("="*80)
print("📊 TEMPORAL ANALYSIS: TIME PERIOD & PATTERNS")
print("="*80)

# ============================================================
# 1. DETERMINE TIME PERIOD
# ============================================================
print("\n1️⃣ DATA TIME PERIOD ANALYSIS:\n")

all_dates = []
for zone in zones:
    try:
        with open(raw_dir / f"{zone}_carbon_180d.json") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['data'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        all_dates.extend(df['datetime'].tolist())
    except Exception as e:
        print(f"❌ {zone}: {e}")

all_dates = sorted(all_dates)
start_date = all_dates[0]
end_date = all_dates[-1]
num_days = (end_date - start_date).days

print(f"📅 START DATE: {start_date.strftime('%B %d, %Y (%A)')}")
print(f"📅 END DATE:   {end_date.strftime('%B %d, %Y (%A)')}")
print(f"⏱️  DURATION:   {num_days} days (6 months)")
print(f"📊 RECORDS:    {len(all_dates)} hourly records")

print("\n📆 BREAKDOWN BY MONTH:")
df_all = pd.DataFrame({'datetime': all_dates})
df_all['year_month'] = df_all['datetime'].dt.strftime('%B %Y')
df_all['month'] = df_all['datetime'].dt.month
df_all['year'] = df_all['datetime'].dt.year

month_counts = df_all['year_month'].value_counts().sort_index()
for month, count in month_counts.items():
    print(f"   {month}: {count} records")

# ============================================================
# 2. DAY vs NIGHT ANALYSIS (using hour of day)
# ============================================================
print("\n" + "="*80)
print("2️⃣ DAY vs NIGHT CARBON INTENSITY ANALYSIS:\n")

day_night_analysis = []

for zone in zones:
    try:
        with open(raw_dir / f"{zone}_carbon_180d.json") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['data'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['carbonIntensity'] = pd.to_numeric(df['carbonIntensity'], errors='coerce')
        
        # Define day (6h-18h) and night (18h-6h next day)
        day_hours = df[df['hour'].between(6, 18)]
        night_hours = df[(df['hour'] >= 18) | (df['hour'] < 6)]
        
        day_avg = day_hours['carbonIntensity'].mean()
        night_avg = night_hours['carbonIntensity'].mean()
        
        # Day vs Night difference
        difference = day_avg - night_avg
        percent_change = (difference / night_avg) * 100 if night_avg > 0 else 0
        
        day_night_analysis.append({
            'Zone': zone,
            'Day Avg (6h-18h)': f"{day_avg:.1f}",
            'Night Avg (18h-6h)': f"{night_avg:.1f}",
            'Difference': f"{difference:.1f}",
            'Change %': f"{percent_change:+.1f}%",
            'Pattern': '🌞 Daytime High' if difference > 0 else '🌙 Nighttime High'
        })
        
        print(f"✅ {zone}:")
        print(f"   Day (6h-18h):   {day_avg:.1f} gCO₂/kWh")
        print(f"   Night (18h-6h): {night_avg:.1f} gCO₂/kWh")
        print(f"   Difference:     {difference:+.1f} gCO₂/kWh ({percent_change:+.1f}%)")
        if difference > 0:
            print(f"   → DAYTIME is {percent_change:.1f}% MORE carbon-intensive")
        else:
            print(f"   → NIGHTTIME is {abs(percent_change):.1f}% MORE carbon-intensive")
        print()
        
    except Exception as e:
        print(f"❌ {zone}: {e}")

# Save day/night analysis
day_night_df = pd.DataFrame(day_night_analysis)
day_night_df.to_csv(plot_dir / 'day_vs_night_analysis.csv', index=False)
print(f"✅ Saved: data/plots/day_vs_night_analysis.csv\n")

# ============================================================
# 3. MONTHLY PATTERNS (if data spans multiple months)
# ============================================================
print("="*80)
print("3️⃣ MONTHLY SEASONAL PATTERNS:\n")

fig, axes = plt.subplots(4, 2, figsize=(16, 12))
axes = axes.flatten()

monthly_data = []

for idx, zone in enumerate(zones):
    try:
        with open(raw_dir / f"{zone}_carbon_180d.json") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['data'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['carbonIntensity'] = pd.to_numeric(df['carbonIntensity'], errors='coerce')
        df['month'] = df['datetime'].dt.strftime('%B')
        df['month_num'] = df['datetime'].dt.month
        
        # Monthly average
        monthly_avg = df.groupby(['month_num', 'month'])['carbonIntensity'].agg(['mean', 'std', 'min', 'max']).reset_index()
        monthly_avg = monthly_avg.sort_values('month_num')
        
        # Plot
        axes[idx].bar(range(len(monthly_avg)), monthly_avg['mean'], 
                     color='steelblue', alpha=0.7, label='Mean', yerr=monthly_avg['std'], capsize=5)
        axes[idx].plot(range(len(monthly_avg)), monthly_avg['mean'], 'ro-', linewidth=2, markersize=6, label='Trend')
        
        axes[idx].set_xticks(range(len(monthly_avg)))
        axes[idx].set_xticklabels(monthly_avg['month'], rotation=45, ha='right', fontsize=9)
        axes[idx].set_title(f'{zone} - Monthly Carbon Intensity', fontweight='bold', fontsize=10)
        axes[idx].set_ylabel('gCO₂/kWh', fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].legend(fontsize=8)
        
        # Store for summary
        for _, row in monthly_avg.iterrows():
            monthly_data.append({
                'Zone': zone,
                'Month': row['month'],
                'Avg Carbon': f"{row['mean']:.1f}",
                'Std Dev': f"{row['std']:.1f}",
                'Min': f"{row['min']:.1f}",
                'Max': f"{row['max']:.1f}"
            })
        
        print(f"✅ {zone}:")
        for _, row in monthly_avg.iterrows():
            print(f"   {row['month']:12}: {row['mean']:6.1f} ± {row['std']:.1f} gCO₂/kWh")
        print()
        
    except Exception as e:
        print(f"❌ {zone}: {e}")

plt.suptitle('📆 Monthly Carbon Intensity Trends', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(plot_dir / 'visualization_4_monthly_patterns.png', dpi=150, bbox_inches='tight')
print(f"\n📁 Saved: data/plots/visualization_4_monthly_patterns.png\n")
plt.close()

# Save monthly data
monthly_df = pd.DataFrame(monthly_data)
monthly_df.to_csv(plot_dir / 'monthly_analysis.csv', index=False)
print(f"✅ Saved: data/plots/monthly_analysis.csv\n")

# ============================================================
# 4. DAY/NIGHT VISUALIZATION
# ============================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
axes = axes.flatten()
fig.suptitle('🌞 vs 🌙 Day and Night Carbon Patterns', fontsize=14, fontweight='bold')

for idx, zone in enumerate(zones):
    try:
        with open(raw_dir / f"{zone}_carbon_180d.json") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['data'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['carbonIntensity'] = pd.to_numeric(df['carbonIntensity'], errors='coerce')
        
        # Day vs Night
        day = df[df['hour'].between(6, 18)]['carbonIntensity'].values
        night = df[(df['hour'] >= 18) | (df['hour'] < 6)]['carbonIntensity'].values
        
        bp = axes[idx].boxplot([day, night], labels=['Day (6h-18h)', 'Night (18h-6h)'], patch_artist=True)
        
        colors = ['#FFD700', '#4B0082']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[idx].set_title(f'{zone}', fontweight='bold', fontsize=10)
        axes[idx].set_ylabel('Carbon Intensity (gCO₂/kWh)', fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        day_mean = day.mean()
        night_mean = night.mean()
        axes[idx].text(0.5, 0.95, f'Day: {day_mean:.0f} | Night: {night_mean:.0f}', 
                      transform=axes[idx].transAxes, ha='center', va='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)
        
    except Exception as e:
        print(f"❌ {zone}: {e}")

plt.tight_layout()
plt.savefig(plot_dir / 'visualization_5_day_vs_night.png', dpi=150, bbox_inches='tight')
print(f"📁 Saved: data/plots/visualization_5_day_vs_night.png\n")
plt.close()

print("="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
