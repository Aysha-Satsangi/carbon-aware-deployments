import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ZONE = "DE"  # Change zone as needed ('DE', 'US-MIDA-PJM', etc)
RAW_DIR = Path("forecasting/data/raw_data_02")

carbon_file = RAW_DIR / f"{ZONE}_carbon_365d_02.json"
power_file  = RAW_DIR / f"{ZONE}_power_365d_02.json"

# Load carbon data
with open(carbon_file) as f:
    carbon_data = json.load(f)["data"]
dfc = pd.DataFrame(carbon_data)
dfc['datetime'] = pd.to_datetime(dfc['datetime'])
dfc = dfc.set_index('datetime').sort_index()

# Load power data (optional)
with open(power_file) as f:
    power_data = json.load(f)["data"]
dfp = pd.DataFrame(power_data)
if not dfp.empty and "datetime" in dfp:
    dfp['datetime'] = pd.to_datetime(dfp['datetime'])
    dfp = dfp.set_index('datetime').sort_index()

plt.figure(figsize=(14, 7))

# Time-series plot: Carbon Intensity
plt.subplot(2, 1, 1)
dfc['carbonIntensity'].plot(alpha=0.7, lw=1)
plt.title(f"{ZONE}: Hourly Carbon Intensity (gCO₂/kWh)")
plt.ylabel("gCO₂/kWh")
plt.grid(axis='x', alpha=0.3)

# Boxplot by hour of day (diurnal pattern)
plt.subplot(2, 2, 3)
dfc['hour'] = dfc.index.hour
dfc.boxplot(column="carbonIntensity", by="hour", grid=False, showfliers=False, whis=[10,90])
plt.title(f"{ZONE}: Carbon Intensity by Hour of Day")
plt.suptitle("")
plt.xlabel("Hour")
plt.ylabel("gCO₂/kWh")

# If renewables present, plot seasonality
if not dfp.empty and "renewablePercentage" in dfp:
    plt.subplot(2, 2, 4)
    dfp['month'] = dfp.index.month
    dfp.boxplot(column="renewablePercentage", by="month", grid=False, showfliers=False, whis=[10,90])
    plt.title(f"{ZONE}: Renewable % by Month")
    plt.suptitle("")
    plt.xlabel("Month")
    plt.ylabel("Renewable %")

plt.tight_layout()
plt.show()
