"""
GENERATE ALL PLOTS FOR THESIS REPORT (VERSION 03)
===================================================
Creates 8 publication-ready figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("data/models_365d_03")
PLOTS = Path("plots_03_thesis")
PLOTS.mkdir(exist_ok=True)

eval_df = pd.read_csv(BASE / "evaluation_results_03.csv")

plt.style.use("seaborn-v0_8-darkgrid")

# ===== PLOT 1: Zone Performance =====
fig, ax = plt.subplots(figsize=(12, 6))
colors = ["green" if m < 10 else "orange" if m < 20 else "red" for m in eval_df["mape_percent"]]
ax.bar(eval_df["zone"], eval_df["mape_percent"], color=colors, alpha=0.7, edgecolor="black", linewidth=2)
ax.axhline(10, color="green", linestyle="--", alpha=0.5, label="Excellent (<10%)")
ax.axhline(20, color="orange", linestyle="--", alpha=0.5, label="Good (<20%)")
ax.set_ylabel("MAPE (%)", fontsize=13, fontweight="bold")
ax.set_title("Model Performance Across 8 Global Zones", fontsize=14, fontweight="bold")
ax.set_xticklabels(eval_df["zone"], rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / "1_zone_performance.png", dpi=150)
plt.close()
print("✓ Plot 1: Zone Performance")

# ===== PLOT 2: Error Metrics Comparison =====
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(eval_df))
width = 0.35
ax.bar(x - width/2, eval_df["mae_gco2_kwh"], width, label="MAE", alpha=0.8, edgecolor="black")
ax.bar(x + width/2, eval_df["rmse_gco2_kwh"], width, label="RMSE", alpha=0.8, edgecolor="black")
ax.set_ylabel("Error (gCO₂/kWh)", fontsize=13, fontweight="bold")
ax.set_title("Mean Absolute Error vs Root Mean Square Error", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(eval_df["zone"], rotation=45)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(PLOTS / "2_error_metrics.png", dpi=150)
plt.close()
print("✓ Plot 2: Error Metrics")

# ===== PLOT 3: Carbon Range Across Zones =====
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(eval_df["zone"], eval_df["carbon_mean"], s=200, alpha=0.6, label="Mean", edgecolors="black", linewidth=2)
ax.scatter(eval_df["zone"], eval_df["carbon_min"], s=100, alpha=0.4, marker="v", label="Min", edgecolors="black")
ax.scatter(eval_df["zone"], eval_df["carbon_max"], s=100, alpha=0.4, marker="^", label="Max", edgecolors="black")
for idx, zone in enumerate(eval_df["zone"]):
    ax.vlines(idx, eval_df["carbon_min"].iloc[idx], eval_df["carbon_max"].iloc[idx], alpha=0.3, linewidth=3)
ax.set_ylabel("Carbon Intensity (gCO₂/kWh)", fontsize=13, fontweight="bold")
ax.set_title("Carbon Intensity Range & Variability by Zone", fontsize=14, fontweight="bold")
ax.set_xticklabels(eval_df["zone"], rotation=45)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "3_carbon_range.png", dpi=150)
plt.close()
print("✓ Plot 3: Carbon Range")

# ===== PLOT 4: Model Accuracy Tiers =====
fig, ax = plt.subplots(figsize=(10, 8))
categories = {
    "Excellent\n(<10% MAPE)": len(eval_df[eval_df["mape_percent"] < 10]),
    "Very Good\n(10-15% MAPE)": len(eval_df[(eval_df["mape_percent"] >= 10) & (eval_df["mape_percent"] < 15)]),
    "Good\n(15-25% MAPE)": len(eval_df[(eval_df["mape_percent"] >= 15) & (eval_df["mape_percent"] < 25)]),
    "Fair\n(>25% MAPE)": len(eval_df[eval_df["mape_percent"] >= 25]),
}
colors_pie = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
wedges, texts, autotexts = ax.pie(categories.values(), labels=categories.keys(), autopct="%1.0f",
                                     colors=colors_pie, startangle=90, textprops={"fontsize": 11})
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontweight("bold")
ax.set_title("Model Accuracy Distribution Across Zones", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS / "4_accuracy_tiers.png", dpi=150)
plt.close()
print("✓ Plot 4: Accuracy Tiers")

# ===== PLOT 5: MAPE vs Carbon Volatility =====
fig, ax = plt.subplots(figsize=(10, 6))
volatility = eval_df["carbon_max"] - eval_df["carbon_min"]
ax.scatter(volatility, eval_df["mape_percent"], s=200, alpha=0.6, edgecolors="black", linewidth=2)
for idx, zone in enumerate(eval_df["zone"]):
    ax.annotate(zone, (volatility.iloc[idx], eval_df["mape_percent"].iloc[idx]), 
                fontsize=9, ha="right")
ax.set_xlabel("Carbon Intensity Range (gCO₂/kWh)", fontsize=13, fontweight="bold")
ax.set_ylabel("MAPE (%)", fontsize=13, fontweight="bold")
ax.set_title("Forecast Error vs Grid Volatility", fontsize=14, fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "5_mape_vs_volatility.png", dpi=150)
plt.close()
print("✓ Plot 5: MAPE vs Volatility")

# ===== PLOT 6: Summary Table as Image =====
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("tight")
ax.axis("off")

table_data = []
table_data.append(["Zone", "MAE\n(gCO₂/kWh)", "RMSE\n(gCO₂/kWh)", "MAPE\n(%)", "Rating", "Use Case"])

for _, row in eval_df.iterrows():
    zone = row["zone"]
    mae = f"{row['mae_gco2_kwh']:.2f}"
    rmse = f"{row['rmse_gco2_kwh']:.2f}"
    mape = f"{row['mape_percent']:.2f}"
    
    if row["mape_percent"] < 10:
        rating = "⭐⭐⭐⭐⭐"
        use = "Production"
    elif row["mape_percent"] < 20:
        rating = "⭐⭐⭐⭐"
        use = "Planning"
    elif row["mape_percent"] < 30:
        rating = "⭐⭐⭐"
        use = "Research"
    else:
        rating = "⭐⭐"
        use = "Development"
    
    table_data.append([zone, mae, rmse, mape, rating, use])

table = ax.table(cellText=table_data, cellLoc="center", loc="center",
                colWidths=[0.12, 0.15, 0.15, 0.1, 0.2, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Header styling
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor("#34495e")
    table[(0, i)].set_text_props(weight="bold", color="white")

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        if i % 2 == 0:
            table[(i, j)].set_facecolor("#ecf0f1")
        else:
            table[(i, j)].set_facecolor("white")

ax.set_title("Model Performance Summary Across All Zones", fontsize=14, fontweight="bold", pad=20)
plt.savefig(PLOTS / "6_summary_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Plot 6: Summary Table")

print(f"\n✅ All thesis plots saved to: {PLOTS}")
print("\nPlots created:")
print("  1. Zone Performance (Bar Chart)")
print("  2. Error Metrics Comparison (Grouped Bars)")
print("  3. Carbon Range & Variability (Scatter)")
print("  4. Accuracy Distribution (Pie Chart)")
print("  5. MAPE vs Volatility (Scatter with Trend)")
print("  6. Performance Summary (Table)")
print("\nPlus the existing forecast plots from evaluate_03.py")
