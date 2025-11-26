# Check for data quality issues
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv("data/processed/SG_processed.csv")

print("Carbon statistics:")
print(f"  Min: {df['carbon'].min()}")
print(f"  Max: {df['carbon'].max()}")
print(f"  Mean: {df['carbon'].mean()}")
print(f"  Std: {df['carbon'].std()}")

# Check for outliers
print(f"\nValues < 0.01: {(df['carbon'] < 0.01).sum()}")
print(f"Values > 0.95: {(df['carbon'] > 0.95).sum()}")

# Plot distribution
import matplotlib.pyplot as plt
plt.hist(df['carbon'], bins=50)
plt.title("SG Carbon Distribution")
plt.savefig("data/plots/SG_distribution.png")
plt.show()
