"""
EVALUATE ENSEMBLE MODELS (VERSION 03)
=====================================
- Load trained ensemble models
- Generate predictions on test set
- Convert to real units (gCO2/kWh)
- Calculate actual MAPE and MAE
- Plot results for thesis/presentation
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras

LOOKBACK = 48
HORIZON = 24

BASE_DIR   = Path(__file__).parent
PROCESSED  = BASE_DIR / "data" / "processed_365d_03"
MODELS_DIR = BASE_DIR / "data" / "models_365d_03"
SCALERS    = BASE_DIR / "data" / "scalers_365d_03"
PLOTS_DIR  = BASE_DIR / "plots_03"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]

def create_sequences(data, lookback=48, horizon=24):
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback : i, :])
        y.append(data[i : i + horizon, 0])
    return np.array(X), np.array(y)

def robust_mape(y_true, y_pred, eps=1e-3):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

print("="*70)
print("EVALUATING ENSEMBLE MODELS (VERSION 03)")
print("="*70)

eval_results = []

for zone in zones:
    print(f"\n---------- {zone} ----------")

    # Load data
    df = pd.read_csv(PROCESSED / f"{zone}_processed_03.csv").sort_values("datetime")
    X_all = df.drop(columns=["datetime"]).values
    datetimes = df["datetime"].values

    X, y = create_sequences(X_all, LOOKBACK, HORIZON)

    # Test set (same split as training)
    n = len(X)
    train_end = int(0.6 * n)
    val_end   = int(0.8 * n)
    X_test, y_test = X[val_end:], y[val_end:]
    test_datetimes = datetimes[LOOKBACK + val_end + HORIZON : LOOKBACK + val_end + HORIZON + len(y_test)]

    print(f"  Test samples: {len(X_test)}")

    # Load 3 models and ensemble
    models = []
    for i in range(1, 4):
        m = keras.models.load_model(MODELS_DIR / f"{zone}_model_{i}_03.keras")
        models.append(m)

    preds = np.array([m.predict(X_test, verbose=0) for m in models])
    y_pred_ensemble = np.mean(preds, axis=0)

    # Load scaler and convert to real units
    with open(SCALERS / f"{zone}_scalers_03.pkl", "rb") as f:
        scalers = pickle.load(f)
    
    carbon_scaler = scalers["carbon"]

    # Inverse transform
    y_test_flat = y_test.flatten().reshape(-1, 1)
    y_pred_flat = y_pred_ensemble.flatten().reshape(-1, 1)

    y_test_real = carbon_scaler.inverse_transform(y_test_flat).flatten()
    y_pred_real = carbon_scaler.inverse_transform(y_pred_flat).flatten()

    # Metrics in real units
    mae_real = np.mean(np.abs(y_test_real - y_pred_real))
    rmse_real = np.sqrt(np.mean((y_test_real - y_pred_real) ** 2))
    mape_real = robust_mape(y_test_real, y_pred_real)

    print(f"  MAE (real): {mae_real:.2f} gCO2/kWh")
    print(f"  RMSE (real): {rmse_real:.2f} gCO2/kWh")
    print(f"  MAPE (real): {mape_real:.2f}%")
    print(f"  Carbon range: {y_test_real.min():.1f} - {y_test_real.max():.1f} gCO2/kWh")

    eval_results.append({
        "zone": zone,
        "test_samples": len(X_test),
        "mae_gco2_kwh": mae_real,
        "rmse_gco2_kwh": rmse_real,
        "mape_percent": mape_real,
        "carbon_min": y_test_real.min(),
        "carbon_max": y_test_real.max(),
        "carbon_mean": y_test_real.mean(),
    })

    # ===== PLOTTING =====
    # Plot 1: Full test set (forecast vs actual)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Time axis
    hours = np.arange(len(y_test_real))

    # First 7 days detail
    axes[0].plot(hours[:168], y_test_real[:168], "o-", alpha=0.7, label="Actual", linewidth=2)
    axes[0].plot(hours[:168], y_pred_real[:168], "s--", alpha=0.7, label="Predicted", linewidth=2)
    axes[0].set_title(f"{zone}: First 7 Days (Hourly Carbon Intensity)")
    axes[0].set_ylabel("gCO₂/kWh")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Full test period
    axes[1].plot(hours, y_test_real, alpha=0.6, label="Actual", linewidth=1)
    axes[1].plot(hours, y_pred_real, alpha=0.6, label="Predicted", linewidth=1)
    axes[1].set_title(f"{zone}: Full Test Period (~5 months)")
    axes[1].set_xlabel("Hours")
    axes[1].set_ylabel("gCO₂/kWh")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{zone}_forecast_vs_actual_03.png", dpi=150)
    plt.close()

    # Plot 2: Error distribution
    errors = np.abs(y_test_real - y_pred_real)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(errors, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_title(f"{zone}: Absolute Error Distribution")
    axes[0].set_xlabel("Absolute Error (gCO₂/kWh)")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(mae_real, color="red", linestyle="--", label=f"MAE={mae_real:.2f}")
    axes[0].legend()

    axes[1].scatter(y_test_real, y_pred_real, alpha=0.5, s=10)
    axes[1].plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()],
                 "r--", label="Perfect Prediction")
    axes[1].set_title(f"{zone}: Actual vs Predicted")
    axes[1].set_xlabel("Actual (gCO₂/kWh)")
    axes[1].set_ylabel("Predicted (gCO₂/kWh)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{zone}_error_analysis_03.png", dpi=150)
    plt.close()

    print(f"  Plots saved to: plots_03/")

# Save evaluation results
eval_df = pd.DataFrame(eval_results)
eval_csv = MODELS_DIR / "evaluation_results_03.csv"
eval_df.to_csv(eval_csv, index=False)

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE")
print("="*70)
print("\n📊 RESULTS (Real Units - gCO₂/kWh):")
print(eval_df[["zone", "mae_gco2_kwh", "rmse_gco2_kwh", "mape_percent"]])
print(f"\nSaved to: {eval_csv}")
print(f"Plots saved to: {PLOTS_DIR}")
