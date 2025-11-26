import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
import pickle
import matplotlib.pyplot as plt

PROCESSED = Path("forecasting/data/processed_365d")
MODELS = Path("forecasting/data/models_365d")
PLOTS = Path("forecasting/data/plots_365d")
SCALERS = Path("forecasting/data/scalers_365d")

MODELS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
LOOKBACK = 24
FORECAST = 24
BATCH_SIZE = 16
EPOCHS = 50

def robust_mape(y_true, y_pred, epsilon=1e-3):
    """
    Calculate MAPE while ignoring near-zero values.
    
    Why: After normalization (0-1), some values are very small.
         Dividing by small values creates huge percentage errors.
         This robust version ignores those problematic cases.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Only calculate MAPE for values > epsilon
    mask = np.abs(y_true_flat) > epsilon
    
    if mask.sum() == 0:
        return np.nan  # No valid values
    
    y_true_masked = y_true_flat[mask]
    y_pred_masked = y_pred_flat[mask]
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
    
    return mape

def create_sequences(data, lookback=24, forecast=24):
    """Create windowed sequences for training"""
    X, y = [], []
    for i in range(lookback, len(data) - forecast + 1):
        X.append(data[i-lookback:i])
        y.append(data[i:i+forecast, 0])
    return np.array(X), np.array(y)

def build_cnn_lstm_model(n_features, lookback):
    """Build CNN-LSTM architecture"""
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        
        layers.Conv1D(filters=16, kernel_size=4, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(filters=32, kernel_size=4, activation='relu'),
        
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        
        layers.Dense(lookback, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

print("🚀 TRAINING ML FORECASTING MODELS (with Robust MAPE)")
print("="*70)

results = []

for zone in zones:
    print(f"\n🎯 Training {zone}...")
    
    # Load processed data
    df = pd.read_csv(PROCESSED / f"{zone}_processed.csv")
    data = df.drop('datetime', axis=1).values
    
    n_features = data.shape[1]
    print(f"   📊 Features: {n_features}, Data points: {len(df)}")
    
    # Create sequences
    X, y = create_sequences(data, LOOKBACK, FORECAST)
    print(f"   📊 Training sequences: {len(X)}")
    
    # Train/Val/Test split: 50/25/25
    total = len(X)
    train_idx = int(0.5 * total)
    val_idx = int(0.75 * total)
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    # Build and train model
    model = build_cnn_lstm_model(n_features, LOOKBACK)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODELS / f"{zone}_best.keras",
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=0
        ),
        keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor='val_loss',
            verbose=0
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )
    
    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Make predictions for robust MAPE
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate robust MAPE
    test_mape_robust = robust_mape(y_test, y_pred)
    
    # Also calculate MAE in original scale
    with open(SCALERS / f"{zone}_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    y_test_original = scaler.inverse_transform(y_test.flatten().reshape(-1, 1)).flatten()
    y_pred_original = scaler.inverse_transform(y_pred.flatten().reshape(-1, 1)).flatten()
    
    mae_original = np.mean(np.abs(y_test_original - y_pred_original))
    
    print(f"   ✅ Test MAE (scaled): {test_mae:.4f}")
    print(f"   ✅ Test MAE (original): {mae_original:.2f} gCO₂/kWh")
    print(f"   ✅ Test MAPE (robust): {test_mape_robust:.2f}%")
    
    results.append({
        'Zone': zone,
        'MAE_Scaled': test_mae,
        'MAE_gCO2_kWh': mae_original,
        'MAPE_Robust': test_mape_robust,
        'Train_Size': len(X_train),
        'Val_Size': len(X_val),
        'Test_Size': len(X_test)
    })

# Summary
print("\n" + "="*70)
print("📊 FINAL TRAINING RESULTS (with Robust Metrics):\n")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print(f"\n✅ Average MAPE: {results_df['MAPE_Robust'].mean():.2f}%")
print(f"✅ Best zone: {results_df.loc[results_df['MAPE_Robust'].idxmin(), 'Zone']} ({results_df['MAPE_Robust'].min():.2f}%)")

# Save results
results_df.to_csv(MODELS / "training_results_final.csv", index=False)

print(f"\n✅ Models saved in: {MODELS}")
print(f"✅ Results saved: {MODELS}/training_results_final.csv")

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: MAPE comparison
axes[0].bar(results_df['Zone'], results_df['MAPE_Robust'], color='steelblue', alpha=0.7)
axes[0].axhline(y=30, color='red', linestyle='--', label='Target: <30%', linewidth=2)
axes[0].set_title('Model Accuracy (Robust MAPE %)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('MAPE (%)')
axes[0].set_ylim(0, 100)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: MAE in original units
axes[1].bar(results_df['Zone'], results_df['MAE_gCO2_kWh'], color='orange', alpha=0.7)
axes[1].set_title('Prediction Error (Original Units)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('MAE (gCO₂/kWh)')
axes[1].grid(True, alpha=0.3, axis='y')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(PLOTS / 'model_performance_summary_365d.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Saved: data/plots_365d/model_performance_summary_365d.png")
plt.close()

print("\n✅ Training complete! Ready for predictions.")
