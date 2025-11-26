"""
IMPROVED MODEL TRAINING WITH POWER DATA (Version 01)
====================================================
Uses 28 features (up from 6 original) including power breakdown data.

ARCHITECTURE IMPROVEMENTS:
1. Bidirectional LSTM (sees past + future context)
2. Batch Normalization (stable training)
3. Huber loss (robust to outliers)
4. Learning rate scheduling (adaptive optimization)
5. Increased epochs with early stopping

FILES READ: {ZONE}_processed_01.csv, {ZONE}_scaler_01.pkl
FILES SAVED: {ZONE}_model_01.keras, training_results_01.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SMART PATH DETECTION
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()

if SCRIPT_DIR.name == 'forecasting':
    PROJECT_ROOT = SCRIPT_DIR.parent
    PROCESSED = SCRIPT_DIR / "data" / "processed_365d_with_power"
    MODELS = SCRIPT_DIR / "data" / "models_365d_with_power"
    PLOTS = SCRIPT_DIR / "data" / "plots_365d_with_power"
    SCALERS = SCRIPT_DIR / "data" / "scalers_365d_with_power"
else:
    PROJECT_ROOT = SCRIPT_DIR
    PROCESSED = PROJECT_ROOT / "forecasting" / "data" / "processed_365d_with_power"
    MODELS = PROJECT_ROOT / "forecasting" / "data" / "models_365d_with_power"
    PLOTS = PROJECT_ROOT / "forecasting" / "data" / "plots_365d_with_power"
    SCALERS = PROJECT_ROOT / "forecasting" / "data" / "scalers_365d_with_power"

MODELS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

print("🔍 PATH DETECTION")
print(f"   Processed data: {PROCESSED}")
print(f"   Models output: {MODELS}")
print(f"   Plots output: {PLOTS}")
print()

zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
LOOKBACK = 24      # Use past 24 hours
FORECAST = 24      # Predict next 24 hours
BATCH_SIZE = 32
EPOCHS = 150       # Increased from 50

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def robust_mape(y_true, y_pred, epsilon=1e-3):
    """
    Calculate MAPE while ignoring near-zero values
    
    Why: Normalized values near 0 cause huge percentage errors
    Solution: Only calculate MAPE for values > epsilon
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    mask = np.abs(y_true_flat) > epsilon
    
    if mask.sum() == 0:
        return np.nan
    
    y_true_masked = y_true_flat[mask]
    y_pred_masked = y_pred_flat[mask]
    mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
    
    return mape

def create_sequences(data, lookback=24, forecast=24):
    """
    Create sliding window sequences for time series
    
    Args:
        data: Array (timesteps, features)
        lookback: How many past hours to use
        forecast: How many future hours to predict
    
    Returns:
        X: (samples, lookback, features) - Input sequences
        y: (samples, forecast) - Target sequences (carbon only)
    
    Example:
        If lookback=24, forecast=24:
        X[0] = hours 0-23 (all 28 features)
        y[0] = hours 24-47 (carbon values only)
    """
    X, y = [], []
    for i in range(lookback, len(data) - forecast + 1):
        X.append(data[i-lookback:i])      # Past 24 hours (all features)
        y.append(data[i:i+forecast, 0])   # Future 24 hours (carbon only, index 0)
    return np.array(X), np.array(y)

def build_improved_cnn_lstm(n_features, lookback):
    """
    IMPROVED CNN-LSTM ARCHITECTURE WITH POWER FEATURES
    ==================================================
    
    Key improvements:
    1. Handles 28 features (vs 6 original)
    2. Deeper CNN layers (32 → 64 → 128 filters)
    3. Batch Normalization after Conv layers
    4. Bidirectional LSTM (captures patterns both directions)
    5. More dropout (0.3) prevents overfitting
    6. Huber loss (robust to outliers)
    
    Architecture flow:
    ----------------
    Input (24 timesteps, 28 features)
        ↓
    Conv1D (32 filters, kernel=3) + BatchNorm
        ↓ Extracts local hourly patterns
    Conv1D (64 filters, kernel=3) + MaxPool + Dropout
        ↓ Deeper pattern extraction
    Conv1D (128 filters, kernel=3) + BatchNorm + MaxPool
        ↓ High-level features
    Bidirectional LSTM (64 units × 2 directions = 128 total)
        ↓ Learns long-term dependencies forward + backward
    Dropout (0.3)
        ↓
    Bidirectional LSTM (32 units × 2 directions = 64 total)
        ↓
    Dropout (0.2)
        ↓
    Dense (64) + BatchNorm + Dropout
        ↓
    Dense (32)
        ↓
    Dense (24) - Output: 24-hour forecast
    
    Why Bidirectional LSTM?
    - Forward pass: Learns patterns from past → future
    - Backward pass: Learns patterns from future → past
    - Combined: Better understanding of temporal context
    
    Why Huber Loss?
    - MSE: Sensitive to outliers (squared error)
    - MAE: Less sensitive but slower convergence
    - Huber: Best of both (MSE for small errors, MAE for large)
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(lookback, n_features)),
        
        # CNN Block 1: Extract local patterns (hour-to-hour changes)
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        # CNN Block 2: Deeper patterns (multi-hour trends)
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # CNN Block 3: High-level features (daily patterns)
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Bidirectional LSTM Layer 1: Long-term dependencies
        # return_sequences=True passes output to next LSTM layer
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.3),
        
        # Bidirectional LSTM Layer 2: Final sequence processing
        # return_sequences=False outputs single vector
        layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
        layers.Dropout(0.2),
        
        # Dense layers: Final prediction mapping
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        
        # Output: 24 future carbon values
        layers.Dense(lookback, activation='linear')
    ])
    
    # Compile with Huber loss
    # Huber delta=1.0: Acts like MSE for |error| < 1, MAE for |error| > 1
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Lower LR for stability
        loss='huber',
        metrics=['mae']
    )
    
    return model

# ============================================================
# MAIN TRAINING LOOP
# ============================================================

print("🚀 IMPROVED ML TRAINING WITH POWER DATA (28 FEATURES)")
print("="*70)

results = []

for zone in zones:
    print(f"\n🎯 Training {zone}...")
    
    # Check if processed file exists
    processed_file = PROCESSED / f"{zone}_processed_01.csv"
    if not processed_file.exists():
        print(f"   ⚠️ Processed file not found: {processed_file}")
        continue
    
    # Load processed data
    df = pd.read_csv(processed_file)
    data = df.drop('datetime', axis=1).values  # All columns except datetime
    
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
    
    print(f"   📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Build improved model
    model = build_improved_cnn_lstm(n_features, LOOKBACK)
    
    print(f"   🏗️ Model parameters: {model.count_params():,}")
    
    # Advanced callbacks
    callbacks = [
        # Save best model based on validation loss
        keras.callbacks.ModelCheckpoint(
            MODELS / f"{zone}_model_01.keras",
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=0
        ),
        
        # Stop if no improvement after 20 epochs
        keras.callbacks.EarlyStopping(
            patience=20,
            restore_best_weights=True,
            monitor='val_loss',
            verbose=0
        ),
        
        # Reduce learning rate if plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,      # Halve the learning rate
            patience=5,      # After 5 epochs without improvement
            min_lr=1e-7,     # Don't go below this
            verbose=0
        )
    ]
    
    # Train model
    print(f"   🔄 Training (max {EPOCHS} epochs, early stopping enabled)...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0  # Silent training, show only summary
    )
    
    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate robust MAPE
    test_mape_robust = robust_mape(y_test, y_pred)
    
    # Calculate MAE in original scale (gCO₂/kWh)
    scaler_file = SCALERS / f"{zone}_scaler_01.pkl"
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    y_test_original = scaler.inverse_transform(y_test.flatten().reshape(-1, 1)).flatten()
    y_pred_original = scaler.inverse_transform(y_pred.flatten().reshape(-1, 1)).flatten()
    
    mae_original = np.mean(np.abs(y_test_original - y_pred_original))
    
    # Get training history
    actual_epochs = len(history.history['loss'])
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"   ✅ Completed in {actual_epochs} epochs (stopped early)")
    print(f"   ✅ Test MAE (scaled): {test_mae:.4f}")
    print(f"   ✅ Test MAE (original): {mae_original:.2f} gCO₂/kWh")
    print(f"   ✅ Test MAPE (robust): {test_mape_robust:.2f}%")
    print(f"   ✅ Train/Val loss: {final_train_loss:.4f} / {final_val_loss:.4f}")
    
    results.append({
        'Zone': zone,
        'Features': n_features,
        'MAE_Scaled': test_mae,
        'MAE_gCO2_kWh': mae_original,
        'MAPE_Robust': test_mape_robust,
        'Train_Size': len(X_train),
        'Val_Size': len(X_val),
        'Test_Size': len(X_test),
        'Epochs_Trained': actual_epochs,
        'Final_Train_Loss': final_train_loss,
        'Final_Val_Loss': final_val_loss,
        'Model_Parameters': model.count_params()
    })

# ============================================================
# RESULTS SUMMARY
# ============================================================

print("\n" + "="*70)
print("📊 IMPROVED TRAINING RESULTS (WITH POWER DATA):\n")

results_df = pd.DataFrame(results)
print(results_df[['Zone', 'Features', 'MAE_gCO2_kWh', 'MAPE_Robust', 'Epochs_Trained']].to_string(index=False))

print(f"\n📈 PERFORMANCE SUMMARY:")
print(f"   Average MAPE: {results_df['MAPE_Robust'].mean():.2f}%")
print(f"   Average MAE: {results_df['MAE_gCO2_kWh'].mean():.2f} gCO₂/kWh")
print(f"   Best zone: {results_df.loc[results_df['MAPE_Robust'].idxmin(), 'Zone']} ({results_df['MAPE_Robust'].min():.2f}%)")
print(f"   Total model parameters: {results_df['Model_Parameters'].iloc[0]:,}")

# Save results with _01 suffix
results_file = MODELS / "training_results_01.csv"
results_df.to_csv(results_file, index=False)
print(f"\n✅ Results saved: {results_file}")

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: MAPE comparison
axes[0].bar(results_df['Zone'], results_df['MAPE_Robust'], color='steelblue', alpha=0.7)
axes[0].axhline(y=30, color='red', linestyle='--', label='Target: <30%', linewidth=2)
axes[0].set_title('Improved Model Accuracy (Robust MAPE %) - 28 Features', 
                  fontsize=12, fontweight='bold')
axes[0].set_ylabel('MAPE (%)')
axes[0].set_ylim(0, max(100, results_df['MAPE_Robust'].max() + 10))
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
plot_file = PLOTS / 'model_performance_01.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"📊 Performance plot saved: {plot_file}")
plt.close()

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
# print(f"\n📁 SAVED FILES (ALL WITH _01 SUFFIX):")
print(f"   Models: {MODELS}")
print(f"   Results: training_results_01.csv")
print(f"   Plot: model_performance_01.png")
print("\nMODEL FILES:")
for zone in zones:
    model_file = MODELS / f"{zone}_model_01.keras"
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"   {zone}_model_01.keras ({size_mb:.1f} MB)")

print("\n NEXT STEPS:")
print("   1. Review training_results_01.csv for detailed metrics")
print("   2. Check model_performance_01.png for visual comparison")
print("   3. Run predictions using {zone}_model_01.keras files")
print("   4. Compare with baseline (6 features) to see improvement")

print("\n EXPECTED IMPROVEMENTS:")
print("   • 30-50% MAE reduction vs baseline (6 features)")
print("   • Better accuracy on volatile grids (DE, IE)")
print("   • Stable performance across all regions")
