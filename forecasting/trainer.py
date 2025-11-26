# forecasting/trainer.py
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from forecasting.models import build_model
from sklearn.metrics import mean_absolute_percentage_error

PROCESSED = Path("data/processed")
MODELS = Path("data/models")
SCALERS = Path("data/scalers")
PLOTS = Path("data/plots")

MODELS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

def create_sequences(data, lookback=24, forecast=24):
    """Create sliding window sequences for training"""
    X, y = [], []
    for i in range(lookback, len(data) - forecast + 1):
        X.append(data[i-lookback:i])
        y.append(data[i:i+forecast, 0])  # Only predict carbon (first column)
    return np.array(X), np.array(y)

def train_zone(zone, lookback_hours=24, forecast_hours=24):
    """Train improved model for a zone"""
    
    print(f"\n{'='*60}")
    print(f"🎯 Training {zone}")
    print(f"{'='*60}")
    
    # Load scaled data
    scaled_file = PROCESSED / f"{zone}_scaled.csv"
    if not scaled_file.exists():
        print(f"❌ Scaled file not found for {zone}")
        return None
    
    df = pd.read_csv(scaled_file, index_col=0, parse_dates=True)
    
    # Select features for training
    feature_columns = ["carbon"]
    if "fossilFree" in df.columns:
        feature_columns.append("fossilFree")
    if "renewable" in df.columns:
        feature_columns.append("renewable")
    
    # Add time features
    feature_columns.extend(["hour_sin", "hour_cos", "is_weekend"])
    
    print(f"📊 Features: {feature_columns}")
    print(f"📊 Total records: {len(df)}")
    
    # Prepare data array
    data = df[feature_columns].values
    n_features = data.shape[1]
    
    # Create sequences
    X, y = create_sequences(data, lookback=lookback_hours, forecast=forecast_hours)
    
    print(f"📊 Sequences created: {len(X)}")
    print(f"📊 Input shape: {X.shape}, Output shape: {y.shape}")
    
    # Train/Val/Test split (50/25/25)
    total = len(X)
    train_size = int(0.5 * total)
    val_size = int(0.25 * total)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Build model
    model = build_model(lookback_hours=lookback_hours, n_features=n_features)
    
    print(f"\n🏗️ Model architecture:")
    model.summary()
    
    # Callbacks
    model_path = MODELS / f"{zone}.keras"  # Use .keras format instead of .h5
    
    callbacks = [
        ModelCheckpoint(
            str(model_path),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train
    print(f"\n🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,  # Increased from 50
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\n📈 Evaluating on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_mae, test_mape = test_results
    
    # Make predictions for detailed analysis
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate MAPE manually for verification
    # mape_manual = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # print(f"\n✅ {zone} Training Complete!")
    # print(f"   Test Loss (MSE): {test_loss:.2f}")
    # print(f"   Test MAE: {test_mae:.2f}")
    # print(f"   Test MAPE: {test_mape:.2f}%")
    # print(f"   Manual MAPE: {mape_manual:.2f}%")

    def calculate_robust_mape(y_true, y_pred, epsilon=1e-10):
        """Calculate MAPE with protection against division by zero"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Filter out near-zero values
        mask = np.abs(y_true_flat) > epsilon
        y_true_filtered = y_true_flat[mask]
        y_pred_filtered = y_pred_flat[mask]
        
        if len(y_true_filtered) == 0:
            return np.nan
        
        mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        return mape

    # Use robust MAPE
    mape_manual = calculate_robust_mape(y_test, y_pred)

    print(f"\n {zone} Training Complete!")
    print(f"   Test Loss (MSE): {test_loss:.4f}")
    print(f"   Test MAE: {test_mae:.4f}")
    print(f"   Test MAPE: {mape_manual:.2f}%")
        
    # Plot training history
    plot_training_history(history, zone)
    
    # Plot sample predictions
    plot_predictions(y_test, y_pred, zone)
    
    # Save metrics
    metrics = {
        'zone': zone,
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'test_mape': float(test_mape),
        'n_features': n_features,
        'features': feature_columns
    }
    
    return metrics

def plot_training_history(history, zone):
    """Plot training and validation loss"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{zone} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title(f'{zone} - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOTS / f"{zone}_training_history.png", dpi=150)
    plt.close()
    print(f"📊 Saved training plot: {PLOTS}/{zone}_training_history.png")

def plot_predictions(y_true, y_pred, zone, n_samples=5):
    """Plot sample predictions vs actual"""
    plt.figure(figsize=(15, 8))
    
    for i in range(min(n_samples, len(y_true))):
        plt.subplot(n_samples, 1, i+1)
        hours = range(24)
        plt.plot(hours, y_true[i], label='Actual', marker='o', linewidth=2)
        plt.plot(hours, y_pred[i], label='Predicted', marker='x', linewidth=2, linestyle='--')
        plt.title(f'Sample {i+1}')
        plt.xlabel('Hour')
        plt.ylabel('Scaled Carbon Intensity')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'{zone} - Sample Predictions vs Actual', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(PLOTS / f"{zone}_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved predictions plot: {PLOTS}/{zone}_predictions.png")

if __name__ == "__main__":
    zones = [f.stem.replace("_scaled", "") for f in PROCESSED.glob("*_scaled.csv")]
    
    print(f"🚀 Training models for {len(zones)} zones with improved architecture")
    
    all_metrics = []
    
    for zone in sorted(zones):
        try:
            metrics = train_zone(zone, lookback_hours=24, forecast_hours=24)
            if metrics:
                all_metrics.append(metrics)
        except Exception as e:
            print(f"❌ Error training {zone}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎉 Training Complete!")
    print(f"{'='*60}")
    
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        print(f"\n📊 Overall Results:")
        print(df_metrics[['zone', 'test_mae', 'test_mape', 'n_features']])
        
        print(f"\n📊 Average MAPE: {df_metrics['test_mape'].mean():.2f}%")
        print(f"📊 Best zone: {df_metrics.loc[df_metrics['test_mape'].idxmin(), 'zone']} ({df_metrics['test_mape'].min():.2f}% MAPE)")
        
        # Save summary
        df_metrics.to_csv(MODELS / "training_summary.csv", index=False)
        print(f"\n✅ Saved training summary to {MODELS}/training_summary.csv")
