# train_03.py
"""TRAIN ENSEMBLE OF 3 MODELS (VERSION 03)
========================================
- 48-hour lookback (not 24)
- 3 different architectures averaged:
  1. CNN + BiLSTM (current best)
  2. GRU-based (simpler, faster)
  3. Attention mechanism (captures dependencies)
- Per-zone training (customizable)
- Saves best ensemble for each zone
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

LOOKBACK = 48      # Extended to 48 hours
HORIZON  = 24
BATCH_SIZE = 32
EPOCHS = 200

BASE_DIR   = Path(__file__).parent
PROCESSED  = BASE_DIR / "data" / "processed_365d_03"
MODELS_DIR = BASE_DIR / "data" / "models_365d_03"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]

print("="*70)
print("TRAINING ENSEMBLE (48H LOOKBACK, 3 ARCHITECTURES)")
print("="*70)

def create_sequences(data, lookback=48, horizon=24):
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback : i, :])
        y.append(data[i : i + horizon, 0])
    return np.array(X), np.array(y)

def build_cnn_bilstm(n_features, lookback, horizon):
    """Architecture 1: CNN + BiLSTM"""
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(horizon, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(delta=0.5), metrics=["mae"])
    return model

def build_gru_model(n_features, lookback, horizon):
    """Architecture 2: GRU-based (faster, captures sequences well)"""
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.Bidirectional(layers.GRU(100, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.GRU(50)),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(horizon, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(delta=0.5), metrics=["mae"])
    return model

def build_attention_model(n_features, lookback, horizon):
    """Architecture 3: Multi-head attention"""
    inputs = layers.Input(shape=(lookback, n_features))
    
    # Attention
    x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    x = layers.LayerNormalization()(x + inputs)
    
    # LSTM after attention
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(horizon, activation="linear")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(delta=0.5), metrics=["mae"])
    return model

def robust_mape(y_true, y_pred, eps=1e-3):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

results = []

for zone in zones:
    path = PROCESSED / f"{zone}_processed_03.csv"
    if not path.exists():
        print(f"\n{zone}: file missing, skipping")
        continue

    print(f"\n---------- {zone} ----------")

    df = pd.read_csv(path).sort_values("datetime")
    X_all = df.drop(columns=["datetime"]).values
    n_features = X_all.shape[1]
    print(f"  rows={len(df)}, features={n_features}")

    X, y = create_sequences(X_all, LOOKBACK, HORIZON)
    print(f"  sequences={len(X)}")

    if len(X) < 500:
        print(f"  too few sequences, skipping")
        continue

    n = len(X)
    train_end = int(0.6 * n)
    val_end   = int(0.8 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    print(f"  split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Build 3 models
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=0),
    ]

    print("  Training 3 models...")
    models = []
    for i, (name, builder) in enumerate([
        ("CNN-BiLSTM", build_cnn_bilstm),
        ("GRU", build_gru_model),
        ("Attention", build_attention_model),
    ]):
        print(f"    Model {i+1}/3: {name}...", end="", flush=True)
        m = builder(n_features, LOOKBACK, HORIZON)
        m.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=callbacks)
        models.append(m)
        print(" ✓")

    # Ensemble prediction
    preds = np.array([m.predict(X_test, verbose=0) for m in models])
    y_pred_ensemble = np.mean(preds, axis=0)

    test_loss_ensemble = np.mean(np.abs(y_test.flatten() - y_pred_ensemble.flatten()))
    mape = robust_mape(y_test, y_pred_ensemble)

    print(f"  Ensemble MAE (norm): {test_loss_ensemble:.4f}, MAPE: {mape:.2f}%")

    # Save ensemble
    for i, m in enumerate(models):
        m.save(MODELS_DIR / f"{zone}_model_{i+1}_03.keras")

    results.append({
        "zone": zone,
        "rows": len(df),
        "seq": len(X),
        "features": n_features,
        "ensemble_mae": test_loss_ensemble,
        "robust_mape": mape,
    })

if results:
    res_df = pd.DataFrame(results)
    out_csv = MODELS_DIR / "training_results_03.csv"
    res_df.to_csv(out_csv, index=False)
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(res_df)
    print(f"\nSaved to: {out_csv}")
else:
    print("No models trained.")
