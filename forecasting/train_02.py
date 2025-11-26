"""
TRAIN MODELS ON 365-DAY DATA (VERSION 02)
=========================================
- Uses processed_365d_02 features (14 columns + datetime)
- CNN + Bidirectional LSTM architecture
- Lookback: 24 hours
- Forecast horizon: 24 hours (next day carbon)

Inputs:  processed_365d_02/{ZONE}_processed_02.csv
Outputs: models_365d_02/{ZONE}_model_02.keras
         models_365d_02/training_results_02.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

LOOKBACK = 24      # past hours
HORIZON  = 24      # future hours to predict
BATCH_SIZE = 32
EPOCHS = 150

BASE_DIR   = Path(__file__).parent
PROCESSED  = BASE_DIR /"forecasting" / "data" / "processed_365d_02"
MODELS_DIR = BASE_DIR / "forecasting" / "data" / "models_365d_02"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]

print("Processed dir:", PROCESSED)
print("Models dir   :", MODELS_DIR)

# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------
def create_sequences(data, lookback=24, horizon=24):
    """
    data: np.array (timesteps, features)
    returns:
       X: (samples, lookback, features)
       y: (samples, horizon)  -> carbon_norm only (first feature)
    """
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback : i, :])
        y.append(data[i : i + horizon, 0])    # 0 = carbon_norm
    return np.array(X), np.array(y)

def train_ensemble(X_train, y_train, X_val, y_val, X_test, y_test, n_features, n_models=3):
    """Train 3 models and return ensemble predictions"""
    models = []
    
    for m in range(n_models):
        model = build_cnn_bilstm_model_v2(n_features, LOOKBACK, HORIZON)
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
        ]
        
        model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=callbacks)
        
        models.append(model)
    
    # Average predictions
    preds = [m.predict(X_test, verbose=0) for m in models]
    y_pred_ensemble = np.mean(preds, axis=0)
    
    return y_pred_ensemble, models

def build_cnn_bilstm_model_v2(n_features, lookback, horizon):
    """Larger model with more filters and LSTM units"""
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        
        # More CNN layers
        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        # Larger BiLSTM
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        
        # More dense layers
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(horizon, activation="linear"),
    ])

    model.compile(
        # optimizer=keras.optimizers.Adam(learning_rate=5e-4),  # Lower LR for stability
        # loss="mse",
        # metrics=["mae"]
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=keras.losses.Huber(delta=0.5),  # Huber is more robust than MSE
        metrics=["mae"]
    )
    return model








def build_cnn_bilstm_model(n_features, lookback, horizon):
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.Conv1D(32, 3, activation="relu", padding="same"),
        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.2),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(horizon, activation="linear"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )
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


# -------------------------------------------------------------------
# main loop
# -------------------------------------------------------------------
results = []

print("\n==============================================")
print("TRAINING ON 365-DAY, 14-FEATURE DATA (V02)")
print("==============================================\n")

for zone in zones:
    path = PROCESSED / f"{zone}_processed_02.csv"
    if not path.exists():
        print(f"{zone}: processed file missing, skipping")
        continue

    print(f"\n---------- {zone} ----------")

    df = pd.read_csv(path)
    df = df.sort_values("datetime")

    # drop datetime, keep features only
    X_all = df.drop(columns=["datetime"]).values
    n_features = X_all.shape[1]
    print(f"  rows={len(df)}, features={n_features}")

    # build sequences
    X, y = create_sequences(X_all, LOOKBACK, HORIZON)
    print(f"  sequences={len(X)} (X shape {X.shape}, y shape {y.shape})")

    if len(X) < 500:
        print("  too few sequences (<500), skipping")
        continue

    # train/val/test split: 60/20/20
    n = len(X)
    train_end = int(0.6 * n)
    val_end   = int(0.8 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    print(f"  split -> train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # model = build_cnn_bilstm_model(n_features, LOOKBACK, HORIZON)
    model = build_cnn_bilstm_model_v2(n_features, LOOKBACK, HORIZON)  # Larger model

    print(f"  params={model.count_params():,}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        ),
        keras.callbacks.ModelCheckpoint(
            MODELS_DIR / f"{zone}_model_02.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks
    )

    best_epochs = len(history.history["loss"])
    print(f"  trained epochs={best_epochs}")

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    mape = robust_mape(y_test, y_pred)

    print(f"  test_loss={test_loss:.4f}, test_MAE(norm)={test_mae:.4f}, robust_MAPE={mape:.2f}%")

    results.append({
        "zone": zone,
        "rows": len(df),
        "seq": len(X),
        "features": n_features,
        "test_loss": test_loss,
        "test_mae_norm": test_mae,
        "robust_mape": mape,
        "epochs": best_epochs,
        "params": model.count_params()
    })

# -------------------------------------------------------------------
# save results
# -------------------------------------------------------------------
if results:
    res_df = pd.DataFrame(results)
    out_csv = MODELS_DIR / "training_results_02.csv"
    res_df.to_csv(out_csv, index=False)
    print("\n==============================================")
    print("TRAINING COMPLETE")
    print("==============================================")
    print(res_df[["zone","features","test_mae_norm","robust_mape","epochs"]])
    print(f"\nSaved metrics to: {out_csv}")
else:
    print("No models were trained (no results).")
