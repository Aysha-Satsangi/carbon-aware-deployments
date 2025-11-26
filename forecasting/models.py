# forecasting/models.py
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.regularizers import l2

def build_model(lookback_hours=24, n_features=1):
    """
    Build improved CNN-LSTM model for carbon forecasting
    
    Args:
        lookback_hours: Number of past hours to use for prediction
        n_features: Number of input features (1=carbon only, 3+=carbon+power breakdown)
    
    Returns:
        Compiled Keras model
    """
    
    model = Sequential([
        # First Conv1D layer - extract short-term patterns
        layers.Conv1D(
            filters=16, 
            kernel_size=4, 
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(0.001),
            input_shape=(lookback_hours, n_features)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Second Conv1D layer - deeper feature extraction
        layers.Conv1D(
            filters=32, 
            kernel_size=4, 
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(0.001)
        ),
        layers.BatchNormalization(),
        
        # LSTM layer - capture temporal dependencies
        layers.LSTM(
            64, 
            return_sequences=False,
            kernel_regularizer=l2(0.001)
        ),
        layers.Dropout(0.2),
        
        # Dense layers for final prediction
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(lookback_hours, activation='linear')  # Output: 24-hour forecast
    ])
    
    # Use Adam optimizer with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']  # Added MAPE metric
    )
    
    return model

def build_baseline_model(lookback_hours=24, n_features=1):
    """Simple LSTM baseline for comparison"""
    model = Sequential([
        layers.LSTM(32, input_shape=(lookback_hours, n_features)),
        layers.Dense(lookback_hours, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    return model
