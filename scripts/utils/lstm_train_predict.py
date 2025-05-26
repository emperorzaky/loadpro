# ===================================================
# lstm_train_predict.py v1.1
# ---------------------------------------------------
# LOADPRO Utility | Train and Evaluate LSTM Model
#
# Fungsi:
# - Membangun model LSTM berdasarkan parameter tuning
# - Melatih model dan mengevaluasi performa
# - Mengembalikan metrik evaluasi: MAPE, RMSE, MAE
# - Menangani NaN/inf pada MAPE agar tidak menyebabkan error
# ===================================================

import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_and_evaluate_model(X, y, params):
    """
    Latih dan evaluasi model LSTM berdasarkan parameter.

    Args:
        X (np.ndarray): Data input, shape = (samples, window_size, 1)
        y (np.ndarray): Target output, shape = (samples,)
        params (dict): Parameter tuning LSTM, key:
            - hiddenUnits (int)
            - learningRate (float)
            - windowSize (int)
            - epochs (int)

    Returns:
        mape (float): Mean Absolute Percentage Error
        rmse (float): Root Mean Squared Error
        mae (float): Mean Absolute Error
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(params['hiddenUnits'], input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learningRate']),
                  loss='mse')

    model.fit(X, y, epochs=params['epochs'], batch_size=32, verbose=0)

    y_pred = model.predict(X).flatten()
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Safe MAPE (tanpa NaN / inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_vals = np.abs((y - y_pred) / y)
        mape_vals = mape_vals[~np.isinf(mape_vals)]  # Buang inf
        mape_vals = mape_vals[~np.isnan(mape_vals)]  # Buang nan
        mape = np.mean(mape_vals) * 100 if mape_vals.size > 0 else np.inf

    tf.keras.backend.clear_session()
    return mape, rmse, mae
