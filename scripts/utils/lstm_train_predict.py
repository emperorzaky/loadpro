# ===================================================
# lstm_train_predict.py v1.3
# ---------------------------------------------------
# LOADPRO Utility | Train and Evaluate LSTM Model
#
# Fungsi:
# - Melatih model LSTM menggunakan data dan parameter tertentu
# - Menghasilkan prediksi, confidence interval, dan metrik evaluasi
# - MAPE dilindungi dari pembagian dengan nol (safe mode)
#
# Output:
# - model: Model terlatih
# - y_pred: Hasil prediksi
# - y_pred_range: Tuple batas bawah dan atas (±10%)
# - mape: Mean Absolute Percentage Error (tanpa inf)
# - rmse: Root Mean Squared Error
# - mae: Mean Absolute Error
# ===================================================

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_and_evaluate_model(X, y, params):
    """
    Melatih model LSTM dan mengevaluasi performa prediksi.

    Parameter:
    ----------
    X : ndarray
        Input data dengan shape (samples, window_size, features).
    y : ndarray
        Target/label data dengan shape (samples,).
    params : dict
        Dictionary parameter hyperparameter model:
            - hiddenUnits (int): Jumlah unit LSTM
            - learningRate (float): Learning rate optimizer
            - epochs (int): Jumlah epoch training

    Return:
    -------
    model : keras.Model
    y_pred : ndarray
    y_pred_range : tuple (y_lower, y_upper)
    mape : float
    rmse : float
    mae : float
    """

    # Bangun model LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(params['hiddenUnits'], input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learningRate']),
        loss='mse'
    )

    # Training model
    model.fit(X, y, epochs=params['epochs'], batch_size=32, verbose=0)

    # Prediksi dan flatten
    y_pred = model.predict(X).flatten()

    # Evaluasi
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Safe MAPE: skip pembagi nol, log warning jika ada
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_vals = np.abs((y - y_pred) / y)
        mape_vals = mape_vals[~np.isinf(mape_vals) & ~np.isnan(mape_vals)]
        mape = np.mean(mape_vals) * 100 if mape_vals.size > 0 else np.inf

    # Confidence Interval ±10%
    y_pred_range = (y_pred * 0.90, y_pred * 1.10)

    # Bersihkan session TensorFlow
    tf.keras.backend.clear_session()

    return model, y_pred, y_pred_range, mape, rmse, mae
