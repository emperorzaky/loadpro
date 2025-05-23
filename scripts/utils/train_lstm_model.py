"""
TRAIN_LSTM_MODEL.PY v1.5
------------------------
LOADPRO Project | Utilitas Training dan Evaluasi Model LSTM

Deskripsi:
- Melatih model LSTM dengan parameter yang diberikan.
- Mengembalikan hasil evaluasi berupa MAPE, MAE, dan model terlatih.
- Mengimplementasikan pembersihan memori eksplisit (K.clear_session + gc.collect).
- Menambahkan EarlyStopping untuk menghindari overfitting.
- Logging device digunakan (GPU/CPU) untuk debugging dan validasi runtime.

Perubahan v1.5:
- Logging device aktif (tf.debugging.set_log_device_placement)
- Komentar lengkap per blok fungsi
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import gc

from sklearn.metrics import mean_absolute_error

# --- Debug mode: log device placement untuk memastikan GPU digunakan ---
tf.debugging.set_log_device_placement(True)

def create_lstm_model(input_shape, hidden_units):
    """
    Membuat arsitektur LSTM sederhana:
    - 1 layer LSTM
    - 1 layer Dense output
    - Optimizer: Adam
    - Loss function: Mean Squared Error
    """
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Menghitung MAPE secara manual (karena MAPE tidak tersedia default di sklearn).
    Menghindari pembagian dengan nol dengan memfilter nilai y_true == 0.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_evaluate_lstm(data, params):
    """
    Fungsi utama untuk melatih dan mengevaluasi model LSTM.

    Args:
        data (tuple): Data training dan validasi: (X_train, y_train, X_val, y_val)
        params (dict): Dictionary parameter yang digunakan, termasuk:
            - hiddenUnits: jumlah unit LSTM
            - windowSize: ukuran jendela input
            - epochs: jumlah epoch pelatihan

    Returns:
        - mape (float): Mean Absolute Percentage Error pada validasi
        - mae (float): Mean Absolute Error pada validasi
        - model (Keras model): Objek model terlatih
    """
    try:
        # --- Load data ---
        X_train, y_train, X_val, y_val = data

        # --- Buat model LSTM ---
        model = create_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            hidden_units=params['hiddenUnits']
        )

        # --- Setup early stopping untuk mencegah overfitting ---
        early_stopping = EarlyStopping(
            monitor='val_loss',           # Pantau loss pada validasi
            patience=5,                   # Berhenti jika tidak ada peningkatan selama 5 epoch
            restore_best_weights=True,    # Kembalikan bobot terbaik
            verbose=0
        )

        # --- Proses training ---
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0  # Set ke 1 jika ingin melihat log training
        )

        # --- Prediksi pada data validasi ---
        y_pred = model.predict(X_val, verbose=0).flatten()

        # --- Evaluasi performa model ---
        mape = mean_absolute_percentage_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)

        return float(mape), float(mae), model

    finally:
        # --- Bersihkan memori Keras dan Python GC ---
        K.clear_session()
        gc.collect()
