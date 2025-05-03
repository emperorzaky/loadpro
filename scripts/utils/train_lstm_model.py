"""
TRAIN_LSTM_MODEL.PY v1.4
------------------------
LOADPRO Project | Utilitas Training dan Evaluasi Model LSTM

Deskripsi:
- Melatih model LSTM dengan parameter yang diberikan.
- Mengembalikan hasil evaluasi berupa MAPE, MAE, dan model terlatih.
- Mengimplementasikan pembersihan memori eksplisit (K.clear_session + gc.collect).
- Menambahkan EarlyStopping untuk menghindari overfitting.

Perubahan v1.4:
- Integrasi EarlyStopping (patience=5, monitor='val_loss')
- Evaluasi dilakukan pada data validasi
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import gc

from sklearn.metrics import mean_absolute_error

def create_lstm_model(input_shape, hidden_units):
    """
    Membuat arsitektur LSTM sederhana.
    """
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Menghitung MAPE dengan penanganan pembagi nol.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_evaluate_lstm(data, params):
    """
    Fungsi utama untuk melatih dan mengevaluasi model LSTM.

    Args:
        data (tuple): (X_train, y_train, X_val, y_val)
        params (dict): {'hiddenUnits', 'windowSize', 'epochs', ...}

    Returns:
        mape (float): Mean Absolute Percentage Error
        mae (float): Mean Absolute Error
        model (Keras model): Model terlatih
    """
    try:
        X_train, y_train, X_val, y_val = data

        model = create_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            hidden_units=params['hiddenUnits']
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0 
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0 #set ke 1 untuk melihat early stop
        )

        y_pred = model.predict(X_val, verbose=0).flatten()

        mape = mean_absolute_percentage_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)

        return float(mape), float(mae), model

    finally:
        K.clear_session()
        gc.collect()
