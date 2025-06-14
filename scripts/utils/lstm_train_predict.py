# lstm_train_predict.py v1.0
# --------------------------------------------------
# Melatih model LSTM dan mengembalikan hasil evaluasi
# Input: X_train, y_train, X_val, y_val, dan parameter dict
# Output: model terlatih dan dict hasil evaluasi
# --------------------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import numpy as np
import gc


def train_and_evaluate_model(X_train, y_train, X_val, y_val, params):
    # Clear session untuk menghindari memory leak
    K.clear_session()
    gc.collect()

    # Ambil parameter
    hidden_units = params['hidden_units']
    learning_rate = params['learning_rate']
    epochs = params['epochs']
    batch_size = params['batch_size']

    # Buat model
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    # Early stopping
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # Fit model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[es]
    )

    # Evaluasi
    y_pred = model.predict(X_val)
    mae = np.mean(np.abs(y_val - y_pred))
    rmse = np.sqrt(np.mean((y_val - y_pred)**2))
    mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8))) * 100 if np.all(y_val != 0) else None

    results = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "epochs": len(history.history['loss'])
    }

    return model, results
