# ===================================================
# TRAIN.PY v1.0
# ---------------------------------------------------
# LOADPRO Project | Training model RNN-LSTM per feeder per kategori
# Output: Model .keras + log hasil training
# ===================================================
# scripts/train.py --feeder <feeder_name> --kategori <siang|malam>

import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib

# --------------------
# Setup Logging
# --------------------
def setup_logger(feeder, kategori):
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'train'))
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    log_path = os.path.join(log_dir, f"{ts}_train_{feeder}_{kategori}.log")
    return open(log_path, 'w')

def log(logfile, msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line)
    logfile.write(line + "\n")

# --------------------
# Load Data
# --------------------
def load_data(feeder, kategori):
    npz_path = os.path.join('data', 'npz', f'{feeder}_{kategori}.npz')
    with np.load(npz_path) as data:
        return data['X'], data['y']

# --------------------
# Train Model
# --------------------
def train_lstm(X, y, logf):
    input_shape = X.shape[1:]
    log(logf, f"📐 Shape input: {X.shape}, target: {y.shape}")
    
    model = keras.Sequential([
        keras.layers.LSTM(50, input_shape=input_shape),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    log(logf, "🧠 Model compiled. Mulai training...")

    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    history = model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=[early_stop])

    log(logf, f"📉 Final Loss: {history.history['loss'][-1]:.4f}")
    log(logf, f"🛑 Early stopped after {len(history.history['loss'])} epochs")
    return model

# --------------------
# Evaluate and Save
# --------------------
def evaluate_and_save(model, X, y, feeder, kategori, logf):
    pred = model.predict(X, verbose=0).flatten()

    mae = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred, squared=False)

    log(logf, f"✅ Evaluation:")
    log(logf, f"   MAE  = {mae:.4f}")
    log(logf, f"   RMSE = {rmse:.4f}")
    log(logf, f"   Min y = {np.min(y):.4f}, Max y = {np.max(y):.4f}")

    if np.any(y == 0):
        log(logf, f"⚠️ MAPE tidak dihitung karena ada nilai y == 0")
    else:
        mape = mean_absolute_percentage_error(y, pred)
        log(logf, f"   MAPE = {mape:.4f}")

    out_dir = os.path.join('models', 'single')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{feeder}_{kategori}.keras')
    model.save(out_path)
    log(logf, f"💾 Model disimpan di: {out_path}")

# --------------------
# Main Entry
# --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feeder', required=True)
    parser.add_argument('--kategori', choices=['siang', 'malam'], required=True)
    args = parser.parse_args()

    feeder = args.feeder
    kategori = args.kategori
    logf = setup_logger(feeder, kategori)

    try:
        X, y = load_data(feeder, kategori)
        model = train_lstm(X, y, logf)
        evaluate_and_save(model, X, y, feeder, kategori, logf)
        log(logf, "🎉 Training selesai tanpa error.")
    except Exception as e:
        log(logf, f"❌ ERROR: {str(e)}")
    finally:
        logf.close()
