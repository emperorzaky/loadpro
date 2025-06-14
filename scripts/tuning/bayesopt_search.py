# bayesopt_search.py v1.1
# --------------------------------------------------
# Melakukan tuning hyperparameter dengan Bayesian Optimization
# untuk 1 penyulang dan 1 kategori (siang/malam)
# Output: model .keras, hasil tuning .pkl, dan log .log
# --------------------------------------------------

import os
import time
import argparse
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skopt import gp_minimize
from skopt.space import Integer, Real

# -----------------------------------------------
def load_data(npz_path):
    data = np.load(npz_path)
    return data['X'], data['y']

# -----------------------------------------------
def build_model(input_shape, hidden_units, learning_rate):
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# -----------------------------------------------
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

# -----------------------------------------------
def objective(params):
    hidden_units = params[0]
    window_size = params[1]
    learning_rate = params[2]

    try:
        X, y = load_data(npz_path)
        X = X[:, -window_size:, :]
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        model = build_model((window_size, 1), hidden_units, learning_rate)
        callback = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[callback])

        rmse = evaluate_model(model, X_val, y_val)
        return rmse

    except Exception as e:
        print(f"⚠️  Error in objective: {e}")
        return np.inf

# -----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feeder", required=True, help="Nama penyulang")
    parser.add_argument("--kategori", required=True, choices=["siang", "malam"], help="Kategori waktu")
    args = parser.parse_args()

    feeder = args.feeder
    kategori = args.kategori
    method_name = "bayesopt"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    npz_path = f"data/npz/{feeder}_{kategori}.npz"
    os.makedirs("logs/tuning", exist_ok=True)
    os.makedirs("models/tuning", exist_ok=True)
    os.makedirs("results/tuning", exist_ok=True)

    log_path = f"logs/tuning/{timestamp}_tuning_{feeder}_{kategori}_{method_name}.log"
    model_path = f"models/tuning/{feeder}_{kategori}_{method_name}.keras"
    result_path = f"results/tuning/{feeder}_{kategori}_{method_name}_result.pkl"

    with open(log_path, "w") as log_file:
        print(f"[🧪] Mulai tuning {feeder} - {kategori}...", file=log_file)
        print(f"[🗂] File NPZ: {npz_path}", file=log_file)

        space = [
            Integer(16, 128, name='hidden_units'),
            Integer(3, 10, name='window_size'),
            Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate')
        ]

        def wrapped(params):
            return objective(params)

        result = gp_minimize(
            wrapped,
            space,
            n_calls=20,
            random_state=42
        )

        best_params = result.x
        print(f"[✅] Tuning selesai.", file=log_file)
        print(f"[🏆] Best params: hidden_units={best_params[0]}, window_size={best_params[1]}, lr={best_params[2]:.5f}", file=log_file)

        # Train ulang model terbaik
        X, y = load_data(npz_path)
        X = X[:, -best_params[1]:, :]
        model = build_model((best_params[1], 1), best_params[0], best_params[2])
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        model.save(model_path)
        print(f"[💾] Model terbaik disimpan di: {model_path}", file=log_file)

        with open(result_path, "wb") as f:
            pickle.dump({
                "feeder": feeder,
                "kategori": kategori,
                "method": method_name,
                "best_params": {
                    "hidden_units": best_params[0],
                    "window_size": best_params[1],
                    "learning_rate": best_params[2],
                },
                "score": result.fun
            }, f)
        print(f"[📊] Hasil tuning disimpan di: {result_path}", file=log_file)
