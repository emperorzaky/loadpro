# ===================================================
# bayesopt_search.py v1.2
# ---------------------------------------------------
# LOADPRO | Bayesian Optimization Tuning Method
#
# Deskripsi:
# - Menggunakan skopt (Gaussian Process) untuk mencari kombinasi hyperparameter terbaik
# - Evaluasi berdasarkan MAPE (safe from zero division)
# - Model terbaik disimpan ke models/tuning/
# - File log hasil tuning (.pkl) disimpan ke results/tuning/
# ===================================================

from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

import numpy as np
import joblib
import os
import tensorflow as tf

from utils.lstm_train_predict import train_and_evaluate_model

# Ruang pencarian hyperparameter
space = [
    Integer(16, 128, name='hiddenUnits'),
    Integer(3, 10, name='windowSize'),
    Real(1e-4, 1e-2, prior='log-uniform', name='learningRate'),
    Integer(5, 30, name='epochs'),
]

# Variabel global untuk menyimpan state terbaik
best_score = np.inf
best_model = None
best_params = {}

@use_named_args(space)
def objective(**params):
    global best_score, best_model, best_params
    print(f"\n🎯 Evaluating: {params}")

    try:
        mape, rmse, mae = train_and_evaluate_model(X, y, params)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 9999  # Penalti tinggi jika gagal

    print(f"📊 MAPE: {mape:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Jika hasil lebih baik, simpan model
    if mape < best_score:
        best_score = mape
        best_params = params

        # Build & simpan ulang model terbaik
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(params['hiddenUnits'], input_shape=(X.shape[1], X.shape[2])),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learningRate']),
                      loss='mse')
        model.fit(X, y, epochs=params['epochs'], batch_size=32, verbose=0)

        os.makedirs("models/tuning", exist_ok=True)
        model_path = f"models/tuning/{feeder}_{kategori}_best.keras"
        model.save(model_path)
        print(f"💾 Model terbaik disimpan ke: {model_path}")
        tf.keras.backend.clear_session()

    return mape

def run_bayesopt(X_input, y_input, feeder_name, kategori_name):
    global X, y, feeder, kategori
    X, y = X_input, y_input
    feeder = feeder_name
    kategori = kategori_name

    # Eksekusi Bayesian Optimization
    result = gp_minimize(objective, space, n_calls=100, random_state=42)

    print(f"\n🏆 Best MAPE: {best_score:.4f}")
    print("Best Params:", best_params)

    os.makedirs("results/tuning", exist_ok=True)
    result_path = f"results/tuning/{feeder}_{kategori}_bayesopt_result.pkl"
    joblib.dump(result, result_path)
    print(f"📝 Log hasil tuning disimpan ke: {result_path}")
