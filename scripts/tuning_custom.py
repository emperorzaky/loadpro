# =====================================================
# TUNING.PY v1.7 - Corporate-grade Hyperparameter Tuning
# -----------------------------------------------------
# LOADPRO Project | Author: Zaky Pradikto
#
# Features:
# - PSO-based tuning of RNN-LSTM for individual feeders
# - Resume tuning progress from checkpoint logs
# - GPU/CPU auto-detection and log summary
# - CLI support: python tuning.py <feeder> [n_particles] [n_iterations]
# =====================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import time
import numpy as np
import sys
from datetime import datetime
import tensorflow as tf

from utils.train_lstm_model import train_and_evaluate_lstm
from utils.pso_optimizer import pso_optimize
from utils.checkpoint import save_progress
from utils.resume import generate_resume_plan

# --- Logging setup ---
def setup_logger():
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = os.path.join(logs_dir, f"{timestamp}_tuning.log")
    return open(log_file, "a")

def log_print(message, logfile):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    logfile.write(full_message + "\n")

# --- Device info logging ---
def print_device_info():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n🚀 [INFO] GPU available: {gpus[0].name}\n")
    else:
        print("\n⚙️ [INFO] GPU not found. Using CPU.\n")

# --- PSO Objective Wrapper ---
def objective_function(params_array, data, feeder_name, particle_idx=None, iteration_idx=None, total_particles=None, total_iterations=None):
    param_names = ['hidden_units', 'learning_rate', 'window_size', 'epochs']
    params = dict(zip(param_names, params_array))
    params['hidden_units'] = int(params['hidden_units'])
    params['window_size'] = int(params['window_size'])
    params['epochs'] = int(params['epochs'])

    if particle_idx is not None and iteration_idx is not None:
        print(f"🛠️ Training Particle {particle_idx}/{total_particles} pada Iterasi {iteration_idx}/{total_iterations}...")

    try:
        #mape, rmse, mae = train_and_evaluate_lstm(data, params)
        mape, mae, _ = train_and_evaluate_lstm(data, params)
    except Exception:
        mape = np.inf
    return mape

# --- Core Tuning Logic for Single Feeder ---
def tune_feeder(filename, logfile, n_particles=10, n_iterations=20):
    split_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'split'))
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'single'))
    os.makedirs(model_dir, exist_ok=True)

    file_path = os.path.join(split_dir, filename)
    df = pd.read_csv(file_path)
    if 'Beban' not in df.columns:
        raise ValueError(f"File {filename} tidak memiliki kolom 'Beban'.")

    data = df[['Timestamp', 'Beban']]
    feeder_name = filename.replace('.csv', '')
    bounds = [(20, 80), (0.0005, 0.003), (7, 14), (10, 30)]
    progress_log = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', f"{feeder_name}_progress.log"))

    completed_combinations = set()
    if os.path.exists(progress_log):
        completed_combinations = generate_resume_plan(progress_log)

    def wrapped_objective(params_array):
        wrapped_objective.particle_idx = getattr(wrapped_objective, 'particle_idx', 0)
        wrapped_objective.iteration_idx = getattr(wrapped_objective, 'iteration_idx', 0)
        wrapped_objective.total_particles = n_particles
        wrapped_objective.total_iterations = n_iterations

        combo = (wrapped_objective.iteration_idx, wrapped_objective.particle_idx)
        if combo in completed_combinations:
            print(f"⏩ Skip Particle {combo} (sudah complete)")
            return np.inf

        return objective_function(
            params_array, data, feeder_name,
            wrapped_objective.particle_idx,
            wrapped_objective.iteration_idx,
            wrapped_objective.total_particles,
            wrapped_objective.total_iterations
        )

    def save_progress_hook(iteration, particle_idx, params, metrics):
        save_progress(
            iteration=iteration,
            particle_idx=particle_idx,
            params={
                'hidden_units': int(params[0]),
                'learning_rate': float(params[1]),
                'window_size': int(params[2]),
                'epochs': int(params[3])
            },
            metrics=metrics,
            progress_log_path=progress_log
        )

    wrapped_objective.save_progress_func = save_progress_hook

    best_params_array = pso_optimize(
        objective_func=wrapped_objective,
        bounds=bounds,
        n_particles=n_particles,
        n_iterations=n_iterations,
        inertia=0.7,
        cognitive=1.4,
        social=2.4
    )

    param_names = ['hidden_units', 'learning_rate', 'window_size', 'epochs']
    best_params = dict(zip(param_names, best_params_array))
    best_params['hidden_units'] = int(best_params['hidden_units'])
    best_params['window_size'] = int(best_params['window_size'])
    best_params['epochs'] = int(best_params['epochs'])

    mape, mae, model = train_and_evaluate_lstm(data, best_params)
    model_path = os.path.join(model_dir, f"{feeder_name}.keras")
    if os.path.exists(model_path):
        os.remove(model_path)
    model.save(model_path)

    log_print(f"✅ Best Params for {filename}: {best_params}", logfile)
    log_print(f"📈 MAPE: {mape:.2f}%, MAE: {mae:.4f}", logfile)
    log_print("-" * 60, logfile)

# --- Entry Point ---
def main():
    print_device_info()
    logfile = setup_logger()

    if len(sys.argv) >= 2:
        feeder = sys.argv[1]
        n_particles = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        n_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        try:
            tune_feeder(f"{feeder}.csv", logfile, n_particles, n_iterations)
        except Exception as e:
            log_print(f"❌ Error: {str(e)}", logfile)
    else:
        log_print("⚠️ Jalankan dengan: python tuning.py <feeder> [n_particles] [n_iterations]", logfile)

    logfile.close()

if __name__ == "__main__":
    main()
