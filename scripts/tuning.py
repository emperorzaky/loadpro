# TUNING.PY v2.4
# -------------------------------------------------------------
# LOADPRO Project | Hyperparameter Tuning Pipeline (PSO + Subprocess + Resume JSON + Memory Safe + Save Model + Detailed Logging)
#
# Fitur:
# - Data dibagi 80% training dan 20% validasi (split_train_val)
# - Optimasi hyperparameter dengan PSO
# - Evaluasi setiap kombinasi parameter dijalankan dalam subprocess
# - Penggunaan memory dijaga tetap aman dari OOM
# - Progress tuning disimpan dalam format JSON untuk keperluan resume
# - Model terbaik dilatih ulang full dan disimpan dalam format JSON + Weights
# - Log granular setiap kombinasi training disimpan untuk resume plan

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import time
import numpy as np
import gc
import psutil
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import subprocess
import json
import uuid

# Import fungsi utilitas utama
from utils.prepare_dataset import split_train_val
from utils.pso_optimizer import pso_optimize
from utils.resume import generate_resume_plan
from utils.train_lstm_model import train_and_evaluate_lstm

# Membuat file log baru untuk mencatat aktivitas tuning

def setup_logger():
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = os.path.join(logs_dir, f"{timestamp}_tuning.log")
    return open(log_file, "a")

# Helper untuk print ke console + log

def log_print(message, logfile):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    logfile.write(full_message + "\n")

# Print informasi memory

def log_memory(prefix=""):
    mem = psutil.virtual_memory()
    print(f"[MEM] {prefix} | Used: {mem.used / (1024**3):.2f} GB | Free: {mem.available / (1024**3):.2f} GB")

# Print device info (GPU/CPU)

def print_device_info():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"\n🚀 [INFO] GPU available: {physical_devices[0].name}\n")
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"[WARN] GPU memory growth failed: {e}")
    else:
        print("\n⚙️ [INFO] GPU not found. Using CPU for training.\n")

# Menyimpan kombinasi parameter dan hasil ke file log JSON

def save_progress_json(feeder, iteration, particle_idx, params, metrics, progress_log_path):
    entry = {
        'feeder': feeder,
        'iteration': iteration,
        'particle': particle_idx,
        'params': params,
        'metrics': metrics
    }
    with open(progress_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')

# Objective function: dijalankan melalui subprocess untuk efisiensi memori

def objective_function(params_array, data, feeder_name, particle_idx=None, iteration_idx=None, total_particles=None, total_iterations=None):
    param_names = ['hiddenUnits', 'learning_rate', 'windowSize', 'epochs']
    params = dict(zip(param_names, params_array))
    params['hiddenUnits'] = int(params['hiddenUnits'])
    params['windowSize'] = int(params['windowSize'])
    params['epochs'] = int(params['epochs'])
    params.update(data)

    if particle_idx is not None and iteration_idx is not None:
        print(f"🛠️ Training Particle {particle_idx}/{total_particles} pada Iterasi {iteration_idx}/{total_iterations}...")

    try:
        uid = uuid.uuid4().hex[:8]
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        input_dir = os.path.join(base_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        data_path = os.path.join(input_dir, f"data_{uid}.npz")
        params_path = os.path.join(input_dir, f"params_{uid}.json")
        result_path = os.path.join(input_dir, f"result_{uid}.json")

        np.savez_compressed(data_path, **{
            'X_train': params['X_train'],
            'y_train': params['y_train'],
            'X_val': params['X_val'],
            'y_val': params['y_val']
        })
        json.dump({k: v for k, v in params.items() if k in param_names}, open(params_path, 'w'))

        subprocess_log = os.path.join(base_dir, 'logs', f"{feeder_name}_subprocess.log")
        with open(subprocess_log, "a") as log_file:
            subprocess.run([
                'python3', 'scripts/eval_single_model.py',
                '--data', data_path,
                '--params', params_path,
                '--output', result_path
            ], stdout=log_file, stderr=log_file, check=True)

        with open(result_path, 'r') as f:
            result = json.load(f)

        return float(result['mape'])

    except Exception as e:
        print(f"[ERROR] Subprocess error: {e}")
        return float('inf')
    finally:
        for file in [data_path, params_path, result_path]:
            try:
                os.remove(file)
            except:
                pass

# Fungsi utama: tuning semua penyulang

def tune_all_feeders():
    start_time = time.time()
    logfile = setup_logger()

    try:
        print_device_info()

        split_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'split'))
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'single'))
        os.makedirs(model_dir, exist_ok=True)

        log_print(f"📁 Checking folder: {split_dir} ...", logfile)
        all_files = [f for f in os.listdir(split_dir) if f.endswith('.csv')]
        feeder_count = len(all_files)
        log_print(f"🔍 Ditemukan {feeder_count} file data feeder untuk tuning.", logfile)

        test_size_ratio = 0.2

        for filename in tqdm(all_files, desc="Tuning Feeders"):
            try:
                log_print(f"🚀 Starting tuning: {filename}", logfile)
                file_path = os.path.join(split_dir, filename)
                df = pd.read_csv(file_path)

                if 'Beban' not in df.columns:
                    raise ValueError(f"File {filename} tidak memiliki kolom 'Beban'.")

                series = df['Beban'].values
                window_size = 7
                X, y = [], []
                for i in range(len(series) - window_size):
                    X.append(series[i:i+window_size])
                    y.append(series[i + window_size])
                X = np.array(X).reshape(-1, window_size, 1)
                y = np.array(y)

                feeder_name = filename.replace('.csv', '')
                bounds = [(20, 80), (0.0005, 0.003), (7, 14), (10, 30)]
                progress_log = os.path.join(os.path.dirname(__file__), '..', 'logs', f"{feeder_name}_progress.log")
                completed_combinations = generate_resume_plan(progress_log)
                print(f"🔁 Resume plan ditemukan: {len(completed_combinations)} kombinasi")

                X_train_fixed, X_val_fixed, y_train_fixed, y_val_fixed = split_train_val(X, y, test_size=test_size_ratio)
                log_print(f"📊 Split rasio train:val = {100 - int(test_size_ratio*100)}:{int(test_size_ratio*100)} ({len(X_train_fixed)} train, {len(X_val_fixed)} val)", logfile)

                def wrapped_objective(params_array, _):
                    current_iter = wrapped_objective.iteration_idx
                    current_particle = wrapped_objective.particle_idx

                    if (current_iter, current_particle) in completed_combinations:
                        print(f"⏩ Skip Particle ({current_iter}, {current_particle}) (sudah complete)")
                        return float('inf')

                    score = objective_function(
                        params_array=params_array,
                        data={
                            'X_train': X_train_fixed,
                            'y_train': y_train_fixed,
                            'X_val': X_val_fixed,
                            'y_val': y_val_fixed
                        },
                        feeder_name=feeder_name,
                        particle_idx=current_particle,
                        iteration_idx=current_iter,
                        total_particles=wrapped_objective.total_particles,
                        total_iterations=wrapped_objective.total_iterations
                    )

                    resume_entry = {
                        "hiddenUnits": int(params_array[0]),
                        "learning_rate": float(params_array[1]),
                        "windowSize": int(params_array[2]),
                        "epochs": int(params_array[3])
                    }
                    save_progress_json(
                        feeder=filename,
                        iteration=current_iter,
                        particle_idx=current_particle,
                        params=resume_entry,
                        metrics={"mape": score},
                        progress_log_path=progress_log
                    )

                    return score

                wrapped_objective.iteration_idx = 0
                wrapped_objective.particle_idx = 0
                wrapped_objective.total_particles = 10
                wrapped_objective.total_iterations = 20

                data_dummy = {}

                best_params_array = pso_optimize(
                    objective_func=wrapped_objective,
                    bounds=bounds,
                    n_particles=10,
                    n_iterations=20,
                    inertia=0.7,
                    cognitive=1.4,
                    social=2.4,
                    extra_args=[data_dummy],
                )

                param_names = ['hiddenUnits', 'learning_rate', 'windowSize', 'epochs']
                best_params = dict(zip(param_names, best_params_array))
                best_params['hiddenUnits'] = int(best_params['hiddenUnits'])
                best_params['windowSize'] = int(best_params['windowSize'])
                best_params['epochs'] = int(best_params['epochs'])

                log_print(f"✅ Best Params for {filename}: {best_params}", logfile)
                log_memory(f"📦 Final RAM setelah feeder {feeder_name}")

                ws = best_params['windowSize']
                X_best, y_best = [], []
                for i in range(len(series) - ws):
                    X_best.append(series[i:i+ws])
                    y_best.append(series[i + ws])
                X_best = np.array(X_best).reshape(-1, ws, 1)
                y_best = np.array(y_best)
                split_best = int((1 - test_size_ratio) * len(X_best))
                data_best = (
                    X_best[:split_best], y_best[:split_best],
                    X_best[split_best:], y_best[split_best:]
                )

                _, _, final_model = train_and_evaluate_lstm(data_best, best_params)

                model_path = os.path.join(model_dir, f"{feeder_name}.json")
                weights_path = os.path.join(model_dir, f"{feeder_name}.weights.h5")
                with open(model_path, 'w') as f:
                    f.write(final_model.to_json())
                final_model.save_weights(weights_path)
                log_print(f"💾 Model disimpan: {model_path}, {weights_path}", logfile)
                log_print("-" * 60, logfile)

            except Exception as e:
                log_print(f"⚠️ ERROR saat tuning {filename}: {str(e)}", logfile)
                log_print("-" * 60, logfile)

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        log_print(f"🎉 Tuning selesai untuk semua {feeder_count} file.", logfile)
        log_print(f"🕒 Total waktu eksekusi: {int(minutes)} menit {int(seconds)} detik", logfile)
        log_print(f"📝 Log aktivitas disimpan di {logfile.name}", logfile)

    finally:
        logfile.close()

if __name__ == "__main__":
    tune_all_feeders()
