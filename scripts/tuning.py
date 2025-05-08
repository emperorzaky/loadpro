# TUNING.PY v2.6
# -------------------------------------------------------------
# LOADPRO Project | Hyperparameter Tuning Pipeline (Parallel PSO + Meta Passing + Resume + Save Model)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #baris ini bertujuan menyembunyikan log warning tidak penting dari TensorFlow.

import pandas as pd #Pandas digunakan untuk membaca dan mengelola data CSV penyulang.
import time
import numpy as np
import gc #Digunakan untuk mengelola dan membebaskan memori secara manual setelah training selesai.
import psutil #Dipakai untuk mengecek penggunaan memori fisik (RAM) selama proses tuning.
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import subprocess
import json
import uuid

from utils.prepare_dataset import split_train_val #Fungsi utilitas untuk membagi data menjadi set pelatihan dan validasi (80:20).
from utils.pso_optimizer import pso_optimize
from utils.resume import generate_resume_plan
from utils.train_lstm_model import train_and_evaluate_lstm

def setup_logger(): #Membuat file log baru untuk mencatat aktivitas tuning per eksekusi.
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = os.path.join(logs_dir, f"{timestamp}_tuning.log")
    return open(log_file, "a")

def log_print(message, logfile): #Fungsi helper untuk mencetak dan mencatat log dengan timestamp.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    logfile.write(full_message + "\n")

def log_memory(prefix=""): #Mencetak informasi penggunaan memori saat ini, untuk debugging jika terjadi OOM.
    mem = psutil.virtual_memory()
    print(f"[MEM] {prefix} | Used: {mem.used / (1024**3):.2f} GB | Free: {mem.available / (1024**3):.2f} GB")

def print_device_info(): #Menampilkan informasi perangkat GPU jika tersedia, dan mengaktifkan memory growth TensorFlow.
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

def save_progress_json(feeder, iteration, particle_idx, params, metrics, progress_log_path): #Menyimpan hasil tuning setiap partikel ke log JSON agar bisa dilanjutkan (resume).
    entry = {
        'feeder': feeder,
        'iteration': iteration,
        'particle': particle_idx,
        'params': params,
        'metrics': metrics
    }
    with open(progress_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')

def objective_function(params_array, meta): #Pertimbangkan menambahkan komentar fungsi yang menjelaskan bahwa ini dieksekusi dalam subprocess dan menerima konteks melalui meta_dict.
    param_names = ['hiddenUnits', 'learning_rate', 'windowSize', 'epochs']
    params = dict(zip(param_names, params_array))
    params['hiddenUnits'] = int(params['hiddenUnits'])
    params['windowSize'] = int(params['windowSize'])
    params['epochs'] = int(params['epochs'])
    params.update(meta['data'])

    if (meta['iteration'], meta['particle']) in meta['completed']:
        print(f"⏩ Skip Particle ({meta['iteration']}, {meta['particle']}) (sudah complete)")
        return float('inf')

    try:
        uid = uuid.uuid4().hex[:8] #Membuat ID unik untuk setiap eksekusi partikel agar file tidak bentrok antar proses.
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

        with open(os.path.join(base_dir, 'logs', f"{meta['feeder_name']}_subprocess.log"), "a") as log_file: #Menangkap seluruh output dari subprocess ke dalam file log terpisah per feeder.
            subprocess.run([
                'python3', 'scripts/eval_single_model.py',
                '--data', data_path,
                '--params', params_path,
                '--output', result_path
            ], stdout=log_file, stderr=log_file, check=True)

        with open(result_path, 'r') as f:
            result = json.load(f)

        score = float(result['mape'])

    except Exception as e:
        print(f"[ERROR] Subprocess error: {e}")
        return float('inf')
    finally:
        for file in [data_path, params_path, result_path]:
            try:
                os.remove(file)
            except:
                pass

    resume_entry = {
        "hiddenUnits": int(params_array[0]),
        "learning_rate": float(params_array[1]),
        "windowSize": int(params_array[2]),
        "epochs": int(params_array[3])
    }
    save_progress_json(
        feeder=meta['filename'],
        iteration=meta['iteration'],
        particle_idx=meta['particle'],
        params=resume_entry,
        metrics={"mape": score},
        progress_log_path=meta['progress_log']
    )

    return score

def tune_all_feeders(): #Pertimbangkan menjelaskan secara singkat dalam komentar bahwa ini adalah entry-point utama untuk seluruh proses tuning dan training ulang model terbaik.
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

        for filename in tqdm(all_files, desc="Tuning Feeders"):
            try:
                log_print(f"🚀 Starting tuning: {filename}", logfile)
                file_path = os.path.join(split_dir, filename)
                df = pd.read_csv(file_path)

                if 'Beban' not in df.columns:
                    raise ValueError(f"File {filename} tidak memiliki kolom 'Beban'.")

                series = df['Beban'].values #Mengambil nilai beban dari kolom CSV sebagai data time-series yang akan diproses.
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

                X_train, X_val, y_train, y_val = split_train_val(X, y, test_size=0.2)
                log_print(f"📊 Split rasio train:val = 80:20 ({len(X_train)} train, {len(X_val)} val)", logfile)

                extra_args_list = [] #Sebaiknya tambahkan komentar sebelum bagian ini untuk menjelaskan bahwa extra_args_list digunakan untuk menyusun konfigurasi parameter per particle per iterasi agar bisa diproses paralel.
                for iteration in range(20):
                    meta_iteration = []
                    for particle in range(100):
                        meta_iteration.append({
                            'iteration': iteration + 1,
                            'particle': particle + 1,
                            'completed': completed_combinations,
                            'data': {
                                'X_train': X_train,
                                'y_train': y_train,
                                'X_val': X_val,
                                'y_val': y_val
                            },
                            'feeder_name': feeder_name,
                            'progress_log': progress_log,
                            'filename': filename
                        })
                    extra_args_list.append(meta_iteration)

                best_params_array = pso_optimize( #Menjalankan optimasi hyperparameter menggunakan PSO dengan 6 proses paralel.
                    objective_func=objective_function,
                    bounds=bounds,
                    n_particles=100,
                    n_iterations=20,
                    inertia=0.7,
                    cognitive=1.4,
                    social=2.4,
                    extra_args_list=extra_args_list,
                    n_jobs=6
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
                split_best = int(0.8 * len(X_best))
                data_best = (
                    X_best[:split_best], y_best[:split_best],
                    X_best[split_best:], y_best[split_best:]
                )

                _, _, final_model = train_and_evaluate_lstm(data_best, best_params) #Melatih ulang model terbaik dengan seluruh data pelatihan dan validasi (dari parameter terbaik).

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

        elapsed = time.time() - start_time
        m, s = divmod(elapsed, 60)
        log_print(f"🎉 Tuning selesai untuk semua {feeder_count} file.", logfile)
        log_print(f"🕒 Total waktu eksekusi: {int(m)} menit {int(s)} detik", logfile)
        log_print(f"📝 Log aktivitas disimpan di {logfile.name}", logfile)

    finally:
        logfile.close()

if __name__ == "__main__": #Entry point untuk menjalankan fungsi tuning saat script dieksekusi langsung.
    tune_all_feeders()
