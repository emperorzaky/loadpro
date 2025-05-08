# TUNING.PY v2.7
# -------------------------------------------------------------
# LOADPRO Project | Hyperparameter Tuning Pipeline (Parallel PSO + Meta Passing + Resume + Save Model + Adaptive Parallelism)

import os # Import library os untuk operasi sistem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Sembunyikan log warning TensorFlow yang tidak penting

import pandas as pd # Import library pandas untuk membaca dan mengelola data CSV
import time # Import library time untuk pengukuran waktu eksekusi
import numpy as np # Import library numpy untuk operasi array dan numerik
import gc # Import library gc untuk manajemen memori manual
import psutil # Import library psutil untuk memantau penggunaan memori fisik (RAM)
import tensorflow as tf # Import library tensorflow untuk deep learning

from datetime import datetime # Import datetime untuk mendapatkan waktu saat ini
from tqdm import tqdm # Import tqdm untuk progress bar

import subprocess # Import subprocess untuk menjalankan proses eksternal
import json # Import json untuk membaca dan menulis file JSON
import uuid # Import uuid untuk membuat ID unik

from utils.prepare_dataset import split_train_val # Import fungsi utilitas untuk membagi data menjadi set pelatihan dan validasi
from utils.pso_optimizer import pso_optimize # Import fungsi utama untuk optimasi PSO
from utils.resume import generate_resume_plan # Import fungsi untuk membuat resume plan dari log
from utils.train_lstm_model import train_and_evaluate_lstm # Import fungsi untuk melatih dan mengevaluasi ulang model LSTM terbaik


# Fungsi untuk membuat file log baru untuk mencatat aktivitas tuning
def setup_logger():
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))  # Tentukan direktori log
    os.makedirs(logs_dir, exist_ok=True)  # Buat direktori jika belum ada
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Format timestamp
    log_file = os.path.join(logs_dir, f"{timestamp}_tuning.log")  # Nama file log
    return open(log_file, "a")  # Kembalikan file log yang terbuka untuk ditulis

# Fungsi helper untuk mencetak dan mencatat log dengan timestamp
def log_print(message, logfile):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format timestamp
    full_message = f"[{timestamp}] {message}"  # Gabungkan timestamp dengan pesan
    print(full_message)  # Cetak pesan ke konsol
    logfile.write(full_message + "\n")  # Tulis pesan ke file log

# Fungsi untuk mencetak informasi penggunaan memori saat ini
def log_memory(prefix=""):
    mem = psutil.virtual_memory()  # Ambil informasi memori
    print(f"[MEM] {prefix} | Used: {mem.used / (1024**3):.2f} GB | Free: {mem.available / (1024**3):.2f} GB")  # Cetak penggunaan memori

# Fungsi untuk menampilkan informasi perangkat GPU jika tersedia
def print_device_info():
    physical_devices = tf.config.list_physical_devices('GPU')  # Daftar perangkat GPU
    if physical_devices:  # Jika GPU tersedia
        print(f"\n🚀 [INFO] GPU available: {physical_devices[0].name}")  # Cetak nama GPU
        print(f"🧠 [INFO] CPU cores: {os.cpu_count()}")  # Cetak jumlah core CPU
        print(f"💾 [INFO] Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB\n")  # Cetak total RAM
        try:
            for gpu in physical_devices:  # Aktifkan memory growth untuk setiap GPU
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:  # Tangani error jika gagal
            print(f"[WARN] GPU memory growth failed: {e}")
    else:  # Jika GPU tidak ditemukan
        print("\n⚙️ [INFO] GPU not found. Using CPU for training.\n")

# Fungsi untuk menghitung jumlah proses paralel adaptif berdasarkan hardware
def get_adaptive_n_jobs():
    cpu_cores = os.cpu_count()  # Ambil jumlah core CPU
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)  # Hitung total RAM dalam GB
    est_ram_per_proc = 2.0  # Estimasi RAM yang dibutuhkan per proses
    ram_based_limit = int((total_ram_gb * 0.8) // est_ram_per_proc)  # Hitung batas proses berdasarkan RAM
    return max(1, min(cpu_cores, ram_based_limit))  # Kembalikan jumlah proses paralel yang optimal

# Fungsi untuk menyimpan parameter dan hasil skor MAPE ke file progress
def save_progress_json(feeder, iteration, particle_idx, params, metrics, progress_log_path):
    entry = {  # Buat entri JSON
        'feeder': feeder,
        'iteration': iteration,
        'particle': particle_idx,
        'params': params,
        'metrics': metrics
    }
    with open(progress_log_path, 'a', encoding='utf-8') as f:  # Tulis entri ke file progress
        f.write(json.dumps(entry) + '\n')

# Fungsi objektif untuk optimasi hyperparameter
def objective_function(params_array, meta):
    param_names = ['hiddenUnits', 'learning_rate', 'windowSize', 'epochs']  # Nama parameter
    params = dict(zip(param_names, params_array))  # Gabungkan nama dan nilai parameter
    params['hiddenUnits'] = int(params['hiddenUnits'])  # Konversi ke integer
    params['windowSize'] = int(params['windowSize'])  # Konversi ke integer
    params['epochs'] = int(params['epochs'])  # Konversi ke integer
    params.update(meta['data'])  # Tambahkan data dari meta

    if (meta['iteration'], meta['particle']) in meta['completed']:  # Jika kombinasi sudah selesai
        print(f"⏩ Skip Particle ({meta['iteration']}, {meta['particle']}) (sudah complete)")  # Lewati
        return float('inf')  # Kembalikan nilai tak hingga

    try:
        uid = uuid.uuid4().hex[:8]  # Buat ID unik
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Direktori dasar
        input_dir = os.path.join(base_dir, 'input')  # Direktori input
        os.makedirs(input_dir, exist_ok=True)  # Buat direktori jika belum ada

        data_path = os.path.join(input_dir, f"data_{uid}.npz")  # Path file data
        params_path = os.path.join(input_dir, f"params_{uid}.json")  # Path file parameter
        result_path = os.path.join(input_dir, f"result_{uid}.json")  # Path file hasil

        np.savez_compressed(data_path, **{  # Simpan data ke file NPZ
            'X_train': params['X_train'],
            'y_train': params['y_train'],
            'X_val': params['X_val'],
            'y_val': params['y_val']
        })
        json.dump({k: v for k, v in params.items() if k in param_names}, open(params_path, 'w'))  # Simpan parameter ke file JSON

        with open(os.path.join(base_dir, 'logs', f"{meta['feeder_name']}_subprocess.log"), "a") as log_file:  # Log subprocess
            subprocess.run([  # Jalankan subprocess untuk evaluasi model
                'python3', 'scripts/eval_single_model.py',
                '--data', data_path,
                '--params', params_path,
                '--output', result_path
            ], stdout=log_file, stderr=log_file, check=True)

        with open(result_path, 'r') as f:  # Baca hasil dari file JSON
            result = json.load(f)

        score = float(result['mape'])  # Ambil skor MAPE

    except Exception as e:  # Tangani error
        print(f"[ERROR] Subprocess error: {e}")
        return float('inf')  # Kembalikan nilai tak hingga
    finally:
        for file in [data_path, params_path, result_path]:  # Hapus file sementara
            try:
                os.remove(file)
            except:
                pass

    resume_entry = {  # Buat entri resume
        "hiddenUnits": int(params_array[0]),
        "learning_rate": float(params_array[1]),
        "windowSize": int(params_array[2]),
        "epochs": int(params_array[3])
    }
    save_progress_json(  # Simpan progress ke file JSON
        feeder=meta['filename'],
        iteration=meta['iteration'],
        particle_idx=meta['particle'],
        params=resume_entry,
        metrics={"mape": score},
        progress_log_path=meta['progress_log']
    )

    return score  # Kembalikan skor MAPE

# Fungsi utama untuk menjalankan proses tuning dan training ulang model terbaik
def tune_all_feeders():
    start_time = time.time()  # Catat waktu mulai
    logfile = setup_logger()  # Buat file log

    try:
        print_device_info()  # Cetak informasi perangkat

        split_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'split'))  # Direktori data split
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'single'))  # Direktori model
        os.makedirs(model_dir, exist_ok=True)  # Buat direktori jika belum ada

        log_print(f"📁 Checking folder: {split_dir} ...", logfile)  # Log pengecekan folder
        all_files = [f for f in os.listdir(split_dir) if f.endswith('.csv')]  # Ambil semua file CSV
        feeder_count = len(all_files)  # Hitung jumlah file
        log_print(f"🔍 Ditemukan {feeder_count} file data feeder untuk tuning.", logfile)  # Log jumlah file

        for filename in tqdm(all_files, desc="Tuning Feeders"):  # Iterasi setiap file feeder
            try:
                log_print(f"🚀 Starting tuning: {filename}", logfile)  # Log mulai tuning
                file_path = os.path.join(split_dir, filename)  # Path file
                df = pd.read_csv(file_path)  # Baca file CSV

                if 'Beban' not in df.columns:  # Jika kolom 'Beban' tidak ada
                    raise ValueError(f"File {filename} tidak memiliki kolom 'Beban'.")  # Lempar error

                series = df['Beban'].values  # Ambil nilai kolom 'Beban' sebagai data time-series
                window_size = 7  # Ukuran jendela
                X, y = [], []  # Inisialisasi X dan y
                for i in range(len(series) - window_size):  # Iterasi untuk membuat data input-output
                    X.append(series[i:i+window_size])
                    y.append(series[i + window_size])
                X = np.array(X).reshape(-1, window_size, 1)  # Ubah X menjadi array numpy
                y = np.array(y)  # Ubah y menjadi array numpy

                feeder_name = filename.replace('.csv', '')  # Nama feeder tanpa ekstensi
                bounds = [(20, 80), (0.0005, 0.003), (7, 14), (10, 30)]  # Batas parameter untuk PSO
                progress_log = os.path.join(os.path.dirname(__file__), '..', 'logs', f"{feeder_name}_progress.log")  # Path log progress
                completed_combinations = generate_resume_plan(progress_log)  # Buat resume plan dari log
                print(f"🔁 Resume plan ditemukan: {len(completed_combinations)} kombinasi")  # Log jumlah kombinasi yang ditemukan

                X_train, X_val, y_train, y_val = split_train_val(X, y, test_size=0.2)  # Bagi data menjadi train dan val
                log_print(f"📊 Split rasio train:val = 80:20 ({len(X_train)} train, {len(X_val)} val)", logfile)  # Log rasio split

                extra_args_list = []  # Daftar argumen tambahan untuk setiap iterasi dan partikel
                for iteration in range(20):  # Iterasi untuk setiap iterasi PSO
                    meta_iteration = []  # Daftar meta untuk setiap partikel
                    for particle in range(100):  # Iterasi untuk setiap partikel
                        meta_iteration.append({  # Tambahkan meta untuk partikel
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
                    extra_args_list.append(meta_iteration)  # Tambahkan meta iterasi ke daftar argumen tambahan

                n_jobs = get_adaptive_n_jobs()  # Hitung jumlah proses paralel adaptif
                log_print(f"⚙️ Parallel split aktif: {n_jobs} proses (auto-tuned based on hardware)", logfile)  # Log jumlah proses paralel

                best_params_array = pso_optimize(  # Jalankan optimasi hyperparameter menggunakan PSO
                    objective_func=objective_function,
                    bounds=bounds,
                    n_particles=100,
                    n_iterations=20,
                    inertia=0.7,
                    cognitive=1.4,
                    social=2.4,
                    extra_args_list=extra_args_list,
                    n_jobs=n_jobs
                )

                param_names = ['hiddenUnits', 'learning_rate', 'windowSize', 'epochs']  # Nama parameter
                best_params = dict(zip(param_names, best_params_array))  # Gabungkan nama dan nilai parameter terbaik
                best_params['hiddenUnits'] = int(best_params['hiddenUnits'])  # Konversi ke integer
                best_params['windowSize'] = int(best_params['windowSize'])  # Konversi ke integer
                best_params['epochs'] = int(best_params['epochs'])  # Konversi ke integer

                log_print(f"✅ Best Params for {filename}: {best_params}", logfile)  # Log parameter terbaik
                log_memory(f"📦 Final RAM setelah feeder {feeder_name}")  # Log penggunaan RAM

                ws = best_params['windowSize']  # Ambil ukuran jendela terbaik
                X_best, y_best = [], []  # Inisialisasi X dan y terbaik
                for i in range(len(series) - ws):  # Iterasi untuk membuat data input-output terbaik
                    X_best.append(series[i:i+ws])
                    y_best.append(series[i + ws])
                X_best = np.array(X_best).reshape(-1, ws, 1)  # Ubah X menjadi array numpy
                y_best = np.array(y_best)  # Ubah y menjadi array numpy
                split_best = int(0.8 * len(X_best))  # Hitung indeks split terbaik
                data_best = (  # Data terbaik untuk pelatihan dan validasi
                    X_best[:split_best], y_best[:split_best],
                    X_best[split_best:], y_best[split_best:]
                )

                _, _, final_model = train_and_evaluate_lstm(data_best, best_params)  # Latih ulang model terbaik

                model_path = os.path.join(model_dir, f"{feeder_name}.json")  # Path file model
                weights_path = os.path.join(model_dir, f"{feeder_name}.weights.h5")  # Path file bobot
                with open(model_path, 'w') as f:  # Simpan model ke file JSON
                    f.write(final_model.to_json())
                final_model.save_weights(weights_path)  # Simpan bobot model
                log_print(f"💾 Model disimpan: {model_path}, {weights_path}", logfile)  # Log penyimpanan model
                log_print("-" * 60, logfile)  # Log pemisah

            except Exception as e:  # Tangani error saat tuning
                log_print(f"⚠️ ERROR saat tuning {filename}: {str(e)}", logfile)  # Log error
                log_print("-" * 60, logfile)  # Log pemisah

        elapsed = time.time() - start_time  # Hitung waktu eksekusi
        m, s = divmod(elapsed, 60)  # Konversi ke menit dan detik
        log_print(f"🎉 Tuning selesai untuk semua {feeder_count} file.", logfile)  # Log selesai tuning
        log_print(f"🕒 Total waktu eksekusi: {int(m)} menit {int(s)} detik", logfile)  # Log waktu eksekusi
        log_print(f"📝 Log aktivitas disimpan di {logfile.name}", logfile)  # Log lokasi file log

    finally:
        logfile.close()  # Tutup file log

# Entry point untuk menjalankan fungsi tuning saat script dieksekusi langsung
if __name__ == "__main__":
    tune_all_feeders()
