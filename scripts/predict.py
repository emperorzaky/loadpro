# =====================================================
# PREDICT.PY v1.2
# =====================================================
# LOADPRO Project | Next-Day Load Forecasting Pipeline
#
# Deskripsi:
# - Melakukan prediksi beban listrik harian (next day) untuk seluruh file feeder
#   yang telah dipreproses dan disimpan di folder data/processed/split
# - Menggunakan model hasil tuning: format .keras atau fallback .json + .weights.h5
# - Output hasil prediksi dalam format CSV, log aktivitas disimpan di /logs
#
# Perubahan v1.2:
# - Menambahkan dukungan fallback: .json + .weights.h5 jika file .keras tidak tersedia
# - Menampilkan status metode pemuatan model (keras/json) secara informatif
# =====================================================

# --- Import library yang dibutuhkan ---
import os                               # Untuk manipulasi path dan direktori
import time                             # Untuk menghitung durasi eksekusi
import numpy as np                      # Untuk manipulasi array input prediksi
import pandas as pd                     # Untuk membaca data CSV
import tensorflow as tf                 # Untuk load model dan prediksi
from tensorflow.keras.models import model_from_json  # Untuk load model dari .json
from datetime import datetime, timedelta            # Untuk manajemen waktu dan tanggal

# --- Setup Logger ---
def log_print(message, logfile=None):
    """
    Fungsi logging: menampilkan pesan ke console dan menyimpan ke file log (jika disediakan)
    """
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_message = f"{timestamp} {message}"
    print(full_message)
    if logfile:
        with open(logfile, "a") as f:
            f.write(full_message + "\n")

# --- Main Prediction Pipeline ---
if __name__ == "__main__":
    start_time = time.time()  # Catat waktu mulai eksekusi

    # --- Inisialisasi folder-folder utama ---
    split_dir = "data/processed/split"     # Lokasi data feeder hasil preprocessing
    model_dir = "models/single"            # Lokasi model hasil tuning
    result_dir = "results"                 # Lokasi penyimpanan hasil prediksi
    log_dir = "logs"                        # Lokasi penyimpanan log aktivitas
    os.makedirs(result_dir, exist_ok=True) # Buat folder results jika belum ada
    os.makedirs(log_dir, exist_ok=True)    # Buat folder logs jika belum ada

    # --- Inisialisasi file log prediksi ---
    log_filename = datetime.now().strftime("%Y%m%d_%H%M_predict.log")  # Nama file log berdasarkan waktu
    log_filepath = os.path.join(log_dir, log_filename)                 # Path lengkap file log

    # --- Inisialisasi file hasil prediksi ---
    prediction_path = os.path.join(result_dir, "prediction_results.csv")
    if not os.path.exists(prediction_path):
        with open(prediction_path, 'w') as f:
            f.write('Feeder,Tanggal Prediksi,Beban Prediksi (A)\n')   # Header CSV jika belum ada

    # --- Mulai prediksi seluruh feeder ---
    log_print(f"📁 Checking folder: {split_dir} ...", log_filepath)
    feeders = [f for f in os.listdir(split_dir) if f.endswith(".csv")]  # Ambil semua file .csv feeder
    log_print(f"🔍 Ditemukan {len(feeders)} file data feeder untuk prediksi.", log_filepath)

    for feeder_file in feeders:
        feeder_path = os.path.join(split_dir, feeder_file)              # Path lengkap file data
        feeder_name = feeder_file.replace(".csv", "")                  # Nama feeder tanpa ekstensi

        try:
            df = pd.read_csv(feeder_path)                               # Load data feeder ke DataFrame

            # --- Tentukan path model yang tersedia ---
            keras_path = os.path.join(model_dir, f"{feeder_name}.keras")
            json_path = os.path.join(model_dir, f"{feeder_name}.json")
            weights_path = os.path.join(model_dir, f"{feeder_name}.weights.h5")

            # --- Load model berdasarkan prioritas file ---
            if os.path.exists(keras_path):
                model = tf.keras.models.load_model(keras_path)         # Load model .keras
                log_print(f"📦 Memuat model {feeder_name} dari file .keras", log_filepath)
            elif os.path.exists(json_path) and os.path.exists(weights_path):
                with open(json_path, 'r') as f:
                    model = model_from_json(f.read())                  # Load arsitektur dari .json
                model.load_weights(weights_path)                       # Load bobot dari .h5
                log_print(f"📦 Memuat model {feeder_name} dari file .json + .weights.h5", log_filepath)
            else:
                log_print(f"⚠️ Model tidak ditemukan untuk {feeder_name}, skip.", log_filepath)
                continue

            # --- Siapkan data input prediksi ---
            window_size = model.input_shape[1]                          # Ambil panjang jendela input
            latest_window = df['Beban'].values[-window_size:]          # Ambil data terakhir sebanyak window size
            X_pred = np.expand_dims(latest_window, axis=0)             # Ubah ke bentuk (1, window_size, 1)

            # --- Eksekusi prediksi ---
            prediction = model.predict(X_pred, verbose=0)[0][0]       # Prediksi nilai beban

            # --- Klasifikasi jenis data (siang/malam) ---
            time_type = "siang" if "siang" in feeder_name else "malam"

            # --- Estimasi tanggal prediksi (next day) ---
            last_date = pd.to_datetime(df['Timestamp'].max())          # Ambil tanggal terakhir
            next_date = last_date + timedelta(days=1)                  # Tambah 1 hari
            next_date_str = next_date.strftime('%d %B %Y')             # Format human readable

            # --- Format nama feeder agar lebih rapi ---
            feeder_clean = feeder_name.replace("_siang", "").replace("_malam", "").replace("_", " ").title()

            # --- Cetak hasil prediksi ---
            log_print(
                f"📈 Prediksi beban {time_type} untuk penyulang {feeder_clean} pada tanggal {next_date_str} adalah {prediction:.2f} Amper.",
                log_filepath
            )

            # --- Simpan hasil prediksi ke file CSV ---
            with open(prediction_path, 'a') as f:
                f.write(f"{feeder_clean} ({time_type.capitalize()}),{next_date.strftime('%Y-%m-%d')},{prediction:.2f}\n")

        except Exception as e:
            # Tangani error jika terjadi kesalahan saat prediksi
            log_print(f"⚠️ ERROR saat prediksi {feeder_file}: {e}", log_filepath)
            log_print("-" * 60, log_filepath)
            continue

    # --- Akhiri log dengan ringkasan waktu eksekusi ---
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    log_print(f"🎉 Prediksi selesai untuk semua {len(feeders)} file.", log_filepath)
    log_print(f"🕒 Total waktu eksekusi: {int(minutes)} menit {int(seconds)} detik", log_filepath)
    log_print(f"📝 Log aktivitas disimpan di {log_filepath}", log_filepath)
    log_print(f"📄 Hasil prediksi disimpan di {prediction_path}", log_filepath)
