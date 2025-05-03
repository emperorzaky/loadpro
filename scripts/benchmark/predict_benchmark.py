'''
PREDICT.PY v1.1
------------------------------
LOADPRO Project | Prediction Pipeline

Deskripsi:
- Melakukan prediksi beban next day untuk seluruh file feeder di data/processed/split
- Menggunakan model hasil tuning (.keras)
- Menyimpan log aktivitas ke file /logs
- Menyimpan hasil prediksi ke /results/prediction_results.csv
- Output prediksi 2 angka di belakang koma dengan narasi informatif

Perubahan v1.1:
- Menghapus tqdm progressbar untuk menghasilkan tampilan console yang lebih bersih dan profesional.
- Memastikan hanya output prediksi yang ditampilkan per feeder.
'''

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta

# --- Setup Logger ---
def log_print(message, logfile=None):
    """
    Menulis pesan ke console dan file log.
    """
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_message = f"{timestamp} {message}"
    print(full_message)
    if logfile:
        with open(logfile, "a") as f:
            f.write(full_message + "\n")

# --- Main Prediction ---
if __name__ == "__main__":
    start_time = time.time()

    split_dir = "data/processed/split"
    model_dir = "models/single"
    result_dir = "results"
    log_dir = "logs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup log file
    log_filename = datetime.now().strftime("%Y%m%d_%H%M_predict.log")
    log_filepath = os.path.join(log_dir, log_filename)

    # Setup prediction result file
    prediction_path = os.path.join(result_dir, "prediction_results.csv")
    if not os.path.exists(prediction_path):
        with open(prediction_path, 'w') as f:
            f.write('Feeder,Tanggal Prediksi,Beban Prediksi (A)\n')

    log_print(f"📁 Checking folder: {split_dir} ...", log_filepath)

    feeders = [f for f in os.listdir(split_dir) if f.endswith(".csv")]
    log_print(f"🔍 Ditemukan {len(feeders)} file data feeder untuk prediksi.", log_filepath)

    for feeder_file in feeders:
        feeder_path = os.path.join(split_dir, feeder_file)
        feeder_name = feeder_file.replace(".csv", "")

        try:
            # Load feeder data
            df = pd.read_csv(feeder_path)

            # Load model
            model_path = os.path.join(model_dir, f"{feeder_name}.keras")
            if not os.path.exists(model_path):
                log_print(f"⚠️ Model tidak ditemukan untuk {feeder_name}, skip.", log_filepath)
                continue

            model = tf.keras.models.load_model(model_path)

            # Prepare input
            window_size = model.input_shape[1]
            latest_window = df['Beban'].values[-window_size:]
            X_pred = np.expand_dims(latest_window, axis=0)

            # Predict
            prediction = model.predict(X_pred, verbose=0)[0][0]

            # Siang/malam
            time_type = "siang" if "siang" in feeder_name else "malam"

            # Prediksi tanggal next day
            last_date = pd.to_datetime(df['Timestamp'].max())
            next_date = last_date + timedelta(days=1)
            next_date_str = next_date.strftime('%d %B %Y')

            # Format feeder name
            feeder_clean = feeder_name.replace("_siang", "").replace("_malam", "").replace("_", " ").title()

            # Log output narasi
            log_print(
                f"📈 Prediksi beban {time_type} untuk penyulang {feeder_clean} pada tanggal {next_date_str} adalah {prediction:.2f} Amper.",
                log_filepath
            )

            # Save ke CSV
            with open(prediction_path, 'a') as f:
                f.write(f"{feeder_clean} ({time_type.capitalize()}),{next_date.strftime('%Y-%m-%d')},{prediction:.2f}\n")

        except Exception as e:
            log_print(f"⚠️ ERROR saat prediksi {feeder_file}: {e}", log_filepath)
            log_print("-" * 60, log_filepath)
            continue

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    log_print(f"🎉 Prediksi selesai untuk semua {len(feeders)} file.", log_filepath)
    log_print(f"🕒 Total waktu eksekusi: {int(minutes)} menit {int(seconds)} detik", log_filepath)
    log_print(f"📝 Log aktivitas disimpan di {log_filepath}", log_filepath)
    log_print(f"📄 Hasil prediksi disimpan di {prediction_path}", log_filepath)
