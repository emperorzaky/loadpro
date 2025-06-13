# ===================================================
# PREDICT_NEXT.PY v1.1
# ---------------------------------------------------
# Memprediksi beban H+1 berdasarkan window terakhir
# dari hasil preprocessing (.npz dan scaler.pkl).
# Tanggal H+1 diambil dari data raw CSV terakhir.
# Hasil prediksi disimpan dalam bentuk deskriptif .txt
# dan log proses di logs/predict/.
# ===================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# --- Logging Setup ---
def setup_logger(feeder, kategori):
    log_dir = os.path.join("logs", "predict_next")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    log_name = f"{ts}_predict_next_{feeder}_{kategori}.log"
    log_path = os.path.join(log_dir, log_name)
    return open(log_path, "a"), log_path

def log_print(msg, logfile):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line)
    logfile.write(line + "\n")

# --- Main Function ---
def main(feeder, kategori):
    basename = f"{feeder}_{kategori}"
    model_path = f"models/single/{basename}.keras"
    npz_path = f"data/npz/{basename}.npz"
    scaler_path = f"data/metadata/{basename}_scaler.pkl"
    csv_path = f"data/raw/{feeder}.csv"

    logfile, log_path = setup_logger(feeder, kategori)

    try:
        log_print(f"📦 Memuat model: {model_path}", logfile)
        model = load_model(model_path)

        log_print(f"📊 Memuat data window terakhir dari: {npz_path}", logfile)
        data = np.load(npz_path)
        X = data['X']
        x_input = X[-1].reshape(1, X.shape[1], X.shape[2])

        log_print(f"🔄 Inverse transform hasil prediksi...", logfile)
        scaler = joblib.load(scaler_path)
        y_pred_scaled = model.predict(x_input).reshape(-1)[0]
        y_pred = scaler.inverse_transform([[y_pred_scaled]])[0][0]

        log_print(f"📅 Membaca tanggal terakhir dari: {csv_path}", logfile)
        df_raw = pd.read_csv(csv_path)
        df_raw = df_raw[df_raw['Waktu'] == kategori]
        df_raw['Tanggal'] = pd.to_datetime(df_raw['Tanggal'], format='%m/%d/%Y')
        last_date = df_raw['Tanggal'].max()
        next_date = last_date + timedelta(days=1)
        next_date_str = next_date.strftime('%A, %d %B %Y')

        output_dir = "results/predict_next"
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, f"next_{basename}.txt")

        with open(txt_path, "w") as f:
            f.write(f"📈 Hasil Prediksi Beban H+1\n")
            f.write(f"Penyulang : {feeder}\n")
            f.write(f"Kategori : {kategori}\n")
            f.write(f"Tanggal  : {next_date_str}\n")
            f.write(f"Beban    : {y_pred:.2f} A\n")

        log_print(f"✅ Prediksi beban H+1 = {y_pred:.2f} A", logfile)
        log_print(f"📄 Hasil disimpan di: {txt_path}", logfile)
        log_print(f"📝 Log tersimpan di: {log_path}", logfile)
        log_print(f"🎉 Prediksi selesai untuk {feeder} ({kategori}).", logfile)
    except Exception as e:
        log_print(f"❌ Terjadi kesalahan: {e}", logfile)

    logfile.close()

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🔮 Prediksi beban H+1 dari window terakhir.")
    parser.add_argument('--feeder', required=True, help='Nama penyulang tanpa ekstensi')
    parser.add_argument('--kategori', required=True, choices=['siang', 'malam'], help='Kategori waktu')
    args = parser.parse_args()

    main(args.feeder, args.kategori)
