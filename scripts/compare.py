""# ===================================================
# COMPARE.PY v1.1
# ---------------------------------------------------
# Membandingkan model .keras dari folder temporary/ dengan
# model lama di folder single/, berdasarkan nilai RMSE.
# Jika model baru lebih baik, maka akan overwrite model lama.
# Jika model lama lebih baik, maka model baru akan dihapus.
# Semua hasil perbandingan dicatat ke dalam logs/compare/.
# ===================================================
# python3 scripts/compare.py --feeder penyulang_bancang --kategori siang

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import joblib
import numpy as np
import argparse
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.metrics import root_mean_squared_error

# --- Setup Logging ---
def setup_logger(feeder, kategori):
    log_dir = os.path.join("logs", "compare")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f"{ts}_compare_{feeder}_{kategori}.log")
    log_file = open(log_path, "a")
    return log_file, log_path

def log_print(msg, logfile):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line)
    logfile.write(line + "\n")

# --- Bandingkan 2 model ---
def compare_models(feeder, kategori, verbose=True):
    basename = f"{feeder}_{kategori}"
    old_model_path = f"models/single/{basename}.keras"
    new_model_path = f"models/temporary/{basename}.keras"
    scaler_path = f"data/metadata/{basename}_scaler.pkl"
    npz_path = f"data/npz/{basename}.npz"

    logfile, log_path = setup_logger(feeder, kategori)

    if not os.path.exists(new_model_path):
        log_print(f"⛔ Model baru tidak ditemukan: {new_model_path}", logfile)
        return
    if not os.path.exists(old_model_path):
        log_print(f"⚠️ Model lama tidak ditemukan, langsung gunakan model baru.", logfile)
        os.replace(new_model_path, old_model_path)
        log_print(f"📄 Log tersimpan di: {log_path}", logfile)
        logfile.close()
        return

    # Load model dan data
    model_old = load_model(old_model_path)
    model_new = load_model(new_model_path)
    data = np.load(npz_path)
    scaler = joblib.load(scaler_path)
    X, y_true = data['X'], data['y']

    y_old = model_old.predict(X).reshape(-1)
    y_new = model_new.predict(X).reshape(-1)

    rmse_old = root_mean_squared_error(y_true, y_old)
    rmse_new = root_mean_squared_error(y_true, y_new)

    log_print(f"🆕 RMSE model baru : {rmse_new:.6f}", logfile)
    log_print(f"📦 RMSE model lama : {rmse_old:.6f}", logfile)

    if rmse_new < rmse_old:
        os.replace(new_model_path, old_model_path)
        log_print(f"✅ Model baru LEBIH BAIK. Menimpa model lama.", logfile)
    else:
        os.remove(new_model_path)
        log_print(f"❌ Model lama lebih baik. Model baru dihapus.", logfile)

    log_print(f"📄 Log tersimpan di: {logfile.name}", logfile)
    logfile.close()

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🔍 Membandingkan model baru dengan model lama berdasarkan RMSE.")
    parser.add_argument('--feeder', required=True, help='Nama penyulang tanpa ekstensi')
    parser.add_argument('--kategori', required=True, choices=['siang', 'malam'], help='Kategori waktu')
    parser.add_argument('--verbose', action='store_true', help='Tampilkan log detail ke terminal')
    args = parser.parse_args()

    compare_models(args.feeder, args.kategori, args.verbose)
