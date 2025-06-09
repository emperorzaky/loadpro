# ===================================================
# PREDICT.PY v1.2
# ---------------------------------------------------
# Melakukan prediksi beban 1 penyulang untuk 1 kategori
# (siang/malam) menggunakan model yang sudah dilatih.
# Output:
# - File CSV hasil prediksi
# - Log evaluasi (MAE, RMSE, MAPE)
# ===================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib


def log_print(msg, logfile):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line)
    logfile.write(line + "\n")


def main(feeder, kategori):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    basename = f"{feeder}_{kategori}"

    # Path
    model_path = f"models/single/{basename}.keras"
    npz_path = f"data/npz/{basename}.npz"
    scaler_path = f"data/metadata/{basename}_scaler.pkl"
    output_path = f"results/predict/{basename}_pred.csv"
    log_dir = "logs/predict"
    os.makedirs("results/predict", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{ts}_predict_{basename}.log")
    logfile = open(log_path, "a")

    # Load
    model = load_model(model_path)
    data = np.load(npz_path)
    scaler = joblib.load(scaler_path)
    X, y_true = data['X'], data['y']

    # Predict
    y_pred = model.predict(X).reshape(-1)

    # Inverse scaling (FIX PATCH v1.2)
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)

    # Evaluasi
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    error = y_pred - y_true
    error_pct = np.where(y_true == 0, np.nan, np.abs(error) / y_true * 100)
    if np.any(y_true == 0):
        mape = None
    else:
        mape = np.mean(error_pct)

    log_print(f"✅ Evaluation:", logfile)
    log_print(f"   MAE  = {mae:.4f}", logfile)
    log_print(f"   RMSE = {rmse:.4f}", logfile)
    if mape is None:
        log_print(f"⚠️ MAPE tidak dihitung karena ada nilai y == 0", logfile)
    else:
        log_print(f"   MAPE = {mape:.2f}%", logfile)

    # Save hasil
    df_out = pd.DataFrame({
        "Beban Aktual": y_true,
        "Beban Prediksi": y_pred,
        "Error": error,
        "Error (%)": error_pct
    })
    df_out.to_csv(output_path, index=False)

    log_print(f"💾 Hasil disimpan di: {output_path}", logfile)
    log_print(f"📄 Log tersimpan di: {log_path}", logfile)
    log_print(f"🎉 Prediksi selesai untuk {feeder} ({kategori}).", logfile)
    logfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🔮 Melakukan prediksi beban penyulang (siang/malam)")
    parser.add_argument("--feeder", required=True, help="Nama file penyulang tanpa ekstensi")
    parser.add_argument("--kategori", required=True, choices=["siang", "malam"], help="Kategori waktu")
    args = parser.parse_args()
    main(args.feeder, args.kategori)
