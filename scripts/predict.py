"""
predict.py v1.1

Deskripsi:
-----------
Melakukan prediksi menggunakan model .keras dan data input .npz hasil preprocessing.
Output prediksi disimpan dalam format .csv dan dicatat ke dalam log.

Penggunaan:
-----------
    python scripts/predict.py --feeder penyulang_aragog --kategori siang

Output:
--------
- Hasil prediksi disimpan di: results/predict/{feeder}_{kategori}_pred.csv
- Log: logs/predict/YYYYMMDD_HHMM_predict_{feeder}_{kategori}.log

Author: Zaky Pradikto
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import argparse
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--feeder', required=True, help="Nama penyulang tanpa ekstensi")
parser.add_argument('--kategori', required=True, choices=['siang', 'malam'], help="Kategori waktu")
args = parser.parse_args()

# Path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
npz_path = os.path.join(base_dir, 'data', 'npz', f"{args.feeder}_{args.kategori}.npz")
pkl_path = os.path.join(base_dir, 'data', 'metadata', f"{args.feeder}_{args.kategori}_scaler.pkl")
model_path = os.path.join(base_dir, 'models', 'single', f"{args.feeder}_{args.kategori}.keras")
out_dir = os.path.join(base_dir, 'results', 'predict')
log_dir = os.path.join(base_dir, 'logs', 'predict')
os.makedirs(out_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{args.feeder}_{args.kategori}_pred.csv")

# Validasi file
if not os.path.exists(npz_path):
    raise FileNotFoundError(f"Data .npz tidak ditemukan: {npz_path}")
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"Scaler .pkl tidak ditemukan: {pkl_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model .keras tidak ditemukan: {model_path}")

# Load data, scaler, model
data = np.load(npz_path)
X, y_true = data['X'], data['y']
scaler = joblib.load(pkl_path)
model = load_model(model_path)

# Prediksi
y_pred = model.predict(X)

# Inverse transform
y_true_amp = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
y_pred_amp = scaler.inverse_transform(y_pred).flatten()

# Simpan ke CSV
df = pd.DataFrame({
    'y_true': y_true_amp,
    'y_pred': y_pred_amp
})
df.to_csv(out_path, index=False)

# Log
now_str = datetime.now().strftime("%Y%m%d_%H%M")
log_path = os.path.join(log_dir, f"{now_str}_predict_{args.feeder}_{args.kategori}.log")
with open(log_path, 'w') as f:
    f.write(f"🕒 Waktu prediksi: {now_str}\n")
    f.write(f"📂 Feeder: {args.feeder}\n")
    f.write(f"🌓 Kategori: {args.kategori}\n")
    f.write(f"📈 Jumlah sampel: {len(y_true)}\n")
    f.write(f"🔍 Contoh y_true vs y_pred:\n")
    for i in range(min(5, len(y_true))):
        f.write(f"   {y_true_amp[i]:.2f} A → {y_pred_amp[i]:.2f} A\n")
    f.write(f"📄 CSV disimpan di: {out_path}\n")

print(f"✅ Prediksi selesai. Hasil disimpan di: {out_path}")
