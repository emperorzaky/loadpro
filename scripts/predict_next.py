"""
predict_next.py v1.1

Deskripsi:
-----------
Melakukan prediksi beban berikutnya (next-day) untuk satu penyulang, menggunakan window terakhir dari .npz hasil preprocessing dan model .keras hasil training.
Hasil ditampilkan di terminal, disimpan ke .log, dan juga disimpan ke .csv.

Penggunaan:
-----------
    python3 scripts/predict_next.py --feeder penyulang_aragog --kategori siang

Output:
--------
- Terminal output: kalimat hasil prediksi
- Log: logs/predict/YYYYMMDD_HHMM_predict_next_{feeder}_{kategori}.log
- CSV: results/predict/next_{feeder}_{kategori}.csv

Author: Zaky Pradikto
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import argparse
import numpy as np
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
import pandas as pd

# Setup
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# Argumen CLI
parser = argparse.ArgumentParser()
parser.add_argument('--feeder', required=True, help="Nama penyulang tanpa ekstensi")
parser.add_argument('--kategori', required=True, choices=['siang', 'malam'], help="Kategori waktu")
args = parser.parse_args()

# Path dasar
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
npz_path = os.path.join(base_dir, 'data', 'npz', f"{args.feeder}_{args.kategori}.npz")
pkl_path = os.path.join(base_dir, 'data', 'metadata', f"{args.feeder}_{args.kategori}_scaler.pkl")
model_path = os.path.join(base_dir, 'models', 'single', f"{args.feeder}_{args.kategori}.keras")
log_dir = os.path.join(base_dir, 'logs', 'predict')
csv_dir = os.path.join(base_dir, 'results', 'predict')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# Validasi
if not os.path.exists(npz_path):
    raise FileNotFoundError(f"File .npz tidak ditemukan: {npz_path}")
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"Scaler .pkl tidak ditemukan: {pkl_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model .keras tidak ditemukan: {model_path}")

# Load window terakhir
data = np.load(npz_path)
X = data['X']
last_window = X[-1].reshape(1, X.shape[1], 1)

# Load scaler dan model
scaler = joblib.load(pkl_path)
model = load_model(model_path)

# Prediksi
y_pred_scaled = model.predict(last_window)
y_pred_amp = scaler.inverse_transform(y_pred_scaled)[0][0]
y_pred_amp = round(y_pred_amp, 2)

# Format kalimat output
summary = f"Prediksi beban {args.kategori} berikutnya untuk {args.feeder} adalah {y_pred_amp} A"
print(f"🔮 {summary}")

# Simpan log
now_str = datetime.now().strftime("%Y%m%d_%H%M")
log_path = os.path.join(log_dir, f"{now_str}_predict_next_{args.feeder}_{args.kategori}.log")
with open(log_path, 'w') as f:
    f.write(summary + "\n")

# Simpan CSV
csv_path = os.path.join(csv_dir, f"next_{args.feeder}_{args.kategori}.csv")
df = pd.DataFrame({
    'feeder': [args.feeder],
    'kategori': [args.kategori],
    'y_pred': [f"{y_pred_amp} A"]
})
df.to_csv(csv_path, index=False)
