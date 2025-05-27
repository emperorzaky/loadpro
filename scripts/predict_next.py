# ===================================================
# predict_next.py v1.2
# ---------------------------------------------------
# Melakukan prediksi beban next-day untuk satu penyulang,
# disertai dengan estimasi range ±5% sebagai rentang percaya diri.
#
# Output:
# - Terminal: kalimat prediksi + range
# - Log: logs/predict/YYYYMMDD_HHMM_predict_next_{feeder}_{kategori}.log
# - CSV: results/predict/next_{feeder}_{kategori}.csv
# ===================================================

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import argparse
import numpy as np
import joblib
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

# Argument CLI
parser = argparse.ArgumentParser()
parser.add_argument('--feeder', required=True, help="Nama penyulang tanpa ekstensi")
parser.add_argument('--kategori', required=True, choices=['siang', 'malam'], help="Kategori waktu")
args = parser.parse_args()

# Path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
npz_path   = os.path.join(base_dir, 'data', 'npz', f"{args.feeder}_{args.kategori}.npz")
pkl_path   = os.path.join(base_dir, 'data', 'metadata', f"{args.feeder}_{args.kategori}_scaler.pkl")
model_path = os.path.join(base_dir, 'models', 'single', f"{args.feeder}_{args.kategori}.keras")
log_dir    = os.path.join(base_dir, 'logs', 'predict')
csv_dir    = os.path.join(base_dir, 'results', 'predict')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# Validasi file
for path in [npz_path, pkl_path, model_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tidak ditemukan: {path}")

# Load window terakhir
data = np.load(npz_path)
X = data['X']
last_window = X[-1].reshape(1, X.shape[1], 1)

# Load model dan scaler
scaler = joblib.load(pkl_path)
model = load_model(model_path)

# Prediksi
y_pred_scaled = model.predict(last_window)
y_pred_amp = scaler.inverse_transform(y_pred_scaled)[0][0]
y_pred_amp = round(y_pred_amp, 2)

# Hitung range ±5%
lower = round(y_pred_amp * 0.95, 2)
upper = round(y_pred_amp * 1.05, 2)

# Format output
summary = (
    f"Prediksi beban {args.kategori} berikutnya untuk {args.feeder} adalah "
    f"{y_pred_amp} A (range: {lower} - {upper} A)"
)
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
    'y_pred': [f"{y_pred_amp} A"],
    'range_min': [lower],
    'range_max': [upper]
})
df.to_csv(csv_path, index=False)
