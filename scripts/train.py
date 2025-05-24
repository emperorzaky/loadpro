"""
train.py v1.2

Deskripsi:
-----------
Melatih model LSTM untuk satu penyulang berdasarkan data hasil preprocessing (.npz)
dan scaler (.pkl). Dirancang untuk eksekusi per-feeder via CLI agar efisien dan
scalable untuk 1000+ penyulang. Versi ini menambahkan fitur logging training
ke dalam folder logs/train/ dan info perangkat (CPU/GPU).

Penggunaan:
-----------
    python train.py --feeder penyulang_khaleesi --kategori malam

Output:
-------
- Model tersimpan di: models/single/{feeder}_{kategori}.keras
- Log training tersimpan di: logs/train/YYYYMMDD_HHMM_{feeder}_{kategori}_train.log

Author: Zaky Pradikto
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Paksa training hanya di CPU

import argparse
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# ----------------------------
# Argument Parser CLI
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--feeder', required=True, help="Nama penyulang tanpa ekstensi")
parser.add_argument('--kategori', required=True, choices=['siang', 'malam'], help="Kategori waktu")
args = parser.parse_args()

# ----------------------------
# Deteksi perangkat
# ----------------------------
physical_devices = tf.config.list_physical_devices('GPU')
device_used = 'GPU' if physical_devices else 'CPU'

# ----------------------------
# Path & File Validasi
# ----------------------------
data_dir = os.path.join('data', 'npz')
meta_dir = os.path.join('data', 'metadata')
model_dir = os.path.join('models', 'single')
log_dir = os.path.join('logs', 'train')
Path(log_dir).mkdir(parents=True, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

npz_path = os.path.join(data_dir, f"{args.feeder}_{args.kategori}.npz")
pkl_path = os.path.join(meta_dir, f"{args.feeder}_{args.kategori}_scaler.pkl")
model_path = os.path.join(model_dir, f"{args.feeder}_{args.kategori}.keras")

if not os.path.exists(npz_path):
    raise FileNotFoundError(f"File .npz tidak ditemukan: {npz_path}")
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"File scaler .pkl tidak ditemukan: {pkl_path}")

# ----------------------------
# Load Data
# ----------------------------
data = np.load(npz_path)
X, y = data['X'], data['y']
print(f"✅ Data loaded: X shape = {X.shape}, y shape = {y.shape}")

# ----------------------------
# Build Model
# ----------------------------
window_size = X.shape[1]
hidden_units = 50

model = Sequential([
    InputLayer(input_shape=(window_size, 1)),
    LSTM(hidden_units),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ----------------------------
# Setup Log File
# ----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_file_name = f"{timestamp}_{args.feeder}_{args.kategori}_train.log"
log_file_path = os.path.join(log_dir, log_file_name)

with open(log_file_path, "w") as f:
    f.write(f"📅 Training Timestamp : {timestamp}\n")
    f.write(f"📁 Feeder             : {args.feeder}\n")
    f.write(f"🕒 Kategori Waktu     : {args.kategori}\n")
    f.write(f"🖥️  Device Digunakan   : {device_used}\n")
    f.write(f"📊 Input Shape (X)    : {X.shape}\n")
    f.write(f"📊 Target Shape (y)   : {y.shape}\n")
    f.write("\n📐 Model Summary:\n")
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# ----------------------------
# Training
# ----------------------------
callbacks = [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
print("\n🚀 Starting training...")
history = model.fit(X, y, epochs=50, batch_size=32, verbose=1, callbacks=callbacks)

# ----------------------------
# Save Model
# ----------------------------
model.save(model_path)
print(f"\n💾 Model saved to: {model_path}")

# ----------------------------
# Save Training Log
# ----------------------------
with open(log_file_path, "a") as f:
    f.write("\n📈 Training History (loss/mae per epoch):\n")
    for i, (l, m) in enumerate(zip(history.history['loss'], history.history['mae'])):
        f.write(f"Epoch {i+1:02d}: Loss = {l:.6f}, MAE = {m:.6f}\n")
    f.write(f"\n💾 Model saved to: {model_path}\n")

print(f"📝 Training log saved to: {log_file_path}")
