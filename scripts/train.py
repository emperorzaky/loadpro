"""
train.py v1.6.1

Deskripsi:
-----------
Melatih model LSTM menggunakan data hasil preprocessing (.npz) dan menyimpan model .keras hasil training.
Versi ini mendukung fallback otomatis ke CPU dengan menjalankan ulang proses training dalam subprocess jika GPU (cuDNN) gagal.

Penggunaan:
-----------
    python3 scripts/train.py --feeder penyulang_aragog --kategori siang
    python3 scripts/train.py --feeder penyulang_aragog --kategori siang --force_cpu

Output:
--------
- Model disimpan di: models/single/{feeder}_{kategori}.keras
- Log training: logs/train/YYYYMMDD_HHMM_{feeder}_{kategori}_train.log

Author: Zaky Pradikto
"""

import os
import argparse
import numpy as np
import joblib
import subprocess
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from utils.device import get_device

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--feeder', required=True, help="Nama penyulang")
parser.add_argument('--kategori', required=True, choices=['siang', 'malam'], help="Kategori waktu")
parser.add_argument('--force_cpu', action='store_true', help="Paksa training di CPU")
args = parser.parse_args()

# Paksa CPU jika diminta
if args.force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Aktifkan memory growth jika GPU digunakan
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# Path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
npz_path = os.path.join(base_dir, 'data', 'npz', f"{args.feeder}_{args.kategori}.npz")
model_dir = os.path.join(base_dir, 'models', 'single')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"{args.feeder}_{args.kategori}.keras")
log_dir = os.path.join(base_dir, 'logs', 'train')
os.makedirs(log_dir, exist_ok=True)
now_str = datetime.now().strftime("%Y%m%d_%H%M")
log_path = os.path.join(log_dir, f"{now_str}_{args.feeder}_{args.kategori}_train.log")

# Load data
print(f"💻 Menggunakan device: {get_device()}")
data = np.load(npz_path)
X, y = data['X'], data['y']
print(f"✅ Data loaded: X shape = {X.shape}, y shape = {y.shape}")

# Fungsi pembuat model
def build_model():
    model = Sequential()
    model.add(LSTM(
        50,
        input_shape=(X.shape[1], X.shape[2]),
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True,
        return_sequences=False,
        unroll=False
    ))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

# Training
callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
model = build_model()
print("\n🚀 Starting training...")
try:
    history = model.fit(X, y, epochs=50, batch_size=32, verbose=1, callbacks=callbacks)
except tf.errors.InternalError:
    if args.force_cpu:
        raise RuntimeError("🔥 Fallback CPU juga gagal. Abort.")
    print("⚠️ cuDNN gagal. Menjalankan ulang training di CPU...")
    subprocess.run([
        'python3', os.path.abspath(__file__),
        '--feeder', args.feeder,
        '--kategori', args.kategori,
        '--force_cpu'
    ])
    exit()

# Save
model.save(model_path)
print(f"\n💾 Model saved to: {model_path}")

# Save log
with open(log_path, 'w') as f:
    f.write(f"Training log: {args.feeder} ({args.kategori})\n")
    f.write(f"Device: {get_device()}\n")
    f.write(f"X shape: {X.shape}, y shape: {y.shape}\n")
    f.write(f"Final loss: {history.history['loss'][-1]:.4f}\n")
    f.write(f"Final mae: {history.history['mae'][-1]:.4f}\n")

print(f"📄 Log saved to: {log_path}")
