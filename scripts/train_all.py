"""
train_all.py

Deskripsi:
-----------
Script untuk melakukan training semua file .npz di folder data/npz.
Akan melewati file yang model .keras-nya sudah tersedia.
Mencetak log ringkas per feeder ke terminal.

Penggunaan:
-----------
    python scripts/train_all.py

Author: Zaky Pradikto
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import subprocess
from datetime import datetime

# Folder sumber dan model
npz_dir = os.path.join('data', 'npz')
model_dir = os.path.join('models', 'single')

# Ambil semua file .npz
files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
print(f"📦 Menemukan {len(files)} file .npz")
print("------------------------------------------")

# Jalankan training per feeder
for file in files:
    feeder_kat = file.replace('.npz', '')
    feeder_parts = feeder_kat.split('_')
    if len(feeder_parts) < 2:
        print(f"⚠️  Lewati file tidak valid: {file}")
        continue

    feeder = '_'.join(feeder_parts[:-1])
    kategori = feeder_parts[-1]

    model_path = os.path.join(model_dir, f"{feeder}_{kategori}.keras")
    if os.path.exists(model_path):
        print(f"✅ SKIP {feeder}_{kategori} — model sudah ada")
        continue

    print(f"🚀 Training {feeder}_{kategori}...")
    cmd = ["python3", "scripts/train.py", "--feeder", feeder, "--kategori", kategori]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ GAGAL training {feeder}_{kategori}: {e}")
    print("------------------------------------------")

print("🎉 Selesai training semua feeder.")
