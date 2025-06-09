# ===================================================
# TRAIN_ALL.PY v1.1
# ---------------------------------------------------
# LOADPRO Project | Batch training semua penyulang
#
# Fitur:
# - Jalankan train.py untuk semua .npz di data/npz
# - Lewati model yang sudah ada (default)
# - Gunakan --overwrite untuk latih ulang
# - Gunakan --output untuk simpan model ke folder lain
#
# Contoh:
# $ python3 scripts/train_all.py
# $ python3 scripts/train_all.py --overwrite
# $ python3 scripts/train_all.py --output models/temporary/
# ===================================================

import os
import argparse
import subprocess

# --- Argument parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', help='Force retrain all even if model exists')
parser.add_argument('--output', default='models/temporary', help='Output folder untuk model .keras')
args = parser.parse_args()

# --- Lokasi folder ---
data_dir = 'data/npz'
model_dir = args.output
os.makedirs(model_dir, exist_ok=True)

# --- Cari semua file .npz ---
npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
print(f"\n📦 Menemukan {len(npz_files)} file .npz")
print("-" * 42)

# --- Loop semua file ---
for filename in npz_files:
    feeder_kat = filename.replace('.npz', '')
    feeder, kategori = feeder_kat.rsplit('_', 1)
    model_path = os.path.join(model_dir, f"{feeder}_{kategori}.keras")

    if not args.overwrite and os.path.exists(model_path):
        print(f"⏩ Melewati: {feeder}_{kategori} (model sudah ada)")
        continue

    print(f"🚀 Training {feeder}_{kategori}...")
    try:
        cmd = [
            'python3', 'scripts/train.py',
            '--feeder', feeder,
            '--kategori', kategori,
            '--output', model_dir
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ GAGAL training {feeder}_{kategori}: {e}")
    print("-" * 42)

print("\n🎉 Selesai training semua penyulang.")
