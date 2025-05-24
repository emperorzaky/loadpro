"""
predict_all.py v1.0

Deskripsi:
-----------
Melakukan prediksi untuk seluruh penyulang yang tersedia:
1. Prediksi seluruh data historis (menggunakan predict.py)
2. Prediksi next-day (menggunakan predict_next.py)

Output:
--------
- Hasil batch prediksi: results/predict/{feeder}_{kategori}_pred.csv
- Hasil prediksi next-day: results/predict/next_{feeder}_{kategori}.csv
- Semua log disimpan di logs/predict/

Author: Zaky Pradikto
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import subprocess

# Path dasar
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
npz_dir = os.path.join(base_dir, 'data', 'npz')

# Ambil semua file .npz
files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
print(f"📂 Ditemukan {len(files)} file penyulang untuk diproses.\n")

# Loop semua penyulang
for file in files:
    feeder_kat = file.replace('.npz', '')
    parts = feeder_kat.split('_')
    feeder = '_'.join(parts[:-1])
    kategori = parts[-1]

    print(f"🔁 Memproses: {feeder}_{kategori}")

    # 1. Prediksi seluruh data historis
    try:
        subprocess.run([
            "python3", "scripts/predict.py",
            "--feeder", feeder,
            "--kategori", kategori
        ], check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Gagal prediksi historis {feeder}_{kategori}")

    # 2. Prediksi next-day
    try:
        subprocess.run([
            "python3", "scripts/predict_next.py",
            "--feeder", feeder,
            "--kategori", kategori
        ], check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Gagal prediksi next-day {feeder}_{kategori}")

    print("--------------------------------------")

print("🎉 Semua prediksi selesai!")
