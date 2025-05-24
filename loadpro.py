"""
loadpro.py v1.2

Deskripsi:
-----------
Entry point utama untuk menjalankan seluruh pipeline:
1. Preprocessing data
2. Training semua penyulang
3. Prediksi historis dan next-day
4. Rangkuman hasil prediksi next-day

Penggunaan:
-----------
    python3 loadpro.py

Output:
--------
- Menampilkan waktu mulai, proses, dan total durasi
- Menyimpan rangkuman prediksi ke:
  - logs/predict/YYYYMMDD_HHMM_loadpro_summary.log
  - results/predict/YYYYMMDD_predict_summary.csv

Author: Zaky Pradikto
"""

import subprocess
import time
import os
import pandas as pd
from datetime import datetime

start_time = time.time()
now_str = datetime.now().strftime("%Y%m%d_%H%M")
now_day = datetime.now().strftime("%Y%m%d")

print("\n⚡ Memulai LOADPRO pipeline...")
print("🕒 Start:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("----------------------------------------")

# Step 1: Preprocess
print("\n[1/3] 🔍 Menjalankan preprocessing...")
subprocess.run(["python3", "scripts/preprocess.py"], check=True)

# Step 2: Train all
print("\n[2/3] 🤖 Menjalankan training semua penyulang...")
subprocess.run(["python3", "scripts/train_all.py"], check=True)

# Step 3: Predict all
print("\n[3/3] 🔮 Menjalankan prediksi semua penyulang...")
subprocess.run(["python3", "scripts/predict_all.py"], check=True)

# Step 4: Summary log from next_*.csv
print("\n📝 Menyusun ringkasan prediksi next-day...")
result_dir = os.path.join("results", "predict")
log_dir = os.path.join("logs", "predict")
os.makedirs(log_dir, exist_ok=True)
summary_lines = []
summary_rows = []

for fname in sorted(os.listdir(result_dir)):
    if fname.startswith("next_") and fname.endswith(".csv"):
        try:
            df = pd.read_csv(os.path.join(result_dir, fname))
            row = df.iloc[0]
            feeder = row['feeder']
            kategori = row['kategori']
            y_pred = row['y_pred']
            line = f"Prediksi beban {kategori} berikutnya untuk {feeder} adalah {y_pred}"
            summary_lines.append(line)
            summary_rows.append({"feeder": feeder, "kategori": kategori, "y_pred": y_pred})
        except Exception as e:
            summary_lines.append(f"Gagal baca {fname}: {e}")

# Simpan ke log
summary_path = os.path.join(log_dir, f"{now_str}_loadpro_summary.log")
with open(summary_path, 'w') as f:
    for line in summary_lines:
        f.write(line + "\n")

# Simpan ke CSV
summary_csv = os.path.join(result_dir, f"{now_day}_predict_summary.csv")
if summary_rows:
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

# Selesai
total_time = time.time() - start_time
minutes = int(total_time // 60)
seconds = int(total_time % 60)

print("\n✅ LOADPRO selesai!")
print(f"🕒 Total waktu eksekusi: {minutes} menit {seconds} detik")
print(f"📄 Ringkasan hasil disimpan di: {summary_path}")
print(f"📊 Summary CSV disimpan di: {summary_csv}\n")
