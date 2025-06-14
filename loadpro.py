# loadpro.py v2.2
# --------------------------------------------------
# Entry point utama untuk menjalankan seluruh pipeline:
# 1. Preprocessing data
# 2. Training semua penyulang
# 3. Pembandingan model hasil training
# 4. Prediksi historis (all)
# 5. Prediksi next-day (all)
# 6. Ringkasan hasil prediksi
# --------------------------------------------------

import subprocess
import time
from datetime import datetime

start_time = time.time()
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
print("\n⚡ Memulai LOADPRO pipeline...")
print("🕒 Start:", timestamp)
print("--------------------------------------------------")

# Langkah 1: Preprocessing
print("\n[1/6] 🔍 Menjalankan preprocessing...")
subprocess.run(["python3", "scripts/preprocess.py"], check=True)

# Langkah 2: Training
print("\n[2/6] 🤖 Menjalankan training semua penyulang...")
subprocess.run(["python3", "scripts/train_all.py"], check=True)

# Langkah 3: Compare model
print("\n[3/6] ⚖️  Membandingkan model sementara dan final...")
subprocess.run(["python3", "scripts/compare_all.py"], check=True)

# Langkah 4: Prediksi historis
print("\n[4/6] 📈 Menjalankan prediksi historis semua penyulang...")
subprocess.run(["python3", "scripts/predict_all.py"], check=True)

# Langkah 5: Prediksi next-day
print("\n[5/6] 🔮 Menjalankan prediksi next-day semua penyulang...")
subprocess.run(["python3", "scripts/predict_next_all.py"], check=True)

# Langkah 6: Ringkasan hasil prediksi
print("\n[6/6] 🧾 Menyusun ringkasan prediksi ke terminal & CSV...")
subprocess.run(["python3", "scripts/summary.py"], check=True)

# Total waktu eksekusi
total_time = time.time() - start_time
minutes = int(total_time // 60)
seconds = int(total_time % 60)
print("\n✅ LOADPRO selesai!")
print(f"🕒 Total waktu eksekusi: {minutes} menit {seconds} detik\n")
