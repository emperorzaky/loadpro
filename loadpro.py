# ===================================================
# LOADPRO v2.0
# ---------------------------------------------------
# Pipeline otomatis: preprocess → train → compare →
# predict → predict next, lalu buat summary.
# ===================================================

import subprocess
import time
import os
from datetime import datetime

def run_script(script_path):
    result = subprocess.run(["python3", script_path])
    return result.returncode == 0

def log_summary(summary_lines):
    os.makedirs("logs/loadpro", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    summary_file = f"logs/loadpro/{timestamp}_summary.log"
    with open(summary_file, "w") as f:
        for line in summary_lines:
            f.write(line + "\n")
    print(f"\n📄 Summary pipeline disimpan di: {summary_file}\n")

if __name__ == "__main__":
    start = time.time()
    summary = []

    summary.append("🚀 LOADPRO v2.0 Pipeline Summary")
    summary.append("----------------------------------------")

    # Step 1: Preprocessing
    summary.append("1️⃣  Menjalankan preprocessing...")
    status = run_script("scripts/preprocess.py")
    summary.append("    ✅ Selesai" if status else "    ❌ Gagal")

    # Step 2: Train All
    summary.append("2️⃣  Menjalankan training semua model...")
    status = run_script("scripts/train_all.py")
    summary.append("    ✅ Selesai" if status else "    ❌ Gagal")

    # Step 3: Compare All
    summary.append("3️⃣  Membandingkan model sementara dengan model final...")
    status = run_script("scripts/compare_all.py")
    summary.append("    ✅ Selesai" if status else "    ❌ Gagal")

    # Step 4: Predict All
    summary.append("4️⃣  Melakukan prediksi seluruh data historis...")
    status = run_script("scripts/predict_all.py")
    summary.append("    ✅ Selesai" if status else "    ❌ Gagal")

    # Step 5: Predict Next All
    summary.append("5️⃣  Melakukan prediksi next day untuk semua penyulang...")
    status = run_script("scripts/predict_next_all.py")
    summary.append("    ✅ Selesai" if status else "    ❌ Gagal")

    # Duration
    dur = time.time() - start
    m, s = divmod(dur, 60)
    summary.append("----------------------------------------")
    summary.append(f"🕒 Total waktu eksekusi: {int(m)} menit {int(s)} detik")

    log_summary(summary)
