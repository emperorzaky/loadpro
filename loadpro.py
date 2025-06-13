# ===================================================
# LOADPRO v2.0
# ---------------------------------------------------
# Menjalankan seluruh pipeline: preprocess → train →
# compare → predict → predict next day.
# Akhirnya membuat summary file rekap beban H+1.
# ===================================================

import subprocess
import time
from datetime import datetime
import os
import re
import csv

# Buat folder log jika belum ada
os.makedirs("logs/loadpro", exist_ok=True)
os.makedirs("rekap", exist_ok=True)

start = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_log_path = f"logs/loadpro/{timestamp}_loadpro.log"
rekap_path = "rekap/rekap_nextday.csv"

with open(summary_log_path, "w") as log:
    def log_print(msg):
        print(msg)
        log.write(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {msg}\n")

    def run_script(name, args=[]):
        log_print(f"🚀 Menjalankan {name}...")
        result = subprocess.run(["python3", f"scripts/{name}.py"] + args, capture_output=True, text=True)
        log_print(result.stdout)
        if result.stderr:
            log_print(result.stderr)
        log_print("------------------------------------------------------------")

    # Jalankan pipeline satu per satu
    run_script("preprocess")
    run_script("train_all")
    run_script("compare_all")
    run_script("predict_all")
    run_script("predict_next_all")

    # Buat file rekap beban H+1 dari hasil .txt
    log_print("📋 Membuat rekap prediksi beban H+1...")
    result_dir = "results/predict_next"
    rows = [("Tanggal", "Hari", "Penyulang", "Kategori", "Beban (A)")]

    for fname in sorted(os.listdir(result_dir)):
        if fname.endswith(".txt") and fname.startswith("next_"):
            try:
                with open(os.path.join(result_dir, fname), "r") as f:
                    content = f.read()

                penyulang = re.search(r"Penyulang\s*:\s*(.+)", content).group(1).strip()
                kategori = re.search(r"Kategori\s*:\s*(.+)", content).group(1).strip()
                tanggal_str = re.search(r"Tanggal\s*:\s*(.+)", content).group(1).strip()
                beban = re.search(r"Beban\s*:\s*([\d.]+)", content).group(1).strip()

                tanggal_dt = datetime.strptime(tanggal_str, "%A, %d %B %Y")
                tanggal_fmt = tanggal_dt.strftime("%Y-%m-%d")
                hari = tanggal_dt.strftime("%A")

                rows.append((tanggal_fmt, hari, penyulang, kategori, beban))

            except Exception as e:
                log_print(f"⚠️ Gagal membaca {fname}: {e}")
                continue

    with open(rekap_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    log_print(f"✅ Rekap prediksi disimpan di: {rekap_path}")

    # Total durasi
    dur = time.time() - start
    m, s = divmod(dur, 60)
    log_print(f"🎉 Pipeline selesai dalam {int(m)} menit {int(s)} detik.")
    log_print(f"📄 Log lengkap: {summary_log_path}")
