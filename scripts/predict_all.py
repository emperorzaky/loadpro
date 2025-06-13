# ===================================================
# PREDICT_ALL.PY v1.3
# ---------------------------------------------------
# Menjalankan prediksi hanya untuk penyulang yang:
# (a) memiliki file .npz di data/npz/
# (b) memiliki model .keras di models/single/
# Hasil disimpan di results/predict/ dan log dicatat ke logs/predict/
# ===================================================

import os
import time
import subprocess
from datetime import datetime

# --- Setup Log ---
def setup_logger():
    log_dir = os.path.join("logs", "predict")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{ts}_predict_all.log")
    return open(log_path, "a")

def log_print(msg, logfile):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line)
    logfile.write(line + "\n")

# --- Main Function ---
def main():
    start = time.time()
    logfile = setup_logger()

    # Ambil semua file .npz dan .keras
    npz_files = sorted([f.replace(".npz", "") for f in os.listdir("data/npz") if f.endswith(".npz")])
    keras_files = sorted([f.replace(".keras", "") for f in os.listdir("models/single") if f.endswith(".keras")])

    # Cari irisan antara .npz dan .keras
    feeders_to_predict = sorted(set(npz_files).intersection(set(keras_files)))

    log_print(f"📂 Tersedia {len(feeders_to_predict)} penyulang untuk diprediksi.", logfile)
    log_print("--------------------------------------------------------", logfile)

    for basename in feeders_to_predict:
        try:
            parts = basename.split("_")
            feeder = "_".join(parts[:-1])
            kategori = parts[-1]

            log_print(f"🔁 Memproses: {feeder} ({kategori})", logfile)

            result = subprocess.run([
                "python3", "scripts/predict.py",
                "--feeder", feeder,
                "--kategori", kategori
            ], capture_output=True, text=True)

            # Cetak output predict.py ke log
            log_print(result.stdout.strip(), logfile)
            if result.stderr.strip():
                log_print(f"⚠️ STDERR: {result.stderr.strip()}", logfile)

        except Exception as e:
            log_print(f"❌ Gagal memproses {basename}: {e}", logfile)
        log_print("--------------------------------------------------------", logfile)

    dur = time.time() - start
    m, s = divmod(dur, 60)
    log_print(f"🎉 Selesai memproses semua prediksi.", logfile)
    log_print(f"🕒 Total waktu: {int(m)} menit {int(s)} detik", logfile)
    log_print(f"📄 Log tersimpan di: {logfile.name}", logfile)
    logfile.close()

if __name__ == "__main__":
    main()
