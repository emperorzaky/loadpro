# ===================================================
# PREDICT_ALL.PY v1.4
# ---------------------------------------------------
# Melakukan prediksi seluruh data validasi (bukan H+1)
# hanya untuk penyulang yang memiliki model .keras.
# Menampilkan info jika model belum tersedia.
# ---------------------------------------------------
# Output disimpan ke: results/predict/
# Log dicatat di: logs/predict/
# ===================================================

import os
import time
import subprocess
from datetime import datetime

def main():
    start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs/predict"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{timestamp}_predict_all.log")

    with open(log_path, "w") as logfile:
        def log_print(msg):
            ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            line = f"{ts} {msg}"
            print(line)
            logfile.write(line + "\n")

        # Ambil semua file .npz sebagai acuan daftar penyulang
        npz_dir = "data/npz"
        npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
        log_print(f"📂 Ditemukan {len(npz_files)} file penyulang untuk diproses.")
        log_print("--------------------------------------------------------")

        for file in npz_files:
            basename = file.replace(".npz", "")
            feeder = "_".join(basename.split("_")[:-1])
            kategori = basename.split("_")[-1]
            model_path = f"models/single/{basename}.keras"

            if not os.path.exists(model_path):
                log_print(f"⚠️  Lewati {basename} karena model belum tersedia.")
                continue

            try:
                log_print(f"🔁 Memproses: {feeder} ({kategori})")
                result = subprocess.run([
                    "python3", "scripts/predict.py",
                    "--feeder", feeder,
                    "--kategori", kategori
                ], capture_output=True, text=True)

                if result.returncode != 0:
                    log_print(f"❌ Gagal prediksi: {basename}")
                    log_print(f"    Pesan error: {result.stderr.strip()}")
                else:
                    log_print(f"✅ Sukses prediksi: {basename}")

                log_print("--------------------------------------------------------")

            except Exception as e:
                log_print(f"❌ Error saat memproses {basename}: {e}")
                continue

        dur = time.time() - start
        m, s = divmod(dur, 60)
        log_print(f"🎉 Selesai memproses semua prediksi.")
        log_print(f"🕒 Total waktu: {int(m)} menit {int(s)} detik")
        log_print(f"📄 Log tersimpan di: {log_path}")

if __name__ == "__main__":
    main()
