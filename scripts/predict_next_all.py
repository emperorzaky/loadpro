# ===================================================
# PREDICT_NEXT_ALL.PY v1.1
# ---------------------------------------------------
# Melakukan prediksi next day (H+1) hanya untuk feeder
# yang memiliki model .keras final di models/single/.
# ---------------------------------------------------
# Output disimpan ke: results/predict_next/
# Log dicatat di: logs/predict_next/
# ===================================================

import os
import time
import subprocess
from datetime import datetime

def main():
    start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs/predict_next"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{timestamp}_predict_next_all.log")

    with open(log_path, "w") as logfile:
        def log(msg):
            ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            line = f"{ts} {msg}"
            print(line)
            logfile.write(line + "\n")

        # Cari semua file .npz
        data_dir = "data/npz"
        npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
        log(f"📂 Ditemukan {len(npz_files)} file penyulang untuk diproses.")
        log("--------------------------------------------------------")

        for file in npz_files:
            try:
                nama = file.replace(".npz", "")
                feeder = "_".join(nama.split("_")[:-1])
                kategori = nama.split("_")[-1]
                model_path = f"models/single/{feeder}_{kategori}.keras"

                if not os.path.exists(model_path):
                    log(f"⚠️  Lewati {feeder}_{kategori} karena model belum tersedia.")
                    continue

                log(f"🔁 Memproses: {feeder} ({kategori})")

                result = subprocess.run([
                    "python3", "scripts/predict_next.py",
                    "--feeder", feeder,
                    "--kategori", kategori
                ], capture_output=True, text=True)

                if result.returncode != 0:
                    log(f"❌ Gagal prediksi: {feeder}_{kategori}")
                    log(f"    Pesan error: {result.stderr.strip()}")
                else:
                    log(f"✅ Sukses prediksi: {feeder}_{kategori}")

                log("--------------------------------------------------------")
            except Exception as e:
                log(f"❌ Error tak terduga saat memproses {file}: {e}")
                continue

        dur = time.time() - start
        m, s = divmod(dur, 60)
        log(f"🎉 Prediksi next day selesai.")
        log(f"🕒 Durasi total: {int(m)} menit {int(s)} detik")
        log(f"📄 Log disimpan di: {log_path}")

if __name__ == "__main__":
    main()
