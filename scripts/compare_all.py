# ===================================================
# COMPARE_ALL.PY v1.2
# ---------------------------------------------------
# Membandingkan seluruh model di models/temporary/
# dengan model final di models/single/, lalu memilih
# yang terbaik berdasarkan RMSE validasi.
# Hanya memproses model yang memiliki data .npz & scaler.
# ===================================================
# python3 scripts/compare_all.py

import os
import subprocess
import time
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

def main():
    start = time.time()
    os.makedirs("logs/compare", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", "compare", f"{timestamp}_compare_all.log")

    with open(log_path, "a") as logfile:
        def log(msg):
            ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            print(f"{ts} {msg}")
            logfile.write(f"{ts} {msg}\n")

        temp_models = [f for f in os.listdir("models/temporary") if f.endswith(".keras")]
        log(f"🔍 Ditemukan {len(temp_models)} model di models/temporary/")
        log("-" * 60)

        for filename in sorted(temp_models):
            try:
                parts = filename.replace(".keras", "").split("_")
                feeder = "_".join(parts[:-1])
                kategori = parts[-1]

                npz_path = f"data/npz/{feeder}_{kategori}.npz"
                pkl_path = f"data/metadata/{feeder}_{kategori}_scaler.pkl"

                if not os.path.exists(npz_path) or not os.path.exists(pkl_path):
                    log(f"⚠️  Lewati {feeder}_{kategori} karena data .npz atau scaler tidak ditemukan.")
                    continue

                log(f"🔬 Membandingkan: {feeder} ({kategori})")
                cmd = ["python3", "scripts/compare.py", "--feeder", feeder, "--kategori", kategori]
                subprocess.run(cmd, check=True)
                log("-" * 60)

            except Exception as e:
                log(f"❌ Gagal proses {filename}: {e}")

        dur = time.time() - start
        m, s = divmod(dur, 60)
        log(f"🎉 Selesai membandingkan semua model sementara.")
        log(f"🕒 Total waktu: {int(m)} menit {int(s)} detik")
        log(f"📄 Log tersimpan di: {log_path}")

if __name__ == "__main__":
    main()
