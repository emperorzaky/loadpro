# tuning.py v1.0
# ------------------------------------------------------------
# Entry-point utama untuk proses tuning hyperparameter LSTM
# Argumen: --feeder, --kategori, --method
# Memanggil metode tuning dari scripts/tuning/<method>_search.py
# ------------------------------------------------------------

import argparse
import time
import os
from datetime import datetime

# Mapping metode ke modul
METHOD_MAP = {
    "bayesopt": "bayesopt_search",
    # nanti bisa ditambah: "grid": "grid_search", dll
}

def main():
    parser = argparse.ArgumentParser(description="Tuning Hyperparameter LSTM untuk LOADPRO")
    parser.add_argument("--feeder", type=str, required=True, help="Nama penyulang (tanpa ekstensi)")
    parser.add_argument("--kategori", type=str, choices=["siang", "malam"], required=True, help="Kategori waktu")
    parser.add_argument("--method", type=str, choices=METHOD_MAP.keys(), required=True, help="Metode tuning")
    args = parser.parse_args()

    feeder = args.feeder.lower()
    kategori = args.kategori.lower()
    method = args.method.lower()

    start_time = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n🔧 Memulai tuning hyperparameter untuk {feeder} ({kategori})")
    print(f"🕒 Waktu mulai: {now}")
    print(f"⚙️  Metode: {method}\n")

    # Path ke file tuning
    try:
        module_name = METHOD_MAP[method]
        tuning_module = __import__(f"scripts.tuning.{module_name}", fromlist=["run_tuning"])
        tuning_module.run_tuning(feeder, kategori)
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat menjalankan metode '{method}': {e}")
        return

    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"\n✅ Tuning selesai untuk {feeder} ({kategori})")
    print(f"🕒 Total durasi: {minutes} menit {seconds} detik\n")

if __name__ == "__main__":
    main()
