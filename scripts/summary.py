# summary.py v1.2
# --------------------------------------------------
# Menyusun ringkasan prediksi next-day dari hasil predict_next
# Format input: results/predict_next/next_*.txt
# Output: logs/summary/summary_YYYYMMDD_HHMM.log dan results/summary/*.csv
# --------------------------------------------------

import os
import time
import pandas as pd
from datetime import datetime

start_time = time.time()
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M")
today = now.strftime("%Y-%m-%d")

result_dir = "results/predict_next"
log_dir = "logs/summary"
csv_dir = "results/summary"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

summary_rows = []
summary_log_lines = []

print("\n================== RINGKASAN HASIL PREDIKSI ==================\n")

for fname in sorted(os.listdir(result_dir)):
    if fname.startswith("next_") and fname.endswith(".txt"):
        path = os.path.join(result_dir, fname)
        try:
            with open(path, "r") as f:
                lines = f.readlines()
                if len(lines) < 5:
                    raise ValueError("File tidak memiliki 5 baris sesuai format.")

                feeder = lines[1].split(":")[1].strip()
                waktu = lines[2].split(":")[1].strip()
                tanggal = lines[3].split(":")[1].strip()
                beban_str = lines[4].split(":")[1].strip().split()[0]  # Ambil nilai sebelum 'A'
                ampere = float(beban_str)

                summary_rows.append({
                    "Tanggal": tanggal,
                    "Penyulang": feeder,
                    "Waktu": waktu,
                    "Prediksi_Ampere": ampere
                })

        except Exception as e:
            print(f"⚠️  Gagal membaca {fname}: {e}")

if summary_rows:
    df = pd.DataFrame(summary_rows)
    df.sort_values(by=["Tanggal", "Penyulang", "Waktu"], inplace=True)

    # Tampilkan ke terminal
    print(df.to_string(index=False))

    # Simpan ke log file dan CSV
    log_path = os.path.join(log_dir, f"summary_{timestamp}.log")
    with open(log_path, "w") as f:
        f.write(df.to_string(index=False))

    csv_path = os.path.join(csv_dir, f"summary_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
else:
    print("⚠️  Tidak ada data rekap ditemukan.")

# Total waktu
total_time = time.time() - start_time
minutes = int(total_time // 60)
seconds = int(total_time % 60)
print(f"\n🕒 Total waktu eksekusi: {minutes} menit {seconds} detik")
print(f"📄 Ringkasan log: {log_path if summary_rows else '-'}")
print(f"📊 Ringkasan CSV: {csv_path if summary_rows else '-'}\n")
