# ===================================================
# PREPROCESS.PY v1.3
# ---------------------------------------------------
# LOADPRO Project | Preprocessing Pipeline
#
# Deskripsi:
# - Membaca seluruh file .csv feeder dari folder data/raw
# - Membersihkan data dari nilai 0 dan NaN
# - Membuat kolom Timestamp gabungan dari Tanggal + Waktu (siang/malam)
# - Membagi data menjadi file siang dan malam
# - Menyimpan hasil preprocessing ke data/processed/split
# - Menulis log aktivitas ke folder logs (stdout dan stderr terpisah)
#
# Perubahan v1.3:
# - Output log preprocessing tetap tampil di terminal
# - Error internal dari TensorFlow (stderr) dialihkan ke file logs terpisah
# ===================================================

import os                    # Untuk manipulasi path dan direktori
import sys                   # Untuk redirect error ke file
import pandas as pd          # Untuk baca dan proses file CSV
import numpy as np           # Untuk operasi numerik dasar
import time                  # Untuk hitung waktu proses
from datetime import datetime  # Untuk timestamp log

# --- Setup Logging ---
def setup_logger():
    # Tentukan folder log dan buat jika belum ada
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    os.makedirs(logs_dir, exist_ok=True)

    # Buat nama file log berbasis waktu
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    stdout_log = os.path.join(logs_dir, f"{timestamp}_preprocess.log")
    stderr_log = os.path.join(logs_dir, f"{timestamp}_preprocess_error.log")

    # Alihkan semua error (stderr) ke file khusus
    sys.stderr = open(stderr_log, "a")

    # Kembalikan file objek log stdout
    return open(stdout_log, "a"), stderr_log

# --- Helper Print ---
def log_print(message, logfile):
    # Format timestamp standar
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_message = f"{timestamp} {message}"

    # Tampilkan ke console dan tulis ke file log
    print(full_message)
    logfile.write(full_message + "\n")

# --- Preprocessing Function ---
def preprocess_all_feeders():
    start_time = time.time()  # Mulai hitung waktu proses
    logfile, stderr_path = setup_logger()  # Setup logging

    try:
        # Setup folder input dan output
        raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
        split_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'split'))
        os.makedirs(split_dir, exist_ok=True)

        # Tampilkan isi folder raw
        log_print(f"📁 Checking folder: {raw_dir} ...", logfile)

        # Ambil semua file .csv feeder
        all_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        feeder_count = len(all_files)
        log_print(f"🔍 Ditemukan {feeder_count} file penyulang untuk diproses.", logfile)

        # Jika tidak ada file .csv
        if feeder_count == 0:
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            log_print(f"🎉 Preprocessing selesai untuk semua 0 file penyulang.", logfile)
            log_print(f"🕒 Total waktu eksekusi: {int(minutes)} menit {int(seconds)} detik", logfile)
            return

        # Loop semua file feeder
        for filename in all_files:
            try:
                log_print(f"🚀 Starting preprocessing: {filename}", logfile)
                file_path = os.path.join(raw_dir, filename)
                df = pd.read_csv(file_path)  # Baca file CSV

                # Gabungkan tanggal + waktu jadi Timestamp (jam diset manual)
                df['Timestamp'] = pd.to_datetime(
                    df['Tanggal'] + ' ' + df['Waktu'].map({'siang': '10:00:00', 'malam': '19:00:00'}),
                    format='%m/%d/%Y %H:%M:%S'
                )

                # Bersihkan data: hapus nilai 0 dan NaN
                initial_count = len(df)
                zeros_removed = (df['Beban'] == 0).sum()
                df = df[df['Beban'] != 0]
                nans_removed = df['Beban'].isna().sum()
                df = df.dropna(subset=['Beban'])
                final_count = len(df)

                # Tulis log hasil normalisasi
                log_print(
                    f"🧹 Normalisasi data:\n   - 0 yang dihapus: {zeros_removed} baris\n   - NaN yang dihapus: {nans_removed} baris\n   - Jumlah awal: {initial_count} baris\n   - Jumlah akhir: {final_count} baris",
                    logfile
                )

                # Pisahkan data siang dan malam berdasarkan jam Timestamp
                df['Jam'] = df['Timestamp'].dt.hour
                df_siang = df[df['Jam'] == 10]
                df_malam = df[df['Jam'] == 19]

                # Log jumlah baris masing-masing
                log_print(f"☀️ Membagi data Siang: {len(df_siang)} baris", logfile)
                log_print(f"🌙 Membagi data Malam: {len(df_malam)} baris", logfile)

                # Simpan hasil split ke folder split/
                feeder_name = filename.replace('.csv', '')
                df_siang[['Timestamp', 'Beban']].to_csv(os.path.join(split_dir, f"{feeder_name}_siang.csv"), index=False)
                df_malam[['Timestamp', 'Beban']].to_csv(os.path.join(split_dir, f"{feeder_name}_malam.csv"), index=False)

                # Log proses penyimpanan
                log_print(f"💾 Menyimpan file siang: {feeder_name}_siang.csv", logfile)
                log_print(f"💾 Menyimpan file malam: {feeder_name}_malam.csv", logfile)
                log_print(f"✅ Preprocessing selesai untuk {filename}", logfile)
                log_print("-" * 60, logfile)

            except Exception as e:
                # Tangani error spesifik pada satu file
                log_print(f"⚠️ ERROR saat memproses {filename}: {str(e)}", logfile)
                log_print("-" * 60, logfile)

        # Log waktu total eksekusi preprocessing
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        log_print(f"🎉 Preprocessing selesai untuk semua {feeder_count} file.", logfile)
        log_print(f"🕒 Total waktu eksekusi: {int(minutes)} menit {int(seconds)} detik", logfile)
        log_print(f"📝 Log aktivitas disimpan di {logfile.name}", logfile)
        log_print(f"🪵 Error internal (jika ada) dicatat di {stderr_path}", logfile)

    finally:
        logfile.close()  # Tutup file log utama di akhir

# --- Eksekusi langsung ---
if __name__ == "__main__":
    preprocess_all_feeders()  # Jalankan fungsi utama jika script dieksekusi langsung
