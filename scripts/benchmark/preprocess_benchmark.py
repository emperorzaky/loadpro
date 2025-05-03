'''
PREPROCESS.PY v1.1
------------------------------
LOADPRO Project | Preprocessing Pipeline

Deskripsi:
- Membaca seluruh file .csv feeder dari folder data/raw
- Membersihkan data dari nilai 0 dan NaN
- Membuat kolom Timestamp gabungan dari Tanggal + Waktu (siang/malam)
- Membagi data menjadi file siang dan malam
- Menyimpan hasil preprocessing ke data/processed/split
- Menulis log aktivitas ke folder logs

Perubahan v1.1:
- Menghapus tqdm progress bar agar tampilan console/log lebih bersih dan profesional.
'''

import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

# --- Setup Logging ---
def setup_logger():
    """
    Membuat folder logs dan membuka file log baru.
    """
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = os.path.join(logs_dir, f"{timestamp}_preprocess.log")
    return open(log_file, "a")

# --- Helper Print ---
def log_print(message, logfile):
    """
    Menulis pesan ke console dan file log.
    """
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_message = f"{timestamp} {message}"
    print(full_message)
    logfile.write(full_message + "\n")

# --- Preprocessing Function ---
def preprocess_all_feeders():
    """
    Preprocessing semua file feeder:
    - Membersihkan
    - Membuat Timestamp
    - Membagi Siang/Malam
    - Save hasil split
    """
    start_time = time.time()
    logfile = setup_logger()

    try:
        # Setup folder
        raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
        split_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'split'))
        os.makedirs(split_dir, exist_ok=True)

        log_print(f"📁 Checking folder: {raw_dir} ...", logfile)

        all_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        feeder_count = len(all_files)
        log_print(f"🔍 Ditemukan {feeder_count} file penyulang untuk diproses.", logfile)

        if feeder_count == 0:
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            log_print(f"🎉 Preprocessing selesai untuk semua 0 file penyulang.", logfile)
            log_print(f"🕒 Total waktu eksekusi: {int(minutes)} menit {int(seconds)} detik", logfile)
            return

        for filename in all_files:
            try:
                log_print(f"🚀 Starting preprocessing: {filename}", logfile)
                file_path = os.path.join(raw_dir, filename)
                df = pd.read_csv(file_path)

                # Membuat kolom Timestamp
                df['Timestamp'] = pd.to_datetime(df['Tanggal'] + ' ' + df['Waktu'].map({'siang': '10:00:00', 'malam': '19:00:00'}), format='%m/%d/%Y %H:%M:%S')

                # Normalisasi: hapus 0 dan NaN
                initial_count = len(df)
                zeros_removed = (df['Beban'] == 0).sum()
                df = df[df['Beban'] != 0]
                nans_removed = df['Beban'].isna().sum()
                df = df.dropna(subset=['Beban'])
                final_count = len(df)

                log_print(f"🧹 Normalisasi data:\n   - 0 yang dihapus: {zeros_removed} baris\n   - NaN yang dihapus: {nans_removed} baris\n   - Jumlah awal: {initial_count} baris\n   - Jumlah akhir: {final_count} baris", logfile)

                # Membagi siang dan malam
                df['Jam'] = df['Timestamp'].dt.hour
                df_siang = df[df['Jam'] == 10]
                df_malam = df[df['Jam'] == 19]

                log_print(f"☀️ Membagi data Siang: {len(df_siang)} baris", logfile)
                log_print(f"🌙 Membagi data Malam: {len(df_malam)} baris", logfile)

                # Save hasil split
                feeder_name = filename.replace('.csv', '')
                df_siang[['Timestamp', 'Beban']].to_csv(os.path.join(split_dir, f"{feeder_name}_siang.csv"), index=False)
                df_malam[['Timestamp', 'Beban']].to_csv(os.path.join(split_dir, f"{feeder_name}_malam.csv"), index=False)

                log_print(f"💾 Menyimpan file siang: {feeder_name}_siang.csv", logfile)
                log_print(f"💾 Menyimpan file malam: {feeder_name}_malam.csv", logfile)
                log_print(f"✅ Preprocessing selesai untuk {filename}", logfile)
                log_print("-" * 60, logfile)

            except Exception as e:
                log_print(f"⚠️ ERROR saat memproses {filename}: {str(e)}", logfile)
                log_print("-" * 60, logfile)

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        log_print(f"🎉 Preprocessing selesai untuk semua {feeder_count} file.", logfile)
        log_print(f"🕒 Total waktu eksekusi: {int(minutes)} menit {int(seconds)} detik", logfile)
        log_print(f"📝 Log aktivitas disimpan di {logfile.name}", logfile)

    finally:
        logfile.close()

# --- Eksekusi langsung ---
if __name__ == "__main__":
    preprocess_all_feeders()
