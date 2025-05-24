# ===================================================
# PREPROCESS.PY v1.6.1
# ---------------------------------------------------
# LOADPRO Project | Preprocessing 1000+ Feeder Skala Besar
#
# Output utama:
# - .npz hasil windowing (X, y) per feeder per waktu (siang/malam)
# - .pkl scaler per feeder per waktu
# - Log terminal + file (stdout & stderr)
#
# Patch v1.6.1:
# - Fix: Kolom 'Waktu' berisi label 'siang/malam', bukan jam
# - Mapping otomatis: 'siang' → '10:00:00', 'malam' → '19:00:00'
# - Logging distribusi jam unik untuk debug
# ===================================================

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- Setup Logging ---
def setup_logger():
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'preprocess'))
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    stdout_log = os.path.join(log_dir, f"{ts}_preprocess.log")
    stderr_log = os.path.join(log_dir, f"{ts}_preprocess_error.log")
    sys.stderr = open(stderr_log, "a")  # redirect internal error
    return open(stdout_log, "a"), stderr_log

def log_print(msg, logfile):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line)
    logfile.write(line + "\n")

# --- Simpan .npz dan scaler ---
def save_npz_and_scaler(feeder, kategori, X, y, scaler, npz_dir, meta_dir):
    npz_name = f"{feeder}_{kategori}.npz"
    scaler_name = f"{feeder}_{kategori}_scaler.pkl"
    np.savez_compressed(os.path.join(npz_dir, npz_name), X=X, y=y)
    joblib.dump(scaler, os.path.join(meta_dir, scaler_name))

# --- Proses 1 file feeder ---
def preprocess_file(file_path, npz_dir, meta_dir, logfile):
    feeder = os.path.splitext(os.path.basename(file_path))[0]
    log_print(f"🚀 Mulai: {feeder}", logfile)

    try:
        df = pd.read_csv(file_path)
        awal = len(df)

        # Hapus 'fail', 0, dan NaN
        fail_count = (df['Beban'] == 'fail').sum()
        df = df[df['Beban'] != 'fail']
        df['Beban'] = pd.to_numeric(df['Beban'], errors='coerce')
        zero_count = (df['Beban'] == 0).sum()
        nan_count = df['Beban'].isna().sum()
        df = df[(df['Beban'] != 0) & (~df['Beban'].isna())]
        akhir = len(df)

        log_print(f"🧹 Pembersihan data:", logfile)
        log_print(f"   - 'fail' dihapus: {fail_count}", logfile)
        log_print(f"   - 0 dihapus: {zero_count}", logfile)
        log_print(f"   - NaN dihapus: {nan_count}", logfile)
        log_print(f"   - Sisa baris: {akhir} dari {awal}", logfile)

        # --- PATCH: Mapping 'siang/malam' ke waktu jam ---
        df['Waktu'] = df['Waktu'].map({'siang': '10:00:00', 'malam': '19:00:00'})
        df['Timestamp'] = pd.to_datetime(df['Tanggal'] + ' ' + df['Waktu'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])

        # Ambil jam dari timestamp
        df['Jam'] = df['Timestamp'].dt.hour
        log_print(f"   - Distribusi jam unik: {sorted(df['Jam'].unique())}", logfile)
        df['Kategori'] = df['Jam'].apply(lambda x: 'siang' if x == 10 else ('malam' if x == 19 else 'ignore'))

        for kategori in ['siang', 'malam']:
            subset = df[df['Kategori'] == kategori]
            log_print(f"   - Jumlah data kategori '{kategori}': {len(subset)}", logfile)
            if len(subset) < 10:
                log_print(f"⚠️  {feeder} kategori {kategori} kurang dari 10 baris, dilewati.", logfile)
                continue

            values = subset['Beban'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(values)

            window = 5
            X, y = [], []
            for i in range(window, len(scaled)):
                X.append(scaled[i - window:i])
                y.append(scaled[i][0])
            X = np.array(X)
            y = np.array(y)

            save_npz_and_scaler(feeder, kategori, X, y, scaler, npz_dir, meta_dir)
            log_print(f"💾 Disimpan: {feeder}_{kategori}.npz ({len(X)} sampel)", logfile)

        log_print(f"✅ Selesai: {feeder}", logfile)
        log_print("-" * 60, logfile)

    except Exception as e:
        log_print(f"❌ ERROR saat proses {feeder}: {str(e)}", logfile)
        log_print("-" * 60, logfile)

# --- Proses semua file ---
def main():
    start = time.time()
    logfile, stderr_path = setup_logger()

    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    npz_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'npz'))
    meta_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'metadata'))
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    log_print(f"📁 Membaca folder: {raw_dir}", logfile)
    files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    log_print(f"🔍 {len(files)} file ditemukan.", logfile)
    log_print("-" * 60, logfile)

    for f in files:
        path = os.path.join(raw_dir, f)
        preprocess_file(path, npz_dir, meta_dir, logfile)

    dur = time.time() - start
    m, s = divmod(dur, 60)
    log_print(f"🎉 Selesai memproses semua {len(files)} file.", logfile)
    log_print(f"🕒 Total waktu: {int(m)} menit {int(s)} detik", logfile)
    log_print(f"📝 Log aktivitas: {logfile.name}", logfile)
    log_print(f"🪵 Error internal dicatat di: {stderr_path}", logfile)
    logfile.close()

# --- Entry point ---
if __name__ == "__main__":
    main()
