# SUMMARY.PY v1.1
# ------------------------------
# LOADPRO Project | Mini Dashboard Tuning Summary
#
# Deskripsi:
# - Membaca hasil tuning dari logs (*.log)
# - Menampilkan rekap: feeder sukses, feeder gagal, MAPE, MAE
# - Progress bar monitoring total feeder
# - Dirancang dengan struktur modular dan siap produksi
#
# Perubahan v1.1:
# - Perbaikan struktur parsing log
# - Modularisasi fungsi baca dan parsing
# - Penambahan validasi robustness
# - Optimalisasi tampilan Streamlit

import os  # Modul untuk manipulasi path dan filesystem
import streamlit as st  # Modul utama untuk membuat web dashboard interaktif
from datetime import datetime  # Tidak digunakan saat ini tapi disiapkan untuk kemungkinan pengembangan timestamp

# --- Konfigurasi Path Direktori ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # Root folder dari project
LOGS_DIR = "/root/loadpro/logs"  # Lokasi log hasil tuning yang akan dibaca

# --- Fungsi untuk Membaca Semua File Log Tuning ---
def read_tuning_logs(logs_dir):
    logs = []
    if not os.path.exists(logs_dir):  # Validasi jika folder belum ada
        return logs

    for filename in os.listdir(logs_dir):
        if filename.endswith('_tuning.log'):  # Hanya baca file log yang sesuai pattern
            file_path = os.path.join(logs_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                logs.extend(f.readlines())  # Gabungkan semua baris dari seluruh file
    return logs

# --- Fungsi untuk Mem-parse Setiap Baris Log Menjadi Data Terstruktur ---
def parse_feeder_logs(log_lines):
    feeder_records = []
    current_feeder, success, mape, mae = None, False, None, None

    for line in log_lines:
        line = line.strip()

        # Menangkap nama feeder saat proses tuning dimulai
        if "Starting tuning:" in line:
            current_feeder = line.split(":")[-1].strip()
            success, mape, mae = False, None, None

        # Jika model berhasil dituning
        elif "✅ Best Params for" in line:
            success = True

        # Parsing nilai MAPE dan MAE dari baris log
        elif "MAPE" in line and "MAE" in line:
            try:
                parts = line.split("MAPE:")[1].split("MAE:")
                mape = float(parts[0].replace('%', '').strip())
                mae = float(parts[1].strip())
            except Exception:
                mape, mae = None, None  # Robust handling jika format tidak sesuai

        # Jika tuning gagal
        elif "ERROR saat tuning" in line:
            success = False

        # Saat menemukan pemisah log feeder, tambahkan ke list record
        elif "------------------------------------------------------------" in line and current_feeder:
            feeder_records.append({
                "Feeder": current_feeder,
                "Status": "✅ Sukses" if success else "❗ Gagal",
                "MAPE": mape,
                "MAE": mae
            })
            current_feeder = None

    return feeder_records

# --- Fungsi Utama Streamlit ---
def main():
    # Konfigurasi halaman dashboard
    st.set_page_config(page_title="LOADPRO Tuning Summary", page_icon="📈", layout="wide")
    st.title("📈 LOADPRO Tuning Summary Dashboard")

    # Tampilkan lokasi log
    st.info(f"📁 Membaca history tuning kita selama ini.")

    # Baca log dari folder
    logs = read_tuning_logs(LOGS_DIR)

    # Jika belum ada log ditemukan
    if not logs:
        st.warning("Belum ada log tuning ditemukan.")
        return

    # Parsing log menjadi data
    feeder_data = parse_feeder_logs(logs)

    if not feeder_data:
        st.warning("Log ditemukan, tetapi belum ada feeder yang diproses.")
        return

    # --- Rekapitulasi Ringkas ---
    total_feeder = len(feeder_data)
    sukses_count = sum(1 for feeder in feeder_data if feeder['Status'] == "✅ Sukses")
    gagal_count = total_feeder - sukses_count
    success_rate = int((sukses_count / total_feeder) * 100)

    st.subheader("📊 Rekapitulasi Tuning")
    st.progress(success_rate / 100)  # Visualisasi progress bar sukses

    # Tiga kolom metrik rekap
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Feeder", value=total_feeder)
    with col2:
        st.metric(label="✅ Sukses", value=sukses_count)
    with col3:
        st.metric(label="❗ Gagal", value=gagal_count)

    # --- Tabel Detail per Feeder ---
    st.subheader("🗂️ Detail Per Feeder")
    st.dataframe(feeder_data, use_container_width=True)  # Tampilkan tabel secara responsif

# --- Eksekusi ---
if __name__ == "__main__":
    main()
