'''
RESET.PY v1.3
------------------------------
LOADPRO Project | Reset Utility

Deskripsi:
- Menghapus seluruh data hasil preprocessing di data/processed/split
- Menghapus seluruh model hasil tuning di models/single
- Menghapus hasil prediksi di folder results/
- Menghapus seluruh file log di folder logs/
- Menampilkan log aktivitas yang rapi dan profesional
'''

# Import modul os untuk manajemen direktori dan path
import os
# Import shutil untuk operasi penghapusan folder secara rekursif
import shutil
# Import datetime untuk mencetak timestamp log
from datetime import datetime

# --- Setup path ---

# Menentukan direktori dasar sebagai lokasi file skrip saat ini
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Path ke folder split hasil preprocessing
split_dir = os.path.join(base_dir, 'data', 'processed', 'split')

# Path ke folder model hasil tuning
model_dir = os.path.join(base_dir, 'models', 'single')

# Path ke folder hasil prediksi
results_dir = os.path.join(base_dir, 'results')

# Path ke folder log aktivitas
logs_dir = os.path.join(base_dir, 'logs')

# --- Format print ---

# Fungsi untuk mencetak log dengan timestamp ke terminal
def log_action(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format timestamp standar
    print(f"[{timestamp}] {message}")  # Output log ke console

# --- Utility untuk menghapus isi folder ---

# Fungsi untuk menghapus semua file dan subfolder dalam direktori tertentu
def clear_folder(folder_path, label):
    log_action(f"🔄 Menghapus isi folder: {label}")  # Log aktivitas awal
    deleted_count = 0  # Inisialisasi counter file/direktori yang terhapus

    if os.path.exists(folder_path):  # Periksa apakah folder ada
        for filename in os.listdir(folder_path):  # Iterasi semua file dalam folder
            file_path = os.path.join(folder_path, filename)  # Bangun path absolut
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):  # Jika file biasa atau symlink
                    os.unlink(file_path)  # Hapus file
                    deleted_count += 1
                elif os.path.isdir(file_path):  # Jika direktori
                    shutil.rmtree(file_path)  # Hapus folder dan isinya
                    deleted_count += 1
            except Exception as e:
                # Tangani kegagalan penghapusan dan tampilkan alasannya
                log_action(f"❌ Gagal menghapus {file_path}. Alasan: {e}")
    else:
        # Jika folder tidak ada, log informasi
        log_action(f"📂 Folder tidak ditemukan: {folder_path}")
        return

    # Log total file/direktori yang berhasil dihapus
    log_action(f"🧾 Total file/direktori yang dihapus dari {label}: {deleted_count}")

# --- Eksekusi utama ---

# Jika script dijalankan langsung, bukan diimpor sebagai modul
if __name__ == "__main__":
    # Bersihkan masing-masing folder sesuai perannya
    clear_folder(split_dir, 'data/processed/split')
    clear_folder(model_dir, 'models/single')
    clear_folder(results_dir, 'results')
    clear_folder(logs_dir, 'logs')

    # Cetak log penutup bahwa semua proses reset telah selesai
    log_action("✅ Reset selesai. Semua data split, model, hasil prediksi, dan log telah dibersihkan.")
