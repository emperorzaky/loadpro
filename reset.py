'''
RESET.PY v1.4
------------------------------
LOADPRO Project | Reset Utility

Deskripsi:
- Menghapus seluruh data hasil preprocessing di data/processed/split
- Menghapus seluruh model hasil tuning di models/single
- Menghapus hasil prediksi di folder results/
- Menghapus seluruh file log di folder logs/
- Menghapus file input harian di folder loadpro/input/
- Menampilkan log aktivitas yang rapi dan profesional
'''

import os
import shutil
from datetime import datetime

# --- Setup path ---
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
split_dir = os.path.join(base_dir, 'data', 'processed', 'split')
model_dir = os.path.join(base_dir, 'models', 'single')
results_dir = os.path.join(base_dir, 'results')
logs_dir = os.path.join(base_dir, 'logs')
input_dir = os.path.join(base_dir, 'input')  # Ini adalah loadpro/input

# --- Format print ---
def log_action(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# --- Utility untuk menghapus isi folder ---
def clear_folder(folder_path, label):
    log_action(f"🔄 Menghapus isi folder: {label}")
    deleted_count = 0
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    deleted_count += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    deleted_count += 1
            except Exception as e:
                log_action(f"❌ Gagal menghapus {file_path}. Alasan: {e}")
    else:
        log_action(f"📂 Folder tidak ditemukan: {folder_path}")
        return

    log_action(f"🧾 Total file/direktori yang dihapus dari {label}: {deleted_count}")

# --- Eksekusi utama ---
if __name__ == "__main__":
    clear_folder(split_dir, 'data/processed/split')
    clear_folder(model_dir, 'models/single')
    clear_folder(results_dir, 'results')
    clear_folder(logs_dir, 'logs')
    clear_folder(input_dir, 'input')  # Folder loadpro/input

    log_action("✅ Reset selesai. Semua data split, model, hasil prediksi, input harian, dan log telah dibersihkan.")
