"""
reset.py v1.1

Deskripsi:
-----------
Utility script untuk menghapus seluruh hasil preprocessing, model, hasil prediksi, dan log.
Digunakan untuk membersihkan seluruh pipeline dan mulai ulang dari awal.

Perubahan v1.1:
---------------
+ Tambahkan penghapusan isi folder:
    - models/tuning/
    - results/tuning/
    - logs/tuning/

Penggunaan:
-----------
    python3 reset.py

Author: Zaky Pradikto
"""

import os
import shutil

# Daftar folder yang akan dihapus isinya (bukan foldernya)
targets = [
    'data/npz',
    'data/metadata',
    'models/single',
    'models/tuning',
    'results/predict',
    'results/tuning',
    'logs/preprocess',
    'logs/train',
    'logs/predict',
    'logs/tuning',
    'logs/validator'
]

# Jalankan pembersihan
print("⚠️  Memulai reset seluruh hasil...\n")
for folder in targets:
    path = os.path.abspath(folder)
    if os.path.exists(path):
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            try:
                if os.path.isfile(fp) or os.path.islink(fp):
                    os.unlink(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
            except Exception as e:
                print(f"❌ Gagal hapus {fp}: {e}")
        print(f"✅ Kosongkan: {folder}")
    else:
        print(f"ℹ️  Folder tidak ditemukan: {folder}")

print("\n🧼 Reset selesai. Semua hasil telah dihapus.")
