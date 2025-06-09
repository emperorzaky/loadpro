🔄 train_all.py — Melatih semua feeder sekaligus (batch training)
📄 Fungsi:
Melakukan looping ke semua file .npz di data/npz/ dan memanggil train.py untuk masing-masing penyulang dan kategori.

🧩 Argumen:

--overwrite	❌ Opsional	Paksa latih ulang walaupun model sudah ada
--output	❌ Opsional	Folder untuk simpan semua file .keras, default: models/single/

🔁 Contoh pemakaian:

# Training semua file .npz, skip yang sudah ada:
python3 scripts/train_all.py

# Training ulang semua (timpa):
python3 scripts/train_all.py --overwrite

# Simpan model ke folder sementara:
python3 scripts/train_all.py --output models/temporary/

# Simpan ke folder lain dan timpa model lama:
python3 scripts/train_all.py --overwrite --output models/temporary/

💡 Logika Tambahan:
Bila model .keras sudah ada dan --overwrite tidak diberikan, maka model diskip.

Bila --overwrite diberikan, maka file lama ditimpa.

📁 Folder Output yang Terkait

data/npz/           Tempat semua dataset hasil preprocessing
data/metadata/	    Scaler MinMaxScaler .pkl
models/single/	    Tempat default untuk file .keras hasil training
models/temporary/	Bisa digunakan untuk hasil baru sebelum diputuskan overwrite
logs/train/	        Semua log proses pelatihan per feeder dan kategori