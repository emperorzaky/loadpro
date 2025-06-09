🧠 train.py — Melatih model untuk 1 feeder + 1 kategori (siang/malam)
📄 Fungsi:
Melatih model RNN-LSTM berdasarkan 1 file .npz dan menyimpannya sebagai file .keras.

🧩 Argumen:

--feeder	✅ Ya	    Nama penyulang, misal: penyulang_bancang
--kategori	✅ Ya	    Pilihan siang atau malam
--output	❌ Opsional	Folder tujuan simpan model, default ke models/single/

🔁 Contoh pemakaian:

python3 scripts/train.py --feeder penyulang_bancang --kategori malam

Atau simpan ke folder lain:

python3 scripts/train.py --feeder penyulang_bancang --kategori malam --output models/temporary/

📦 Input:

File .npz di data/npz/penyulang_bancang_malam.npz
File .pkl scaler di data/metadata/penyulang_bancang_malam_scaler.pkl

💾 Output:
File .keras hasil pelatihan, disimpan ke --output (default: models/single/)
File log di logs/train/{timestamp}_train_{feeder}_{kategori}.log

📊 Metrik:
MAE (Mean Absolute Error)

RMSE (Root Mean Square Error)

MAPE (jika semua y ≠ 0)