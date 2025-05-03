# data/processed/split/

Folder ini menyimpan hasil akhir dari proses preprocessing dan windowing data beban penyulang.

---

## 📁 Struktur File

Setiap file di folder ini merepresentasikan satu kombinasi:
- Penyulang (nama feeder)
- Waktu (siang atau malam)
- Format .csv

Contoh nama file:


---

## 📋 Penjelasan Isi File

File .npz merupakan hasil windowing yang sudah siap digunakan untuk pelatihan model RNN-LSTM. Umumnya berisi:
- `X`: array input dengan bentuk `(samples, windowSize, 1)`
- `y`: array target output dengan bentuk `(samples,)`

---

## 🛠️ Sumber Data

Data ini dihasilkan oleh script berikut:

```bash
python scripts/preprocess.py
```

---

## 📌 Catatan

- File di folder ini akan di-*overwrite* setiap kali preprocessing dijalankan ulang.
- Tidak disarankan untuk mengedit file di folder ini secara manual.
- Jika ingin menyimpan versi backup, gunakan folder `data/archive/` (bila tersedia).
