# data/raw/

Folder ini menyimpan data mentah hasil ekspor dari SCADA, AMR, atau sumber eksternal lainnya. (dalam case ini data berasal dari optel)

---

## 📋 Format File

Setiap file mewakili satu penyulang dan berisi data harian beban puncak, siang pukul 10:00 dan malam pukul 19:00.

**Nama file:**
```
penyulang_nama.csv
```

**Contoh:**
```
penyulang_leonidas.csv
penyulang_aragog.csv
```

---

## 🧾 Struktur File CSV

```
+------------+--------+--------+
| Tanggal    | Waktu  | Beban  |
+------------+--------+--------+
| 05/01/2024 | Siang  | 101.2  |
| 05/01/2024 | Malam  | 87.6   |
| 05/02/2024 | Siang  | 104.0  |
| 05/02/2024 | Malam  | 88.4   |
```

**Penjelasan Kolom:**

| Kolom  | Format       | Deskripsi                                         |
|--------|--------------|---------------------------------------------------|
| A: `Tanggal` | `MM/DD/YYYY` | Tanggal pencatatan beban                     |
| B: `Waktu`   | `Siang` / `Malam` | Pembagian waktu pengukuran (shift harian) |
| C: `Beban`   | `float` (A)   | Nilai beban aktual dari penyulang           |

---

## 🛠️ Digunakan Oleh

Folder ini digunakan oleh script:

```bash
python scripts/preprocess.py
```

Script preprocess.py akan membaca seluruh file `.csv` di folder ini dan menghasilkan output ke `data/processed/split/`.

---

## ⚠️ Catatan

- File harus berformat `.csv` dan menggunakan encoding **UTF-8**.
- Format tanggal wajib menggunakan format **MM/DD/YYYY** (bukan DD/MM/YYYY).
- Nama file tidak boleh mengandung spasi.
- Jangan tambahkan file selain `.csv` ke folder ini.
- File template tersedia di `data/template/feeder_template.csv`.
