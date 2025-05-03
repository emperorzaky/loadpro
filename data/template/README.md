# data/template/

Folder ini berisi file template `.csv` sebagai referensi struktur input data mentah penyulang.

---

## 🎯 Tujuan

Template ini digunakan sebagai acuan bagi tim input/manual agar data yang dimasukkan ke folder `data/raw/` memiliki format yang sesuai dan bisa langsung diproses oleh sistem LOADPRO.

---

## 📋 Struktur Template

File template biasanya bernama:

```
penyulang_aragog.csv
```

Dengan isi sebagai berikut:

```
Tanggal,Waktu,Beban
05/01/2024,Siang,101.2
05/01/2024,Malam,87.6
05/02/2024,Siang,104.0
05/02/2024,Malam,88.4
...
```

---

## 🧾 Penjelasan Kolom

| Kolom  | Format       | Deskripsi                                         |
|--------|--------------|---------------------------------------------------|
| A: `Tanggal` | `MM/DD/YYYY` | Tanggal pencatatan beban                     |
| B: `Waktu`   | `Siang` / `Malam` | Shift waktu pengukuran harian             |
| C: `Beban`   | `float` (kW)   | Nilai beban aktual dari penyulang           |

---

## 🧠 Catatan Teknis

- File harus disimpan dengan encoding **UTF-8**.
- Gunakan format desimal dengan titik (`.`) — contoh: `101.2`, bukan `101,2`.
- Format tanggal harus konsisten `MM/DD/YYYY`.
- File template ini **tidak akan diproses secara otomatis**, hanya sebagai panduan input.

---

## 📂 File Terkait

- `data/template/feeder_template.csv` – Template standar input penyulang
- `data/raw/` – Tempat file input sebenarnya disimpan
