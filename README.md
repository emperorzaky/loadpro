# LOADPRO - Load Prediction Optimization

**LOADPRO** adalah sistem AI untuk prediksi beban puncak listrik harian berbasis RNN-LSTM. Versi **v.4 (Stable)** ini dirancang agar dapat berjalan **stabil di CPU maupun GPU**, tanpa proses tuning. Sistem dapat di-*scale-up* untuk distribusi PLN multi-penyulang.

---

## 📚 Deskripsi

- Prediksi beban harian per penyulang menggunakan deep learning (RNN-LSTM).
- Versi ini **tanpa tuning**, menggunakan struktur model default.
- Output utama: model terbaik + hasil prediksi historis dan beban next day per penyulang.

---

## 🧩 Spesifikasi Sistem

| Komponen               | Versi / Spesifikasi        |
|------------------------|----------------------------|
| OS                     | Ubuntu 24.04 LTS           |
| Python                 | 3.11.7                     |
| CUDA Toolkit           | 12.2                       |
| cuDNN                  | 8.9.7                      |
| GPU Support            | NVIDIA GTX 1660 Ti (6GB)   |
| RAM Minimum            | 16 GB                      |
| Swap Memory Disarankan | 16 GB                      |
| Virtual Environment    | `venv` (Python built-in)   |

---

## 📦 Python Package Requirements

| Library        | Versi     |
|----------------|-----------|
| TensorFlow     | 2.15.0    |
| Keras          | 2.15.0    |
| Pandas         | 2.2.2     |
| NumPy          | 1.26.4    |
| Scikit-Learn   | 1.4.2     |
| Matplotlib     | 3.8.4     |
| tqdm           | 4.66.4    |
| joblib         | 1.4.2     |

> Semua dependensi tersedia di `requirements.txt`

---

## ⚙️ Instalasi

```bash
# Clone repo
cd ~ && git clone https://github.com/emperorzaky/loadpro.git
cd loadpro

# Install pyenv & Python 3.11.7 sesuai SETUP.md
# Aktifkan virtualenv
python3 -m venv venv
source venv/bin/activate

# Install dependensi
pip install -r requirements.txt
```

---

## 🚀 Eksekusi Pipeline (v.4 - tanpa tuning)

```bash
# Jalankan semua proses end-to-end:
python3 loadpro.py

# Atau jalankan manual per tahap:
python3 scripts/preprocess.py
python3 scripts/train_all.py
python3 scripts/predict_all.py
```

---

## 🧠 Fitur Utama

- ✅ Preprocessing siang & malam otomatis
- ✅ Training per penyulang dengan fallback GPU → CPU
- ✅ Prediksi all historical + next day
- ✅ Ringkasan hasil dalam log & CSV
- ✅ Struktur modular & terdokumentasi

---

## 🗂️ Struktur Folder

```
loadpro/
├── data/            # Data input dan hasil preprocessing
├── models/          # Model LSTM (.keras)
├── results/         # Hasil prediksi
├── logs/            # Log preprocess, train, predict
├── scripts/         # Script utama (preprocess, train, predict)
├── docs/            # Dokumentasi (SETUP.md, FLOW.md, dll)
└── loadpro.py       # Entry point pipeline
```

---

## 🧪 Benchmarking (Test CPU vs GPU)

- Tes TensorFlow vs GPU dengan `scripts/test.py`
- Benchmark Matrix Multiply 10k x 10k

| Device | Time (approx) |
|--------|----------------|
| CPU    | 5.4 detik      |
| GPU    | 0.8 detik      |

> Gunakan CUDA 12.2 + cuDNN 8.9.7 agar GPU dapat digunakan penuh

---

## 🔄 Versi

| Versi | Deskripsi                              |
|--------|------------------------------------------|
| v.3    | Dengan PSO tuning (100 particles, 5 iterasi) |
| v.4    | Tanpa tuning, stabil di CPU/GPU         |

---

## 👤 Author
Zaky Pradikto  
Team Leader Teknik - PLN ULP Pacet  
📧 zakypradikto@gmail.com  
🔗 github.com/emperorzaky


## 📄 Lisensi
Proyek ini bersifat **private** dan tidak diperkenankan untuk disalin, disebarluaskan, atau digunakan ulang tanpa izin tertulis dari pemilik resmi.
