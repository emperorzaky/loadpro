# LOADPRO - Load Prediction Optimization

**LOADPRO** adalah sistem AI untuk prediksi beban puncak listrik harian berbasis RNN-LSTM. Versi **v.4 (Stable)** ini dirancang agar dapat berjalan **stabil di CPU maupun GPU**, dengan dukungan **tuning modular**. Sistem dapat di-*scale-up* untuk distribusi PLN multi-penyulang.

---

## 📚 Deskripsi

* Prediksi beban harian per penyulang menggunakan deep learning (RNN-LSTM).
* Mendukung 5 metode tuning: Grid Search, Random Search, PSO, Bayesian Optimization, dan Genetic Algorithm.
* Output utama: model terbaik (.keras), hasil tuning (.pkl), hasil prediksi (.csv), log proses (.log).

---

## 🧹 Spesifikasi Sistem

| Komponen               | Versi / Spesifikasi      |
| ---------------------- | ------------------------ |
| OS                     | Ubuntu 24.04 LTS         |
| Python                 | 3.11.7                   |
| CUDA Toolkit           | 12.2                     |
| cuDNN                  | 8.9.7                    |
| GPU Support            | NVIDIA GTX 1660 Ti (6GB) |
| RAM Minimum            | 16 GB                    |
| Swap Memory Disarankan | 16 GB                    |
| Virtual Environment    | `venv` (Python built-in) |

---

## 📦 Python Package Requirements

| Library      | Versi  |
| ------------ | ------ |
| TensorFlow   | 2.15.0 |
| Keras        | 2.15.0 |
| Pandas       | 2.2.2  |
| NumPy        | 1.26.4 |
| Scikit-Learn | 1.4.2  |
| Matplotlib   | 3.8.4  |
| tqdm         | 4.66.4 |
| joblib       | 1.4.2  |

> Semua dependensi tersedia di `requirements.txt`

---

## ⚙️ Instalasi

```bash
# Clone repo
git clone https://github.com/emperorzaky/loadpro.git
cd loadpro

# Aktifkan virtualenv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🔀 Alur Pipeline LOADPRO

Berikut penjelasan end-to-end:

### 1. Preprocessing

```bash
python3 scripts/preprocess.py
```

* Membaca semua `.csv` dari `data/raw/`
* Membersihkan NaN, nol, dan baris fail
* Menyimpan data ke `data/npz/` dan scaler ke `data/metadata/`
* Log tersimpan di `logs/preprocess/`

### 2. Tuning (Opsional)

```bash
python3 scripts/tuning.py --feeder penyulang_x --kategori malam --method bayesopt
```

* Menjalankan tuning hyperparameter LSTM
* Menyimpan log ke `logs/tuning/`
* Menyimpan model terbaik ke `models/tuning/`
* Menyimpan hasil tuning `.pkl` ke `results/tuning/`

### 3. Training Final Model

```bash
python3 scripts/train.py --feeder penyulang_x --kategori malam
```

* Melatih model dengan best params
* Menyimpan `.keras` ke `models/single/`
* Menyimpan log training ke `logs/train/`

### 4. Prediksi

#### Semua file historis:

```bash
python3 scripts/predict.py --feeder penyulang_x --kategori malam
```

#### Prediksi next-day:

```bash
python3 scripts/predict_next.py --feeder penyulang_x --kategori malam
```

* Output `results/predict/{feeder}_{kategori}_pred.csv`
* Output next-day `results/predict/next_{feeder}_{kategori}.csv`
* Log disimpan di `logs/predict/`

---

## 📁 Struktur Folder

```
loadpro/
├── data/
│   ├── raw/            # CSV mentah
│   ├── npz/            # Hasil preprocessing (X, y)
│   └── metadata/       # Scaler (.pkl) per penyulang
│
├── models/
│   ├── single/         # Model final (.keras)
│   └── tuning/         # Model hasil tuning terbaik (.keras)
│
├── results/
│   ├── predict/        # Hasil prediksi .csv
│   └── tuning/         # Hasil tuning .pkl
│
├── logs/
│   ├── preprocess/     # Log preprocessing
│   ├── train/          # Log training final model
│   ├── tuning/         # Log hasil tuning
│   └── predict/        # Log prediksi dan next-day
│
├── scripts/            # Semua pipeline script
├── docs/               # SETUP.md, FLOW.md, dll
└── loadpro.py          # CLI entrypoint (opsional)
```

---

## 🚀 Eksekusi Otomatis

```bash
# Jalankan semua proses end-to-end:
python3 loadpro.py

# Atau manual satu per satu sesuai tahap di atas.
```

---

## 🧦 Benchmarking GPU

Tes TensorFlow + CUDA:

```bash
python3 scripts/test.py
```

| Device | Waktu Eksekusi |
| ------ | -------------- |
| CPU    | 5.4 detik      |
| GPU    | 0.8 detik      |

---

## 🔄 Versi

| Versi | Fitur                                 |
| ----- | ------------------------------------- |
| v3    | PSO tuning (100 particles, 5 iterasi) |
| v4    | Modular, stabil CPU+GPU, tanpa tuning |

---

## 👤 Author

Zaky Pradikto
Team Leader Teknik - PLN ULP Pacet
📧 [zakypradikto@gmail.com](mailto:zakypradikto@gmail.com)
🔗 github.com/emperorzaky

---

## 📄 Lisensi

Proyek ini bersifat **open** dan diperkenankan untuk disalin, disebarluaskan, atau digunakan ulang tanpa izin tertulis dari pemilik resmi.
