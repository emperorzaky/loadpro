# LOADPRO - Load Prediction Optimization

LOADPRO adalah sebuah AI yang sedang dikembangkan untuk melakukan prediksi beban puncak listrik harian berbasis RNN-LSTM, dengan tuning hyperparameter yang dioptimalkan menggunakan Particle Swarm Optimization (PSO). Tujuannya adalah untuk mempermudah monitoring beban harian penyulang secara otomatis, akurat, dan dapat di-scale-up ke seluruh sistem distribusi PLN.

---

## 📚 Deskripsi

* Prediksi beban harian per penyulang menggunakan deep learning (RNN-LSTM).
* Hyperparameter tuning menggunakan Particle Swarm Optimization (PSO).
* Mendukung metode tuning lain seperti Random Search, Bayesian Optimization, Genetic Algorithm.
* Output utama: model terbaik + hasil prediksi harian per feeder.

---

## 🧩 Spesifikasi Sistem

LOADPRO dikembangkan dan diuji pada environment berikut:

| Komponen               | Versi / Spesifikasi      |
| ---------------------- | ------------------------ |
| OS                     | Ubuntu 24.04 LTS         |
| Python                 | 3.10                     |
| CUDA Toolkit           | 12.2                     |
| cuDNN                  | 8.9                      |
| GPU Support            | NVIDIA GTX 1660 Ti (6GB) |
| RAM Minimum            | 16 GB                    |
| Swap Memory Disarankan | 16 GB                    |
| Virtual Environment    | venv (Python built-in)   |

> Untuk menggunakan GPU, pastikan driver NVIDIA dan versi CUDA/cuDNN sesuai dengan TensorFlow dan Keras yang digunakan.

---

## 📦 Python Package Requirements

Daftar library Python utama (dan versi rekomendasi):

| Library         | Versi  |
| --------------- | ------ |
| TensorFlow      | 2.15.0 |
| Keras           | 2.15.0 |
| Pandas          | 2.2.2  |
| NumPy           | 1.26.4 |
| Scikit-Learn    | 1.4.2  |
| Matplotlib      | 3.8.4  |
| tqdm            | 4.66.2 |
| joblib          | 1.3.2  |
| scikit-optimize | 0.10.1 |
| pyswarms        | 1.3.0  |
| h5py            | 3.10.0 |
| pyyaml          | 6.0.1  |
| protobuf        | 4.25.3 |
| psutil          | 5.9.8  |
| absl-py         | 2.1.0  |
| grpcio          | 1.60.1 |
| packaging       | 25.0   |
| wrapt           | 1.14.1 |

> Semua dependensi tersedia di `docs/requirements.txt`.

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
pip install -r docs/requirements.txt
```

---

## 🚀 Eksekusi Pipeline

### Menjalankan pipeline penuh:

```bash
python loadpro.py
```

### Opsi Reset (Mulai dari awal):

```bash
python loadpro.py --reset
```

Opsi ini akan menjalankan `reset.py` sebelum memulai ulang pipeline. Yang akan dihapus:

* Data hasil split
* Model hasil tuning
* File prediksi
* Seluruh log proses
* File input sementara

### Detail Tahapan Pipeline

1. **Preprocessing (`scripts/preprocess.py`)**

   * Membersihkan data feeder (.csv) dari nilai 0 dan NaN.
   * Menggabungkan kolom Tanggal dan Waktu menjadi `Timestamp`.
   * Memisahkan data menjadi dua file: **siang** (jam 10:00) dan **malam** (jam 19:00).
   * Output disimpan di: `data/processed/split/`

2. **Tuning Hyperparameter (`scripts/tuning.py`)**

   * Membagi data menjadi 80% training dan 20% validation.
   * Mengoptimalkan parameter LSTM (hidden units, learning rate, window size, epochs) menggunakan **Particle Swarm Optimization (PSO)**.
   * Tiap kombinasi dievaluasi dalam subprocess untuk mencegah OOM.
   * Progress tuning dicatat ke log resume.
   * Model terbaik disimpan dalam format `.json + .weights.h5` ke `models/single/`.

3. **Prediksi Beban (`scripts/predict.py`)**

   * Memuat model terbaik untuk tiap feeder.
   * Melakukan prediksi beban hari berikutnya berdasarkan data terakhir.
   * Output ditulis ke: `results/prediction_results.csv`
   * Narasi prediksi ditampilkan ke terminal dan dicatat ke log.

---

## 🧠 Fitur Utama

* ✅ Preprocessing data siang/malam terpisah
* ✅ Tuning PSO dengan fitur resume log
* ❌ Skema 2 tahap (eksplorasi → eksploitasi lokal) sedang dikembangkan
* ✅ Evaluasi akurasi multi-metrik (MAPE, RMSE, MAE)
* ✅ Struktur kode modular & terdokumentasi

---

## 🗂️ Struktur Folder

```
loadpro/
├── dashboard/       # UI/Backend untuk visualisasi beban
├── data/            # Data input dan hasil preprocessing
├── docs/            # Dokumentasi teknis dan catatan internal
├── input/           # File parameter, konfigurasi, dan input manual
├── logs/            # Log proses training, tuning, dan prediksi
├── models/          # File model LSTM hasil tuning
├── results/         # Hasil prediksi, evaluasi, grafik
├── scripts/         # Pipeline utama: preprocess, tuning, prediksi
└── README.md        # Dokumentasi utama proyek
```

---

## 📌 Status Perkembangan

* ✅ Preprocessing pipeline
* ✅ PSO hyperparameter tuning
* ✅ Model evaluation & log tracking
* ✅ Struktur multi-feeder siap pakai
* 🚧 Two-stage PSO (eksplorasi → eksploitasi)
* 🚧 Dashboard Streamlit deployment
* 📦 Public version sanitasi & rilis terbuka

---

## 👤 Author

**Zaky Pradikto**
Team Leader Engineering - PLN ULP Pacet
📧 [zakypradikto@gmail.com](mailto:zakypradikto@gmail.com)
🔗 github.com/emperorzaky

---

## 📄 Lisensi

Proyek ini bersifat private dan tidak diperkenankan untuk disalin, disebarluaskan, atau digunakan ulang tanpa izin tertulis dari pemilik resmi.
