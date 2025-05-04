# LOADPRO - Load Prediction Optimization

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange)
![Status](https://img.shields.io/badge/Status-Private-inactive)

**LOADPRO** adalah sebuah AI yang sedang dikembangkan untuk melakukan prediksi beban puncak listrik harian berbasis RNN-LSTM, dengan tuning hyperparameter yang dioptimalkan menggunakan Particle Swarm Optimization (PSO). Tujuannya adalah untuk mempermudah monitoring beban harian penyulang secara otomatis, akurat, dan dapat di-*scale-up* ke seluruh sistem distribusi PLN.

---

## 📚 Deskripsi

- Prediksi beban harian per penyulang menggunakan deep learning (RNN-LSTM).
- Hyperparameter tuning menggunakan Particle Swarm Optimization (PSO).
- Mendukung metode tuning lain seperti Random Search, Bayesian Optimization, Genetic Algorithm.
- Output utama: model terbaik + hasil prediksi harian per feeder.

---

## 🧩 Spesifikasi Sistem

LOADPRO dikembangkan dan diuji pada environment berikut:

| Komponen               | Versi / Spesifikasi                  |
|------------------------|--------------------------------------|
| OS                     | Ubuntu 24.04 LTS                     |
| Python                 | 3.10                                 |
| CUDA Toolkit           | 12.2                                 |
| cuDNN                  | 8.9                                  |
| GPU Support            | NVIDIA GTX 1660 Ti (6GB)             |
| RAM Minimum            | 16 GB                                |
| Swap Memory Disarankan | 16 GB                                |
| Virtual Environment    | `venv` (Python built-in)             |

> Untuk menggunakan GPU, pastikan driver NVIDIA dan versi CUDA/cuDNN sesuai dengan TensorFlow dan Keras yang digunakan.

---

## 📦 Python Package Requirements

Daftar library Python utama (dan versi rekomendasi):

| Library          | Versi     |
|------------------|-----------|
| TensorFlow       | 2.16.1    |
| Keras            | 3.2.1     |
| Pandas           | 2.2.2     |
| NumPy            | 1.26.4    |
| Scikit-Learn     | 1.4.2     |
| Matplotlib       | 3.10.1    |
| tqdm             | 4.66.4    |
| joblib           | 1.4.2     |
| namex (custom)   | 0.0.7     |

> Semua dependensi tersedia di `requirements.txt`.

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

🚀 Eksekusi Pipeline

# Preprocessing data .csv
python scripts/preprocess.py

# Tuning hyperparameter (PSO)
python scripts/tuning.py

# Prediksi final menggunakan model terbaik
python scripts/predict.py

🧠 Fitur Utama

    ✅ Preprocessing data siang/malam terpisah

    ✅ Tuning PSO dengan fitur resume log

    ❌ Skema 2 tahap (eksplorasi → eksploitasi lokal) sedang dikembangkan

    ✅ Evaluasi akurasi multi-metrik (MAPE, RMSE, MAE)

    ✅ Struktur kode modular & terdokumentasi

🗂️ Struktur Folder

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

📌 Status Perkembangan

Preprocessing pipeline

PSO hyperparameter tuning

Model evaluation & log tracking

Struktur multi-feeder siap pakai

Two-stage PSO (eksplorasi → eksploitasi)

Dashboard Streamlit deployment

    Public version sanitasi & rilis terbuka

👤 Author

Zaky Pradikto
Team Leader Engineering - PLN ULP Pacet
📧 zakypradikto@gmail.com
🔗 github.com/emperorzaky

📄 Lisensi

Proyek ini bersifat private dan tidak diperkenankan untuk disalin, disebarluaskan, atau digunakan ulang tanpa izin tertulis dari pemilik resmi.
