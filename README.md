# LOADPRO - Load Prediction Optimization

**LOADPRO** adalah sebuah AI yang sedang dikembangkan untuk melakukan prediksi beban puncak listrik harian berbasis RNN-LSTM dengan tuning yang dioptimalkan menggunakan Particle Swarm Optimization untuk mempermudah monitoring beban harian penyulang secara otomatis, akurat, dan dapat di_scale-up_.

---

## 📚 Deskripsi

- Prediksi beban harian per penyulang menggunakan deep learning (RNN-LSTM).
- Hyperparameter tuning awal menggunakan Particle Swarm Optimization (PSO).
- Fleksibel untuk pengembangan metode tuning lain seperti Random Search, Bayesian Optimization, dan Genetic Algorithm.
- Output utama berupa model terbaik dan hasil prediksi per feeder.

---

## 🧠 Fitur Utama

- ✅ Preprocessing data siang/malam terpisah
- ✅ Tuning PSO berbasis resume (checkpoint log)
- ✅ Skema 2 tahap (eksplorasi → eksploitasi lokal)
- ✅ Evaluasi multi-metrik (MAPE, RMSE, MAE)
- ✅ Struktur modular dan dokumentasi lengkap

---

## ⚙️ Instalasi

Pastikan environment Python 3.8+ sudah aktif dan `venv` sudah di-setup:

```bash
pip install -r requirements.txt

🚀 Eksekusi Pipeline

# Preprocessing data .csv
python scripts/preprocess.py

# Tuning hyperparameter (PSO)
python scripts/tuning.py

# Prediksi final menggunakan model terbaik
python scripts/predict.py

🗂️ Struktur Folder

loadpro/
├── dashboard/       # UI/Backend untuk visualisasi beban
├── data/            # Data input dan hasil preprocessing
├── docs/            # Dokumentasi teknis dan catatan internal
├── input/           # File parameter, konfigurasi, dan input manual
├── logs/            # Log proses training, tuning, dan prediksi
├── models/          # File model LSTM hasil tuning
├── results/         # Hasil prediksi, evaluasi, grafik
├── scripts/         # Pipeline utama preprocessing, tuning, dll.
└── README.md        # Dokumentasi utama proyek

🧪 Status Perkembangan

Preprocessing pipeline

Tuning PSO

Evaluasi dan logging

Struktur multi-feeder

Manuver otomatis antar penyulang

Dashboard Streamlit full deployment

    Rilis public version (tanpa data PLN)

👤 Author

    Zaky Emperorz
    Team Leader Engineering - PLN UP3 Mojokerto
    Email: zakypradikto@gmail.com
    GitHub: github.com/emperorzaky

📄 Lisensi

Proyek ini bersifat private. Tidak diperkenankan distribusi ulang tanpa izin tertulis.
