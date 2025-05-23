# LOADPRO v4.0 - Load Prediction Optimization

LOADPRO adalah sistem prediksi beban harian berbasis RNN-LSTM yang dirancang untuk distribusi energi PLN. Versi 4.0 menata ulang struktur proyek secara modular dan siap untuk skala besar.

## 📂 Struktur Direktori

```
loadpro/
├── data/
│   ├── raw/                # CSV mentah
│   ├── processed/          # (opsional) hasil split csv
│   ├── npz/                # .npz hasil preprocessing
│   └── metadata/           # scaler.pkl / JSON info feeder
│
├── models/
│   ├── single/             # model .keras per penyulang
│   └── tuning/             # hasil tuning (log/csv)
│
├── results/
│   ├── predict/            # hasil prediksi harian
│   └── benchmark/          # evaluasi dan perbandingan
│
├── logs/
│   ├── preprocess/         # log preprocessing
│   ├── tuning/             # log tuning
│   └── predict/            # log prediksi
│
├── scripts/
│   ├── preprocess.py       # preprocessing utama
│   ├── tuning_fast.py      # tuning hyperparameter cepat
│   ├── save_best_model.py  # training ulang dari tuning
│   ├── predict.py          # prediksi semua penyulang
│   └── utils/              # modul bantu (split, scaler, model)
│
├── loadpro.py              # (opsional) entry-point CLI
├── requirements.txt        # dependensi Python
└── README.md               # deskripsi proyek
```

## 🚀 Setup Environment (Ubuntu 24.04)

### 1. Install Prasyarat & Pyenv
```bash
sudo apt update && sudo apt install -y build-essential git curl wget unzip nano \
  python3-full python3-pip python3-venv \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libncursesw5-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

curl https://pyenv.run | bash
```

Tambahkan ke ~/.bashrc:
```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```
Lalu aktifkan:
```bash
source ~/.bashrc
```

### 2. Install Python 3.11.7 & Virtualenv
```bash
pyenv install 3.11.7
cd ~/loadpro
pyenv local 3.11.7
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Auto-Activate venv (opsional)
Tambahkan ke ~/.bashrc:
```bash
function cd() {
  builtin cd "$@" || return
  if [[ "$PWD" == "$HOME/loadpro" && -d "$PWD/venv" ]]; then
    [[ -z "$VIRTUAL_ENV" ]] && source "$PWD/venv/bin/activate"
  elif [[ -n "$VIRTUAL_ENV" ]]; then
    deactivate
  fi
}
```

Lalu jalankan:
```bash
source ~/.bashrc
```

### 5. Install CUDA 12.2 Toolkit (tanpa driver)
Unduh dari: https://developer.nvidia.com/cuda-12-2-0-download-archive

```bash
chmod +x cuda_12.2.0_535.54.03_linux.run
sudo ./cuda_12.2.0_535.54.03_linux.run
```

Tambahkan ke ~/.bashrc:
```bash
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

### 6. Install cuDNN 8.9.7 for CUDA 12.2
Unduh dari: https://developer.nvidia.com/rdp/cudnn-archive

```bash
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/*.gpg /usr/share/keyrings/
sudo apt update
sudo apt install libcudnn8 libcudnn8-dev libcudnn8-samples
```

Verifikasi:
```bash
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```
Harus muncul `8.9.7`

### 7. Cegah Sleep Saat Tuning (opsional)
```bash
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false
```

---

## ✅ Status Setup
- [x] Python 3.11.7 via pyenv
- [x] Virtualenv `venv/`
- [x] Git system terpasang
- [x] CUDA 12.2 + cuDNN 8.9.7 siap pakai TensorFlow GPU
- [ ] Dataset .csv siap di `data/raw/`
- [ ] Script `preprocess.py` siap jalan

---

Siap gas PLN sejati! ⚡
