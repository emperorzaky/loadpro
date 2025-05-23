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
sudo apt update && sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl llvm libncursesw5-dev \
xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git

curl https://pyenv.run | bash
```

Lalu tambahkan ke ~/.bashrc:
```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
source ~/.bashrc
```

### 2. Install Python 3.11.7 & Virtualenv
```bash
pyenv install 3.11.7
pyenv global 3.11.7
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Auto Activate venv (opsional)
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

---

## ✅ Status Setup
- [x] Python 3.11.7 via pyenv
- [x] Virtualenv `venv/`
- [x] Git system terpasang (untuk VS Code / git opsional)
- [ ] Dataset .csv siap di `data/raw/`
- [ ] Script `preprocess.py` siap jalan

---

Siap gas PLN sejati! ⚡

