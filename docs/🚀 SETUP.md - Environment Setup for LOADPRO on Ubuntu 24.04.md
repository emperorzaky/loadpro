🚀 SETUP.md — LOADPRO Deployment Guide (Ubuntu 24.04)

Dokumen ini menjelaskan setup penuh dari environment LOADPRO, mulai dari fresh install Ubuntu 24.04 hingga TensorFlow dapat mendeteksi dan memanfaatkan GPU (GTX 1660 Ti, CUDA 12.2, cuDNN 8.9.7).
🧩 1. Install Dependencies

sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget unzip nano \
  python3-full python3-pip python3-venv \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libncursesw5-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

🐍 2. Python Setup with pyenv

curl https://pyenv.run | bash

Tambahkan ke ~/.bashrc:

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

Kemudian:

source ~/.bashrc
pyenv install 3.11.7

📦 3. Setup Project & Virtual Environment

cd ~/loadpro
pyenv local 3.11.7
python -m venv venv
source venv/bin/activate
pip install --upgrade pip

Install dependencies:

pip install -r requirements.txt

⚙️ 4. Auto-Activate venv Saat Masuk ke ~/loadpro

Tambahkan ini ke akhir ~/.bashrc:

function cd() {
  builtin cd "$@" || return
  if [[ "$PWD" == "$HOME/loadpro" && -d "$PWD/venv" ]]; then
    [[ -z "$VIRTUAL_ENV" ]] && source "$PWD/venv/bin/activate"
  elif [[ -n "$VIRTUAL_ENV" ]]; then
    deactivate
  fi
}

Lalu jalankan:

source ~/.bashrc

✅ 5. Verifikasi Environment

python --version         # Expect: 3.11.7
which python             # Expect: ~/loadpro/venv/bin/python
python -c "import tensorflow as tf; print(tf.__version__)"  # Expect: 2.15.0
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

⚡ 6. GPU Setup: CUDA + cuDNN
a. Cek Driver NVIDIA

nvidia-smi

    Pastikan Driver Version ≥ 535, dan GPU terdeteksi.

b. Install CUDA Toolkit 12.2 (tanpa driver)

Unduh dari:
https://developer.nvidia.com/cuda-12-2-0-download-archive

Lalu:

chmod +x cuda_12.2.0_535.54.03_linux.run
sudo ./cuda_12.2.0_535.54.03_linux.run

Tambahkan ke ~/.bashrc:

export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

source ~/.bashrc

c. Install cuDNN 8.9.7 untuk CUDA 12.2

Unduh .deb dari:
https://developer.nvidia.com/rdp/cudnn-archive

sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/*.gpg /usr/share/keyrings/
sudo apt update
sudo apt install libcudnn8 libcudnn8-dev libcudnn8-samples

Verifikasi:

cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

    Harus muncul 8.9.7

🛡️ 7. Prevent Sleep During Tuning

gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false

🧪 8. Benchmarking (Opsional)

Jalankan:

python benchmark.py --default

Untuk menguji kecepatan CPU vs GPU dengan skenario ringan.
🚀 Status Akhir

✅ TensorFlow 2.15.0 mendeteksi GPU
✅ CUDA 12.2 dan cuDNN 8.9.7 aktif
✅ Virtual Environment stabil
✅ Bebas error boot (modeset/driver fixed)
✅ LOADPRO siap untuk training berat



💻 Langkah-langkah Ubah Swap Memory jadi 16GB (tested on Ubuntu):
1. Nonaktifkan swap lama (jika ada)

sudo swapoff -a

2. Hapus swapfile lama (optional, jika bukan bawaan system)

sudo rm -f /swapfile

3. Buat swapfile baru 16GB

sudo fallocate -l 16G /swapfile

Kalau fallocate tidak tersedia atau error:

sudo dd if=/dev/zero of=/swapfile bs=1G count=16

4. Ubah permission

sudo chmod 600 /swapfile

5. Set sebagai swap

sudo mkswap /swapfile

6. Aktifkan swap

sudo swapon /swapfile

7. Permanentkan di fstab

echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

8. Cek status

swapon --show
free -h
