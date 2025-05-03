# 📘 SETUP.md – Environment Setup for LOADPRO (Ubuntu 24.04)

Dokumen ini menjelaskan tahapan setup penuh untuk menjalankan proyek **LOADPRO** di lingkungan Ubuntu 24.04, mulai dari instalasi Python hingga integrasi CUDA/cuDNN agar TensorFlow dapat mendeteksi GPU (GTX 1660 Ti).

---

## 🧩 1. Install Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget unzip nano \\
  python3-full python3-pip python3-venv \\
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \\
  libncursesw5-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

🐍 2. Python Setup with pyenv

Always show details

curl https://pyenv.run | bash

Tambahkan ke ~/.bashrc:

Always show details

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

Lalu:

Always show details

source ~/.bashrc
pyenv install 3.11.7

📦 3. Setup Project & Virtual Environment

Always show details

cd ~/loadpro
pyenv local 3.11.7
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

⚙️ 4. Auto-Activate venv Saat Masuk ke ~/loadpro

Tambahkan ke akhir ~/.bashrc:

Always show details

function cd() {
  builtin cd "$@" || return
  if [[ "$PWD" == "$HOME/loadpro" && -d "$PWD/venv" ]]; then
    [[ -z "$VIRTUAL_ENV" ]] && source "$PWD/venv/bin/activate"
  elif [[ -n "$VIRTUAL_ENV" ]]; then
    deactivate
  fi
}

Kemudian:

Always show details

source ~/.bashrc

✅ 5. Verifikasi Environment

Always show details

python --version
which python
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

⚡ 6. GPU Setup: CUDA + cuDNN
a. Cek Driver NVIDIA

Always show details

nvidia-smi

b. Install CUDA Toolkit 12.2

Always show details

chmod +x cuda_12.2.0_535.54.03_linux.run
sudo ./cuda_12.2.0_535.54.03_linux.run

Tambahkan ke ~/.bashrc:

Always show details

export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
source ~/.bashrc

c. Install cuDNN 8.9.7

Always show details

sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/*.gpg /usr/share/keyrings/
sudo apt update
sudo apt install libcudnn8 libcudnn8-dev libcudnn8-samples

Verifikasi:

Always show details

cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

🛡️ 7. Prevent Sleep Saat Training

Always show details

gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false

💾 Tambahan: Ubah Swap Memory Jadi 16GB

Always show details
sudo swapoff -a
sudo rm -f /swapfile
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
swapon --show
free -h
