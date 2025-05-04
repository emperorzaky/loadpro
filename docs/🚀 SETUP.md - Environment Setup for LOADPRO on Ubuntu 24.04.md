# 🚀 SETUP.md - Environment Setup for LOADPRO on Ubuntu 24.04

This guide will walk you through setting up a clean Ubuntu 24.04 system for running the LOADPRO Project, including Python, dependencies, GPU support (optional), and project structure.

---

## ✅ 1. System Preparation

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl git wget make libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev libopenblas-dev liblapack-dev libhdf5-dev \
software-properties-common
```

> 📦 This includes all development tools, libraries needed for TensorFlow, NumPy, Pandas, etc.

---

## 🐍 2. Install pyenv + pyenv-virtualenv

```bash
curl https://pyenv.run | bash
```

### Add to .bashrc / .zshrc:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Then apply changes:

```bash
exec "$SHELL"
```

---

## 🐍 3. Install Python and Create Virtual Environment

```bash
pyenv install 3.10.14
pyenv virtualenv 3.10.14 loadpro-env
pyenv activate loadpro-env
pyenv global loadpro-env
```

You can verify with:

```bash
python --version
```

Should return `Python 3.10.14`

---

## 📁 4. Clone the Repository

```bash
git clone https://github.com/yourusername/loadpro.git
cd loadpro
```

Optional: create `.python-version` in project root to auto-activate venv

```bash
echo "loadpro-env" > .python-version
```

---

## 📦 5. Install Python Dependencies

```bash
pip install -r docs/requirements.txt
```

If `requirements.txt` is missing, use this base:

```text
# Core ML Libraries
tensorflow==2.15.0
tensorflow-estimator==2.15.0
keras==2.15.0
scikit-learn==1.4.2
scikit-optimize==0.10.1
pyswarms==1.3.0
numpy==1.26.4
pandas==2.2.2
scipy==1.15.2
joblib==1.3.2
tqdm==4.66.2

# Visualization
matplotlib==3.8.4

# File Handling
h5py==3.10.0
pyyaml==6.0.1
protobuf==4.25.3

# Logging and Utility
absl-py==2.1.0
grpcio==1.60.1
packaging==25.0
wrapt==1.14.1
psutil==5.9.8
```

---

## ⚡ 6. (Optional) Setup GPU Support

If your machine has NVIDIA GPU:

```bash
sudo ubuntu-drivers autoinstall
reboot
```

After reboot, verify driver:

```bash
nvidia-smi
```

> Optional: Install CUDA toolkit if required by TensorFlow extensions (not needed for basic GPU use)

TensorFlow 2.15 includes GPU runtime automatically. To test:

```python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 🧪 7. Verify Project Structure

```
loadpro/
├── data/
│   └── raw/                # CSV input files per feeder
├── models/
│   └── single/            # Saved models (.keras / .json + .weights.h5)
├── results/               # Prediction output
├── logs/                  # Tuning and preprocessing logs
├── input/                 # Temp input for subprocess evals
├── scripts/               # Core pipeline scripts
│   ├── preprocess.py
│   ├── tuning.py
│   ├── predict.py
│   └── eval_single_model.py
├── reset.py
├── loadpro.py             # Master runner
└── requirements.txt
```

---

## 🚀 8. Run the Pipeline

```bash
python3 loadpro.py
```

Optional to reset previous results:

```bash
python3 loadpro.py --reset
```

---

## 🛠️ 9. Troubleshooting

* If `pyenv` not found: ensure shell config is sourced properly (`.bashrc` / `.zshrc`)
* If `tensorflow` fails to detect GPU: ensure `nvidia-smi` works and reboot
* If `PermissionError`: check folder permissions on `data`, `models`, `logs`, etc.

---

## 📌 10. (Optional) Enable Auto Activation on Terminal Start

Add this to `.bashrc`:

```bash
cd ~/loadpro && pyenv activate loadpro-env
```

---

## ✅ Setup Completed

You are now ready to develop and run LOADPRO efficiently on a fresh Ubuntu 24.04 environment. Happy tuning!

> Developed and maintained by Zaky Pradikto, ULP Pacet
