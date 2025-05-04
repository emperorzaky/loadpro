# 🚀 SETUP.md - Environment Setup for LOADPRO on Ubuntu 24.04

This guide outlines the step-by-step procedure to prepare a clean Ubuntu 24.04 system for the LOADPRO project, including Python environment setup, dependencies installation, and repository initialization.

---

## 🧰 Prerequisites

Ensure your system is connected to the internet and updated.

```bash
sudo apt update && sudo apt upgrade -y
```

Install essential build tools and Python headers (required for building Python with pyenv):

```bash
sudo apt install -y \
  build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev \
  libxmlsec1-dev libffi-dev liblzma-dev git
```

---

## 🐍 Python Environment with `pyenv`

### 1. Install `pyenv`

```bash
curl https://pyenv.run | bash
```

### 2. Configure Shell Environment

Add the following lines to the bottom of your `~/.bashrc`:

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
```

Then, apply the changes:

```bash
source ~/.bashrc
```

### 3. Install Python via `pyenv`

```bash
pyenv install 3.11.7
pyenv global 3.11.7
```

### 4. Verify Installation

```bash
python --version  # Should show Python 3.11.7
```

---

## 📁 Clone LOADPRO Repository

```bash
git clone <your-private-repo-url>
cd loadpro
```

---

## 🧪 Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

---

## 📦 Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Ensure the following packages are listed in `requirements.txt`:

* tensorflow==2.15.0
* keras==2.15.0
* scikit-learn==1.4.2
* pyswarms==1.3.0
* numpy==1.26.4
* pandas==2.2.2
* matplotlib==3.8.4
* tqdm==4.66.2
* h5py==3.10.0
* joblib==1.3.2
* scikit-optimize==0.10.1
* absl-py==2.1.0
* grpcio==1.60.1
* packaging==25.0
* wrapt==1.14.1
* psutil==5.9.8

---

## ⚙️ Run the Pipeline

To execute the entire LOADPRO pipeline:

```bash
python3 loadpro.py
```

To reset all processed data and start fresh:

```bash
python3 loadpro.py --reset
```

---

## ✅ Verification

After running, verify these directories are populated:

* `data/processed/split/` → contains siang & malam CSV files
* `models/single/` → contains .json and .weights.h5 files
* `results/prediction_results.csv` → contains final predictions
* `logs/` → contains all tuning and inference logs

---

## 📞 Support

For issues or support, contact: `zaky.pradikto@pln.co.id`

---

> Built with ❤ for UP3 Mojokerto
