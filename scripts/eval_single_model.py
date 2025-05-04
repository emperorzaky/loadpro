# =====================================================
# EVAL_SINGLE_MODEL.PY
# =====================================================
# LOADPRO Project | Eksekusi Single Model via Subprocess
#
# Deskripsi:
# - Script ini dijalankan sebagai subprocess dari tuning.py
# - Dirancang untuk melatih dan mengevaluasi 1 kombinasi hyperparameter
#   tanpa membebani memori utama (anti OOM).
# - Input: file .npz (data) dan .json (parameter)
# - Output: file .json berisi hasil evaluasi (MAPE & MAE)
#
# Cara pakai:
# python eval_single_model.py --data path/to/data.npz --params path/to/params.json --output path/to/result.json
# =====================================================

import argparse
import json
import numpy as np
import gc
import os
from utils.train_lstm_model import train_and_evaluate_lstm

# --- 1. Parsing argumen dari command-line ---
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Path ke file .npz berisi X_train, y_train, X_val, y_val')
parser.add_argument('--params', required=True, help='Path ke file .json berisi parameter hyperparameter')
parser.add_argument('--output', required=True, help='Path ke file output hasil evaluasi')
args = parser.parse_args()

# --- 2. Load data training dan validasi dari file .npz ---
with np.load(args.data) as npzfile:
    X_train = npzfile['X_train']
    y_train = npzfile['y_train']
    X_val = npzfile['X_val']
    y_val = npzfile['y_val']

# --- 3. Load parameter hyperparameter dari file .json ---
with open(args.params, 'r') as f:
    params = json.load(f)

# --- 4. Latih dan evaluasi model ---
try:
    # Fungsi akan mengembalikan MAPE, MAE, dan model terlatih
    mape, mae, model = train_and_evaluate_lstm((X_train, y_train, X_val, y_val), params)
    result = {
        'mape': mape,
        'mae': mae
    }
except Exception as e:
    # Jika terjadi error saat training, log sebagai inf
    result = {
        'mape': float('inf'),
        'mae': float('inf'),
        'error': str(e)
    }

# --- 5. Simpan hasil evaluasi ke file .json ---
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, 'w') as f:
    json.dump(result, f, indent=2)

# --- 6. Bersihkan memori secara eksplisit ---
del X_train, y_train, X_val, y_val, model
import tensorflow as tf
from tensorflow.keras import backend as K
K.clear_session()
gc.collect()
