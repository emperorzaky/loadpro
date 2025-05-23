'''
EVAL_SINGLE_MODEL.PY
--------------------
LOADPRO Project | Eksekusi Single Model untuk Subprocess Tuning (Anti-OOM)

Deskripsi:
- Script ini dijalankan secara terpisah via subprocess dari tuning.py
- Membaca input: data (.npz) dan parameter (.json)
- Melatih model LSTM, mengevaluasi, dan menyimpan hasil (.json)
- Setelah eksekusi, proses akan ditutup sehingga memori dibersihkan oleh OS

Cara pakai:
python eval_single_model.py --data path/to/data.npz --params path/to/params.json --output path/to/result.json
'''

import argparse
import json
import numpy as np
import gc
import os
from utils.train_lstm_model import train_and_evaluate_lstm

# --- Parsing argumen CLI ---
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Path ke file .npz berisi X_train, y_train, X_val, y_val')
parser.add_argument('--params', required=True, help='Path ke file .json berisi parameter hyperparameter')
parser.add_argument('--output', required=True, help='Path ke file output hasil evaluasi')
args = parser.parse_args()

# --- Load data ---
with np.load(args.data) as npzfile:
    X_train = npzfile['X_train']
    y_train = npzfile['y_train']
    X_val = npzfile['X_val']
    y_val = npzfile['y_val']

# --- Load parameter ---
with open(args.params, 'r') as f:
    params = json.load(f)

# --- Training dan Evaluasi ---
try:
    mape, mae, model = train_and_evaluate_lstm((X_train, y_train, X_val, y_val), params)
    result = {
        'mape': mape,
        'mae': mae
    }
except Exception as e:
    result = {
        'mape': float('inf'),
        'mae': float('inf'),
        'error': str(e)
    }

# --- Simpan hasil ---
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, 'w') as f:
    json.dump(result, f, indent=2)

# --- Cleanup memory keras ---
del X_train, y_train, X_val, y_val, model
import tensorflow as tf
from tensorflow.keras import backend as K
K.clear_session()
gc.collect()
