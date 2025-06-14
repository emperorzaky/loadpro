# load_dataset.py v1.0
# --------------------------------------------------
# Memuat dataset dari file .npz dan scaler .pkl
# Output: X_train, y_train, X_val, y_val, scaler
# --------------------------------------------------

import numpy as np
import pickle
import os

def load_dataset(npz_path, scaler_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File NPZ tidak ditemukan: {npz_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"File scaler tidak ditemukan: {scaler_path}")

    # Load dataset dari NPZ
    with np.load(npz_path) as data:
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']

    # Load scaler dari pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return X_train, y_train, X_val, y_val, scaler
