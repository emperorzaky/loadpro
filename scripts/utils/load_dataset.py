# ===================================================
# load_dataset.py
# ---------------------------------------------------
# LOADPRO Utility | Dataset Loader
#
# Fungsi:
# - Memuat data .npz hasil preprocessing (X, y)
# - Mengembalikan array X, y siap latih
# ===================================================

import numpy as np
import os

def load_dataset(feeder_name: str, kategori: str):
    """
    Load dataset hasil preprocessing untuk 1 penyulang & 1 kategori.

    Args:
        feeder_name (str): Nama file penyulang (tanpa ekstensi).
        kategori (str): 'siang' atau 'malam'.

    Returns:
        X (np.ndarray): Input features untuk LSTM.
        y (np.ndarray): Target values.
    """
    base_path = f"data/npz/{feeder_name}_{kategori}.npz"
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"File tidak ditemukan: {base_path}")
    
    data = np.load(base_path)
    return data["X"], data["y"]
