'''
EVALUATE_MODEL.PY v0.3
------------------------------
LOADPRO Project | Model Evaluation Metrics

Deskripsi:
- Berisi fungsi evaluasi performa model prediksi
- Menghitung dua metrik utama:
  1. Mean Absolute Percentage Error (MAPE)
  2. Mean Absolute Error (MAE)

Perubahan v0.3:
- Penambahan komentar profesional
- Struktur minimalis namun siap production
'''

import numpy as np

# --- Fungsi Menghitung MAPE ---
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Menghitung MAPE (Mean Absolute Percentage Error):
    - Mengukur persentase error relatif terhadap nilai aktual
    - Mengabaikan nilai aktual yang bernilai 0
    Args:
        y_true (array-like): Nilai aktual
        y_pred (array-like): Nilai prediksi
    Returns:
        float: Nilai MAPE (%)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0  # Hindari pembagian dengan nol
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# --- Fungsi Menghitung MAE ---
def mean_absolute_error(y_true, y_pred):
    """
    Menghitung MAE (Mean Absolute Error):
    - Mengukur rata-rata error absolut antara prediksi dan aktual
    Args:
        y_true (array-like): Nilai aktual
        y_pred (array-like): Nilai prediksi
    Returns:
        float: Nilai MAE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))
