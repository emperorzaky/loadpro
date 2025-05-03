"""
PREPARE_DATASET.PY v1.0
------------------------
LOADPRO Project | Utilitas Pembagian Dataset

Deskripsi:
- Membagi dataset hasil preprocessing (X, y) menjadi 2 subset:
  Training set (80%) dan Validation set (20%).
- Tidak menggunakan shuffle karena data bertipe time-series.

Fungsi:
- split_train_val(X, y, test_size=0.2): Membagi data dengan rasio default 80:20.

"""

from sklearn.model_selection import train_test_split

def split_train_val(X, y, test_size=0.2):
    """
    Membagi dataset menjadi training dan validation (default 80:20).
    Tidak menggunakan shuffle untuk menjaga urutan time-series.

    Args:
        X (np.ndarray): Input features dengan dimensi (samples, timesteps, features)
        y (np.ndarray): Target labels
        test_size (float): Proporsi data untuk validation (default = 0.2)

    Returns:
        X_train, X_val, y_train, y_val
    """
    return train_test_split(X, y, test_size=test_size, shuffle=False)
