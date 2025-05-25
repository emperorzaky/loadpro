"""
device.py

Utility kecil untuk mendeteksi apakah TensorFlow dapat menggunakan GPU.
Digunakan oleh seluruh pipeline LOADPRO untuk menampilkan device yang aktif.
"""

import tensorflow as tf

def get_device():
    gpus = tf.config.list_physical_devices('GPU')
    return 'GPU' if gpus else 'CPU'
