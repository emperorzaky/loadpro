import os
import time
import numpy as np
import tensorflow as tf

# Aktifkan memory growth untuk semua GPU (wajib sebelum digunakan)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("⚠️ Gagal set memory growth:", e)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def benchmark(device_name):
    print(f"\n🚀 Benchmarking on {device_name.upper()}...")
    tf.keras.backend.clear_session()

    with tf.device(device_name):
        X = np.random.random((1024, 10, 16)).astype(np.float32)
        y = np.random.random((1024, 1)).astype(np.float32)

        model = create_model((10, 16))

        start = time.time()
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        duration = time.time() - start

        print(f"✅ {device_name.upper()} training time: {duration:.2f} seconds")
        return duration

# Jalankan benchmark CPU
cpu_time = benchmark('/CPU:0')

# Jalankan benchmark GPU jika tersedia
if gpus:
    try:
        gpu_time = benchmark('/GPU:0')
        print(f"\n⚖️  GPU is {cpu_time / gpu_time:.2f}× faster than CPU\n")
    except Exception as e:
        print(f"⚠️ GPU benchmark gagal: {e}")
else:
    print("\n⚠️ Tidak ditemukan GPU. Hanya CPU yang diuji.")
