# ============================================
# BENCHMARK_PARTICLE.PY
# --------------------------------------------
# Simulasi training LSTM 1 partikel untuk benchmark memori
# Digunakan untuk menguji penggunaan RAM dan GPU/CPU pada 1 job LSTM
# ============================================

import tensorflow as tf
import numpy as np
import time
import os

# --- Setup Dummy Dataset ---
# Membuat data acak dengan dimensi yang sama seperti windowing beban asli
# X_dummy: 5000 sampel, 7 timestep, 1 fitur (mirip beban harian 7 hari)
# y_dummy: 5000 target nilai (regresi)
X_dummy = np.random.rand(5000, 7, 1)  # (batch_size, window_size, feature_dim)
y_dummy = np.random.rand(5000, 1)     # (batch_size, target_dim)

# --- Fungsi Pembuatan Model LSTM ---
def create_lstm_model(hidden_units=50, window_size=7):
    """
    Membuat model LSTM sederhana untuk regresi.
    Arsitektur:
    - Input layer (7 timestep, 1 fitur)
    - 1 LSTM layer
    - 1 Dense output layer
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, 1)),
        tf.keras.layers.LSTM(hidden_units),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# --- Proses Training Simulasi ---
print("🛠️ Building LSTM model...")
model = create_lstm_model(hidden_units=50, window_size=7)

print("🚀 Starting training simulation (this will simulate 1 particle load)...")
start_time = time.time()

# Training dummy selama 5 epoch, batch size 32
# Output verbose=2 untuk melihat setiap epoch
history = model.fit(X_dummy, y_dummy, epochs=5, batch_size=32, verbose=2)

end_time = time.time()

# --- Output Waktu Eksekusi ---
print(f"✅ Benchmark finished. Total training time: {end_time - start_time:.2f} seconds.")
print("📊 Please check your RAM usage during the process (htop or free -h).")

# --- Delay waktu agar user bisa cek penggunaan memori manual ---
time.sleep(60)  # Delay 60 detik (1 menit)
