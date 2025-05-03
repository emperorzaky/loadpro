# benchmark_particle.py
import tensorflow as tf
import numpy as np
import time
import os

# Setup dummy data
X_dummy = np.random.rand(5000, 7, 1)  # (batch, window_size, feature_dim)
y_dummy = np.random.rand(5000, 1)     # (batch, output_dim)

# Define simple LSTM model
def create_lstm_model(hidden_units=50, window_size=7):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, 1)),
        tf.keras.layers.LSTM(hidden_units),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

print("🛠️ Building LSTM model...")
model = create_lstm_model(hidden_units=50, window_size=7)

print("🚀 Starting training simulation (this will simulate 1 particle load)...")
start_time = time.time()
history = model.fit(X_dummy, y_dummy, epochs=5, batch_size=32, verbose=2)
end_time = time.time()

print(f"✅ Benchmark finished. Total training time: {end_time - start_time:.2f} seconds.")
print("📊 Please check your RAM usage during the process (htop or free -h).")

# Sleep a bit to let user inspect
time.sleep(60)  # Delay 60 seconds to inspect manually
