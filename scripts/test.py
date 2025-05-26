# benchmark_tensorflow_cpu_vs_gpu.py
import tensorflow as tf
import time

print("🔍 Cek GPU yang terdeteksi TensorFlow:")
gpus = tf.config.list_physical_devices('GPU')
print(gpus if gpus else "❌ Tidak ada GPU terdeteksi")

# Ukuran matriks besar untuk test performa
shape = [10000, 10000]

# Fungsi benchmark
def benchmark(device_name):
    with tf.device(device_name):
        print(f"\n🚀 Benchmarking pada {device_name}...")
        start = time.time()
        a = tf.random.normal(shape)
        b = tf.random.normal(shape)
        c = tf.matmul(a, b)
        _ = c.numpy()  # Force sync untuk GPU
        tf.keras.backend.clear_session()
        end = time.time()
        print("📌 Hasil shape:", c.shape)
        print(f"⏱️ Waktu eksekusi: {end - start:.4f} detik")

# Jalankan benchmark CPU
benchmark("/CPU:0")

# Jalankan benchmark GPU (jika tersedia)
if gpus:
    try:
        benchmark("/GPU:0")
    except Exception as e:
        print(f"⚠️ Benchmark GPU gagal: {e}")
