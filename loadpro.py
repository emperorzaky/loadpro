# ===========================================
# LOADPRO.PY v1.2 (Safe GPU/CPU Detection Patch)
# ===========================================
# LOADPRO Project | Master Pipeline Runner
# Designed and Developed by Zaky Pradikto
# Update: Menambahkan verifikasi device placement TensorFlow secara aman.

# --- Import libraries ---
import os                     # Untuk konfigurasi environment dan path
import sys                    # Untuk menangani argumen CLI dan sistem exit
import time                   # Untuk kalkulasi waktu dan delay
import subprocess             # Untuk mengeksekusi script eksternal
import tensorflow as tf       # Framework deep learning utama yang digunakan

# Disable verbose TensorFlow warnings (level 3 = hanya error)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Setup Device Info (Verifikasi GPU/CPU) ---
physical_devices = tf.config.list_physical_devices('GPU')  # Deteksi perangkat GPU
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)  # Aktifkan memory growth agar alokasi tidak boros
        print(f"\n🚀 [INFO] GPU available: {physical_devices[0].name}\n")    # Tampilkan informasi GPU yang terdeteksi
    except Exception as e:
        # Jika gagal mengatur memory growth, tampilkan warning
        print(f"\n⚠️ [WARNING] GPU detected but failed to set memory growth: {e}\n")
else:
    # Jika GPU tidak tersedia, fallback ke CPU
    print("\n⚙️ [INFO] GPU not found. Using CPU.\n")

# Aktifkan log device placement untuk debugging (opsional, bisa dimatikan jika terlalu verbose)
tf.debugging.set_log_device_placement(True)

# --- Utility function untuk print + delay ---
def delayed_print(text, delay=0.1):
    """
    Menampilkan teks dengan efek ketik lambat (mirip typing) lalu delay sejenak.
    Digunakan untuk efek dramatis di splash screen.
    """
    for char in text:
        print(char, end='', flush=True)
        time.sleep(0.001)  # Delay per karakter
    print()
    time.sleep(delay)      # Delay antar baris

# --- Splash Screen ---
def show_splash_screen():
    """
    Menampilkan branding visual dari LOADPRO saat awal eksekusi.
    """
    print("===========================================")
    delayed_print("LOADPRO | LOAD PRediction Optimization")
    delayed_print("Designed and Developed by Zaky Pradikto, ULP Pacet")
    delayed_print("An Intelligent Load Forecasting System")
    delayed_print("Powered by RNN-LSTM")
    delayed_print("Tuned with Particle Swarm Optimization (PSO)")
    delayed_print("Precision in Every Prediction")
    print("===========================================")

# --- Main function ---
def main():
    # Tampilkan splash screen branding
    show_splash_screen()

    # Jika user menambahkan flag '--reset' maka data lama akan dihapus
    if '--reset' in sys.argv:
        print("\n⚠️ Reset mode aktif. Menghapus data lama...")
        subprocess.run(["python3", "reset.py"])  # Jalankan reset.py
        print("✅ Reset selesai. Memulai pipeline baru...\n")

    # Mulai stopwatch untuk mengukur durasi total pipeline
    start_time = time.time()

    try:
        # Jalankan pipeline utama secara berurutan
        subprocess.run(["python3", "scripts/preprocess.py"])  # Step 1: Preprocessing data
        subprocess.run(["python3", "scripts/tuning.py"])      # Step 2: Hyperparameter tuning
        subprocess.run(["python3", "scripts/predict.py"])     # Step 3: Prediksi dan output hasil
    except Exception as e:
        # Jika salah satu step gagal, tampilkan pesan error
        print(f"❌ Error saat eksekusi pipeline: {e}")

    # Hitung durasi eksekusi pipeline secara keseluruhan
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n✨ Pipeline LOADPRO selesai dalam {int(minutes)} menit {int(seconds)} detik.")

# Eksekusi fungsi main jika script dijalankan langsung (bukan di-import)
if __name__ == "__main__":
    main()
