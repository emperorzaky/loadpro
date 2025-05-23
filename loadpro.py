# ===========================================
# LOADPRO.PY v1.2 (Safe GPU/CPU Detection Patch)
# ===========================================
# LOADPRO Project | Master Pipeline Runner
# Designed and Developed by Zaky Pradikto
# Update: Menambahkan verifikasi device placement TensorFlow secara aman.

import os
import sys
import time
import subprocess
import tensorflow as tf

# Disable verbose TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Setup Device Info ---
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"\n🚀 [INFO] GPU available: {physical_devices[0].name}\n")
    except Exception as e:
        print(f"\n⚠️ [WARNING] GPU detected but failed to set memory growth: {e}\n")
else:
    print("\n⚙️ [INFO] GPU not found. Using CPU.\n")

# Enable TensorFlow device placement logging
tf.debugging.set_log_device_placement(True)

# --- Utility function untuk print + delay ---
def delayed_print(text, delay=0.1):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(0.001)
    print()
    time.sleep(delay)

# --- Splash Screen ---
def show_splash_screen():
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
    show_splash_screen()

    if '--reset' in sys.argv:
        print("\n⚠️ Reset mode aktif. Menghapus data lama...")
        subprocess.run(["python3", "reset.py"])
        print("✅ Reset selesai. Memulai pipeline baru...\n")

    start_time = time.time()

    try:
        subprocess.run(["python3", "scripts/preprocess.py"])
        subprocess.run(["python3", "scripts/tuning.py"])
        subprocess.run(["python3", "scripts/predict.py"])
    except Exception as e:
        print(f"❌ Error saat eksekusi pipeline: {e}")

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n✨ Pipeline LOADPRO selesai dalam {int(minutes)} menit {int(seconds)} detik.")

if __name__ == "__main__":
    main()
