# ===========================================
# BENCHMARK.PY v2.0
# ===========================================
# Corporate-Grade Benchmarking CPU vs GPU for LOADPRO
# Dengan opsi feeder, partikel, dan iterasi kustom

import os
import subprocess
import time
import sys
from datetime import datetime

# --- Utility Functions ---

def log_message(message, logfile):
    """
    Menampilkan pesan ke console dan menyimpan ke file log.
    """
    print(message)
    with open(logfile, 'a') as f:
        f.write(f"{message}\n")

def run_script(script_args, env, logfile):
    """
    Menjalankan script Python dengan environment tertentu dan mencatat waktunya.
    """
    log_message(f"[RUNNING] {' '.join(script_args)}", logfile)
    start = time.time()
    result = subprocess.run(script_args, env=env)
    elapsed = time.time() - start

    if result.returncode != 0:
        # Jika script gagal dijalankan, tampilkan error dan hentikan eksekusi.
        log_message(f"[ERROR] {' '.join(script_args)} gagal dijalankan. Exit code: {result.returncode}", logfile)
        raise RuntimeError(f"{' '.join(script_args)} failed with exit code {result.returncode}")

    # Catat waktu eksekusi.
    minutes, seconds = divmod(elapsed, 60)
    log_message(f"[TIME] {' '.join(script_args)} selesai dalam {int(minutes)} menit {int(seconds)} detik.", logfile)
    return elapsed

def benchmark(mode, feeder, particles, iterations, logfile):
    """
    Melakukan benchmark dalam mode CPU atau GPU dengan feeder dan konfigurasi tertentu.
    """
    env = os.environ.copy()
    if mode == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU
    else:
        env.pop('CUDA_VISIBLE_DEVICES', None)  # Enable GPU

    log_message(f"[BENCHMARK START] Mode: {mode.upper()}", logfile)
    total_start = time.time()

    # Jalankan preprocessing dan tuning dengan konfigurasi saat ini.
    run_script(["python3", "scripts/preprocess.py"], env, logfile)
    run_script(["python3", "scripts/tuning_custom.py", feeder, str(particles), str(iterations)], env, logfile)

    total_elapsed = time.time() - total_start
    minutes, seconds = divmod(total_elapsed, 60)
    log_message(f"[SUMMARY] {mode.upper()} Mode selesai dalam {int(minutes)} menit {int(seconds)} detik.", logfile)
    return total_elapsed

# --- Main ---

def main():
    """
    Fungsi utama untuk menjalankan benchmark CPU dan GPU serta menghitung speedup.
    """
    if len(sys.argv) != 4:
        print("Usage: python3 benchmark.py <feeder_name> <particles> <iterations>")
        sys.exit(1)

    feeder = sys.argv[1]  # Nama penyulang (feeder)
    particles = int(sys.argv[2])  # Jumlah partikel untuk tuning
    iterations = int(sys.argv[3])  # Jumlah iterasi PSO

    os.makedirs("logs/benchmark", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"logs/benchmark/{timestamp}_benchmark_{feeder}.log"

    try:
        # Reset data sebelum benchmark dimulai
        log_message("\n--- RESET DATA AWAL ---", logfile)
        subprocess.run(["python3", "reset.py"], check=True)

        # Benchmark mode CPU
        log_message("\n--- BENCHMARK CPU ---", logfile)
        cpu_time = benchmark('cpu', feeder, particles, iterations, logfile)

        # Reset ulang sebelum benchmark GPU
        log_message("\n--- RESET DATA SEBELUM GPU ---", logfile)
        subprocess.run(["python3", "reset.py"], check=True)

        # Benchmark mode GPU
        log_message("\n--- BENCHMARK GPU ---", logfile)
        gpu_time = benchmark('gpu', feeder, particles, iterations, logfile)

        # Reset final
        log_message("\n--- RESET DATA FINAL ---", logfile)
        subprocess.run(["python3", "reset.py"], check=True)

        # Hitung speedup GPU dibanding CPU
        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            log_message(f"\n[FINAL RESULT] Speedup GPU dibanding CPU: {speedup:.2f}x lebih cepat.", logfile)
        else:
            log_message("\n[FINAL RESULT] Error: GPU time is zero.", logfile)

    except Exception as e:
        log_message(f"[FATAL ERROR] {str(e)}", logfile)

if __name__ == "__main__":
    main()
