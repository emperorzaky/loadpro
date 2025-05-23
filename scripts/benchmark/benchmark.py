# ===========================================
# BENCHMARK.PY v2.1
# ===========================================
# Corporate-Grade Benchmarking CPU vs GPU for LOADPRO
# Dengan opsi feeder, partikel, dan iterasi kustom
# Perubahan v2.1:
# - Penambahan auto-create folder untuk path logfile

import os
import subprocess
import time
import sys
from datetime import datetime

# --- Utility Functions ---
def log_message(message, logfile):
    print(message)
    
    # Auto-create folder if missing
    log_dir = os.path.dirname(logfile)
    os.makedirs(log_dir, exist_ok=True)
    
    with open(logfile, 'a') as f:
        f.write(f"{message}\n")

def run_script(script_args, env, logfile):
    log_message(f"[RUNNING] {' '.join(script_args)}", logfile)
    start = time.time()
    result = subprocess.run(script_args, env=env)
    elapsed = time.time() - start

    if result.returncode != 0:
        log_message(f"[ERROR] {' '.join(script_args)} gagal dijalankan. Exit code: {result.returncode}", logfile)
        raise RuntimeError(f"{' '.join(script_args)} failed with exit code {result.returncode}")

    minutes, seconds = divmod(elapsed, 60)
    log_message(f"[TIME] {' '.join(script_args)} selesai dalam {int(minutes)} menit {int(seconds)} detik.", logfile)
    return elapsed

def benchmark(mode, feeder, particles, iterations, logfile):
    env = os.environ.copy()
    if mode == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU
    else:
        env.pop('CUDA_VISIBLE_DEVICES', None)  # Enable GPU

    log_message(f"[BENCHMARK START] Mode: {mode.upper()}", logfile)
    total_start = time.time()

    run_script(["python3", "scripts/preprocess.py"], env, logfile)
    run_script(["python3", "scripts/tuning_custom.py", feeder, str(particles), str(iterations)], env, logfile)

    total_elapsed = time.time() - total_start
    minutes, seconds = divmod(total_elapsed, 60)
    log_message(f"[SUMMARY] {mode.upper()} Mode selesai dalam {int(minutes)} menit {int(seconds)} detik.", logfile)
    return total_elapsed

# --- Main ---
def main():
    if len(sys.argv) != 4:
        print("Usage: python3 benchmark.py <feeder_name> <particles> <iterations>")
        sys.exit(1)

    feeder = sys.argv[1]
    particles = int(sys.argv[2])
    iterations = int(sys.argv[3])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"logs/benchmark/{timestamp}_benchmark_{feeder}.log"

    try:
        log_message("\n--- RESET DATA AWAL ---", logfile)
        subprocess.run(["python3", "reset.py"], check=True)

        log_message("\n--- BENCHMARK CPU ---", logfile)
        cpu_time = benchmark('cpu', feeder, particles, iterations, logfile)

        log_message("\n--- RESET DATA SEBELUM GPU ---", logfile)
        subprocess.run(["python3", "reset.py"], check=True)

        log_message("\n--- BENCHMARK GPU ---", logfile)
        gpu_time = benchmark('gpu', feeder, particles, iterations, logfile)

        log_message("\n--- RESET DATA FINAL ---", logfile)
        subprocess.run(["python3", "reset.py"], check=True)

        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            log_message(f"\n[FINAL RESULT] Speedup GPU dibanding CPU: {speedup:.2f}x lebih cepat.", logfile)
        else:
            log_message("\n[FINAL RESULT] Error: GPU time is zero.", logfile)

    except Exception as e:
        log_message(f"[FATAL ERROR] {str(e)}", logfile)

if __name__ == "__main__":
    main()
