# ===================================================
# tuning.py v1.0
# ---------------------------------------------------
# LOADPRO | Hyperparameter Tuning Entry Point
#
# Deskripsi:
# - Menjalankan tuning berdasarkan metode yang dipilih
# - Mendukung modularisasi metode tuning (grid, random, bayesopt, dll)
# - Menyimpan log hasil tuning secara otomatis ke logs/tuning/
# ===================================================

import os
import sys
import argparse
from datetime import datetime

# Import fungsi tuning dari metode yang tersedia
from tuning.bayesopt_search import run_bayesopt

# Import utilitas umum
from utils.load_dataset import load_dataset
from utils.lstm_train_predict import train_and_evaluate_model

# Pastikan folder logs dan models tersedia
os.makedirs("logs/tuning", exist_ok=True)
os.makedirs("models/tuning", exist_ok=True)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--feeder", type=str, required=True, help="Nama file penyulang (tanpa ekstensi)")
parser.add_argument("--kategori", type=str, required=True, choices=["siang", "malam"])
parser.add_argument("--method", type=str, default="bayesopt", choices=["bayesopt"], help="Metode tuning")
args = parser.parse_args()

# Logging ke file
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
logfile = f"logs/tuning/{timestamp}_tuning_{args.method}.log"
sys.stdout = open(logfile, "w")

print(f"📌 Tuning dimulai untuk penyulang: {args.feeder} [{args.kategori}] menggunakan metode {args.method}")

# Load data
X, y = load_dataset(args.feeder, args.kategori)

# Jalankan metode tuning yang dipilih
if args.method == "bayesopt":
    run_bayesopt(X, y, args.feeder, args.kategori)

print("✅ Tuning selesai.")
