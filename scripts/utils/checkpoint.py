'''CHECKPOINT.PY v0.2
------------------------------
LOADPRO Project | Checkpointing Utility (JSON Format)

Deskripsi:
- Menyimpan progress tuning secara real-time dengan format JSON per baris
- Kompatibel dengan resume.py untuk evaluasi ulang dan lanjutan tuning
- Menyediakan opsi reset log sebelum tuning baru dimulai

Perubahan v0.2:
- Format penyimpanan full JSON, satu entry per baris
- Timestamp otomatis per entry
- Penulisan ulang agar konsisten dengan pipeline tuning v2.1
'''

import json
import os
from datetime import datetime

# --- Fungsi untuk Menyimpan Progress Tuning ---
def save_progress(iteration, particle_idx, params, metrics, progress_log_path):
    """
    Menyimpan satu entry hasil tuning ke file log dalam format JSON per baris.

    Args:
        iteration (int): Iterasi saat ini dalam proses tuning (PSO loop)
        particle_idx (int): Indeks partikel saat ini
        params (dict): Hyperparameter yang digunakan
        metrics (dict): Hasil evaluasi model (mape, mae, dll)
        progress_log_path (str): Lokasi file log penyimpanan
    """
    os.makedirs(os.path.dirname(progress_log_path), exist_ok=True)

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "iteration": iteration,
        "particle": particle_idx,
        "params": params,
        "metrics": metrics
    }

    with open(progress_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')
        f.flush()

# --- Fungsi untuk Mereset File Progress ---
def reset_progress(progress_log_path):
    """
    Mengosongkan isi file progress log.

    Args:
        progress_log_path (str): Path file log
    """
    if os.path.exists(progress_log_path):
        with open(progress_log_path, 'w', encoding='utf-8') as f:
            f.write("")
