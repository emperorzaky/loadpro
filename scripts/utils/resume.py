'''
RESUME.PY v0.1
------------------------------
LOADPRO Project | Resume After Crash Utilities

Deskripsi:
- Utility untuk membaca progress log, menghapus entry terakhir, dan menentukan kombinasi tuning mana yang perlu dilanjutkan.
- Digunakan untuk mengimplementasikan fitur resume tuning setelah crash atau gangguan proses.

Perubahan v0.1:
- Penambahan fungsi read_progress_log()
- Penambahan fungsi remove_last_entry()
- Penambahan fungsi generate_resume_plan()
- Strukturisasi komentar profesional dan modularisasi siap production
'''

import json
import os

# --- Fungsi Membaca Progress Log ---
def read_progress_log(progress_log_path):
    """
    Membaca semua entry dari file progress log.

    Args:
        progress_log_path (str): Path penuh ke file progress log (*.log)

    Returns:
        list: List of dictionaries berisi semua entry progress
    """
    if not os.path.exists(progress_log_path):
        return []

    entries = []
    with open(progress_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError:
                continue  # Abaikan baris yang corrupt

    return entries

# --- Fungsi Menghapus Entry Terakhir ---
def remove_last_entry(progress_log_path):
    """
    Menghapus entry terakhir dari progress log untuk menghindari resume dari kombinasi corrupt.

    Args:
        progress_log_path (str): Path penuh ke file progress log (*.log)
    """
    entries = read_progress_log(progress_log_path)
    if not entries:
        return  # Tidak ada yang perlu dihapus

    entries = entries[:-1]  # Buang entry terakhir

    # Rewrite file dengan entries baru
    with open(progress_log_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# --- Fungsi Membuat Rencana Resume ---
def generate_resume_plan(progress_log_path):
    """
    Membuat daftar kombinasi yang sudah dievaluasi dari progress log.

    Args:
        progress_log_path (str): Path penuh ke file progress log (*.log)

    Returns:
        set: Set berisi tuple (iteration, particle) yang sudah selesai
    """
    entries = read_progress_log(progress_log_path)
    completed_set = set()

    for entry in entries:
        iteration = entry.get('iteration')
        particle = entry.get('particle')
        if iteration is not None and particle is not None:
            completed_set.add((iteration, particle))

    return completed_set
