# PSO_OPTIMIZER.PY v1.1
"""
LOADPRO Project | Particle Swarm Optimization (PSO) with Parallel Meta Passing

Deskripsi:
- Versi ini menggantikan closure dengan passing meta dict agar multiprocessing berjalan tanpa pickle error
- Fungsi objektif menerima tuple (params_array, meta_dict)
- Fully compatible dengan ProcessPoolExecutor
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor  # Untuk eksekusi paralel menggunakan multi-proses

# ---------------------------------------------------------------------------- #
# Fungsi global yang akan dijalankan oleh tiap worker dalam pool
# Harus berada di global scope agar bisa dipickle oleh multiprocessing
# ---------------------------------------------------------------------------- #
def single_eval_worker(args):
    params_array, meta, objective_func = args  # unpack argumen
    return objective_func(params_array, meta)  # jalankan fungsi objektif

# ---------------------------------------------------------------------------- #
# Fungsi utama PSO dengan dukungan multiprocessing dan passing meta
# ---------------------------------------------------------------------------- #
def pso_optimize(objective_func, bounds, n_particles=10, n_iterations=20,
                 inertia=0.5, cognitive=1.5, social=2.0, extra_args_list=None,
                 n_jobs=6):
    """
    PSO Optimization dengan parallel subprocess dan meta passing.

    Args:
        objective_func (function): Fungsi objektif (menerima (params_array, meta_dict))
        bounds (list of tuple): Rentang parameter untuk tiap dimensi.
        n_particles (int): Jumlah partikel yang digunakan dalam populasi.
        n_iterations (int): Jumlah iterasi PSO (generasi).
        inertia (float): Faktor inersia untuk menjaga kecepatan partikel.
        cognitive (float): Faktor kognitif (daya tarik ke posisi terbaik pribadi).
        social (float): Faktor sosial (daya tarik ke posisi global terbaik).
        extra_args_list (list of dict): Meta-data per particle per iterasi, digunakan sebagai context.
        n_jobs (int): Jumlah proses paralel yang digunakan (slot GPU/CPU).

    Returns:
        np.ndarray: Parameter terbaik yang ditemukan (global_best).
    """

    # Inisialisasi populasi awal (random dalam rentang bounds)
    num_dimensions = len(bounds)
    particles = np.random.rand(n_particles, num_dimensions)  # posisi awal
    for i in range(num_dimensions):
        low, high = bounds[i]
        particles[:, i] = low + particles[:, i] * (high - low)

    velocities = np.zeros_like(particles)  # inisialisasi kecepatan nol
    personal_best = particles.copy()       # simpan posisi terbaik masing-masing partikel
    personal_best_scores = np.full(n_particles, np.inf)  # nilai terbaik (MAPE minimal)
    global_best = None                     # posisi terbaik global
    global_best_score = np.inf

    resume_full_flag = True  # flag untuk deteksi apakah semua kombinasi sudah selesai (resume penuh)

    # Loop utama PSO
    for iteration in range(n_iterations):
        args_list = []  # daftar tugas yang akan diberikan ke worker pool

        # Siapkan meta untuk setiap partikel
        for i in range(n_particles):
            meta = extra_args_list[iteration][i]  # meta dict khusus untuk partikel i di iterasi ini
            args_list.append((particles[i], meta, objective_func))  # masukkan ke daftar tugas

        # Jalankan parallel subprocess untuk evaluasi partikel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(single_eval_worker, args_list))  # hasil berupa list of skor

        # Update posisi dan kecepatan berdasarkan hasil evaluasi
        for i, score in enumerate(results):
            if np.isinf(score):
                continue  # skip jika skor tidak valid (misalnya hasil dari resume skip)

            resume_full_flag = False  # set false karena ada kombinasi yang belum dievaluasi

            # Hitung velocity update menggunakan rumus PSO klasik
            r1 = np.random.rand(num_dimensions)
            r2 = np.random.rand(num_dimensions)

            cognitive_velocity = cognitive * r1 * (personal_best[i] - particles[i])
            social_velocity = social * r2 * (global_best - particles[i]) if global_best is not None else 0
            velocities[i] = inertia * velocities[i] + cognitive_velocity + social_velocity
            particles[i] += velocities[i]

            # Clipping posisi agar tetap dalam rentang yang diperbolehkan
            for d in range(num_dimensions):
                low, high = bounds[d]
                particles[i, d] = np.clip(particles[i, d], low, high)

            # Update personal best jika skor membaik
            if score < personal_best_scores[i]:
                personal_best[i] = particles[i].copy()
                personal_best_scores[i] = score
                # Update global best jika ini skor terbaik sejauh ini
                if score < global_best_score:
                    global_best = particles[i].copy()
                    global_best_score = score

    # Jika seluruh kombinasi sebelumnya sudah dievaluasi (resume penuh)
    if resume_full_flag:
        print("⚠️ Semua kombinasi telah dievaluasi sebelumnya (resume penuh). Tidak ada training yang dilakukan.")
        return np.array([bounds[i][0] for i in range(num_dimensions)])  # fallback: batas bawah default

    return global_best  # kembalikan solusi terbaik
