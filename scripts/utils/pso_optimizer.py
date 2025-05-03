### PSO_OPTIMIZER.PY v0.9
"""
LOADPRO Project | Particle Swarm Optimization (PSO)

Deskripsi:
- PSO modular untuk optimasi hyperparameter LSTM.
- Mendukung resume, checkpointing, dan context-aware objective.
- Mencegah re-evaluasi kombinasi (iteration, particle) yang sudah tuntas.
- Menangani kasus seluruh kombinasi telah dievaluasi (resume penuh).

Perubahan v0.9:
- Menambahkan validasi global_best untuk menghindari return None.
- Tambahan flag resume_full jika semua kombinasi sudah di-skip.
- Pencatatan warning jika tuning dilewati seluruhnya.
"""

import numpy as np

def pso_optimize(objective_func, bounds, n_particles=10, n_iterations=20,
                 inertia=0.5, cognitive=1.5, social=2.0, extra_args=None):
    """
    PSO Optimization untuk minimasi fungsi objektif.

    Args:
        objective_func (function): Fungsi objektif (wrapped) yang akan diminimasi.
        bounds (list of tuple): Batas bawah dan atas setiap parameter.
        n_particles (int): Jumlah partikel.
        n_iterations (int): Jumlah iterasi.
        inertia (float): Koefisien inersia.
        cognitive (float): Koefisien kognitif.
        social (float): Koefisien sosial.
        extra_args (list or tuple): Argumen tambahan untuk objective_func.

    Returns:
        np.ndarray: Parameter terbaik yang ditemukan.
    """
    num_dimensions = len(bounds)

    # --- Inisialisasi posisi dan kecepatan partikel ---
    particles = np.random.rand(n_particles, num_dimensions)
    for i in range(num_dimensions):
        low, high = bounds[i]
        particles[:, i] = low + particles[:, i] * (high - low)

    velocities = np.zeros_like(particles)
    personal_best = particles.copy()
    personal_best_scores = np.full(n_particles, np.inf)
    global_best = None
    global_best_score = np.inf

    resume_full_flag = True

    # --- Iterasi utama PSO ---
    for iteration in range(n_iterations):
        for i in range(n_particles):

            # --- Set konteks ke objective_func ---
            if hasattr(objective_func, 'iteration_idx'):
                objective_func.iteration_idx = iteration + 1
            if hasattr(objective_func, 'particle_idx'):
                objective_func.particle_idx = i + 1

            # --- Evaluasi skor ---
            score = objective_func(particles[i], *extra_args) if extra_args else objective_func(particles[i])

            # --- Skip jika kombinasi sudah pernah dijalankan ---
            if np.isinf(score):
                continue

            resume_full_flag = False

            # --- Update velocity dan posisi ---
            r1 = np.random.rand(num_dimensions)
            r2 = np.random.rand(num_dimensions)

            cognitive_velocity = cognitive * r1 * (personal_best[i] - particles[i])
            social_velocity = social * r2 * (global_best - particles[i]) if global_best is not None else 0
            velocities[i] = inertia * velocities[i] + cognitive_velocity + social_velocity
            particles[i] += velocities[i]

            for d in range(num_dimensions):
                low, high = bounds[d]
                particles[i, d] = np.clip(particles[i, d], low, high)

            try:
                if hasattr(objective_func, 'save_progress_func'):
                    objective_func.save_progress_func(
                        iteration=iteration + 1,
                        particle_idx=i + 1,
                        params=particles[i],
                        metrics={"MAPE": score, "RMSE": None, "MAE": None}
                    )
            except Exception as e:
                print(f"[Checkpoint Error] {str(e)}", flush=True)

            if score < personal_best_scores[i]:
                personal_best[i] = particles[i].copy()
                personal_best_scores[i] = score
                if score < global_best_score:
                    global_best = particles[i].copy()
                    global_best_score = score

    if resume_full_flag:
        print("⚠️ Semua kombinasi telah dievaluasi sebelumnya (resume penuh). Tidak ada training yang dilakukan.")
        return np.array([bounds[i][0] for i in range(num_dimensions)])  # return nilai batas bawah default

    return global_best
