"""
validator.py v1.1

Deskripsi:
-----------
Script ini digunakan untuk melakukan validasi hasil preprocessing data pada proyek LOADPRO.
Validator akan:
- Mengecek seluruh file .npz hasil preprocessing di folder data/npz/
- Membaca dan menampilkan bentuk (shape) array X dan y, serta contoh data pertama
- Mencoba memuat file scaler (.pkl) yang sesuai dari data/metadata/
- Mencoba memuat model .keras dari models/single/ dan menampilkan info arsitektur
- Menyimpan hasil validasi dalam format log teks (.log) dan tabel HTML (.html) di logs/validator/

Penggunaan:
-----------
    python scripts/validator.py

Output:
-------
- File log teks: logs/validator/YYYYMMDD_HHMM_validator.log
- File log HTML: logs/validator/YYYYMMDD_HHMM_validator.html

Author: Zaky Pradikto
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Paksa CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hapus warning TensorFlow
import numpy as np
import joblib
from datetime import datetime
import pandas as pd
from tensorflow.keras.models import load_model

# Konfigurasi folder
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
npz_dir = os.path.join(base_dir, 'data', 'npz')
meta_dir = os.path.join(base_dir, 'data', 'metadata')
model_dir = os.path.join(base_dir, 'models', 'single')
log_dir = os.path.join(base_dir, 'logs', 'validator')
os.makedirs(log_dir, exist_ok=True)

# Waktu untuk penamaan file log
now_str = datetime.now().strftime("%Y%m%d_%H%M")
log_path = os.path.join(log_dir, f"{now_str}_validator.log")
html_path = os.path.join(log_dir, f"{now_str}_validator.html")

# Validasi .npz + .pkl
npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
log_lines = []
table_rows = []
log_lines.append(f"🔍 Menemukan {len(npz_files)} file .npz untuk divalidasi...\n")

for file in sorted(npz_files):
    feeder = file.replace('.npz', '')
    npz_path = os.path.join(npz_dir, file)
    pkl_path = os.path.join(meta_dir, feeder + "_scaler.pkl")

    entry = {
        "File": file,
        "X shape": "",
        "y shape": "",
        "X[0]": "",
        "y[0]": "",
        "Scaler Min": "",
        "Scaler Max": "",
        "Status": "✅ Success"
    }

    log_lines.append(f"📁 {file}")

    try:
        data = np.load(npz_path)
        X, y = data['X'], data['y']
        entry["X shape"] = str(X.shape)
        entry["y shape"] = str(y.shape)
        entry["X[0]"] = ', '.join([f"{v:.3f}" for v in X[0].flatten()])
        entry["y[0]"] = f"{y[0]:.3f}"
        log_lines.append(f"   ✅ Loaded .npz | X shape: {X.shape}, y shape: {y.shape}")
        log_lines.append(f"   🔹 X[0]: {entry['X[0]']}")
        log_lines.append(f"   🔹 y[0]: {entry['y[0]']}")
    except Exception as e:
        entry["Status"] = f"❌ NPZ Error: {str(e)}"
        log_lines.append(f"   ❌ Gagal load .npz: {str(e)}")
        table_rows.append(entry)
        log_lines.append("-" * 60)
        continue

    if os.path.exists(pkl_path):
        try:
            scaler = joblib.load(pkl_path)
            entry["Scaler Min"] = ', '.join([str(x) for x in scaler.data_min_])
            entry["Scaler Max"] = ', '.join([str(x) for x in scaler.data_max_])
            log_lines.append(f"   🧪 Scaler loaded:")
            log_lines.append(f"     - Min: {entry['Scaler Min']}")
            log_lines.append(f"     - Max: {entry['Scaler Max']}")
        except Exception as e:
            entry["Status"] = f"❌ PKL Error: {str(e)}"
            log_lines.append(f"   ❌ Gagal load .pkl: {str(e)}")
    else:
        entry["Status"] = "⚠️ No Scaler"
        log_lines.append("   ⚠️ Scaler .pkl tidak ditemukan.")

    log_lines.append("-" * 60)
    table_rows.append(entry)

# Validasi .keras
keras_rows = []
keras_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
log_lines.append(f"\n📦 Menemukan {len(keras_files)} model .keras untuk dicek:\n")

for f in sorted(keras_files):
    path = os.path.join(model_dir, f)
    row = {"Model File": f, "Layers": "", "Input": "", "Output": "", "Params": "", "Status": "✅ Success"}
    log_lines.append(f"📦 {f}")
    try:
        model = load_model(path)
        row["Layers"] = str(len(model.layers))
        row["Input"] = str(model.input_shape)
        row["Output"] = str(model.output_shape)
        row["Params"] = str(model.count_params())
        log_lines.append(f"   ✅ Loaded model: {model.count_params()} params")
        log_lines.append(f"   📐 Input: {model.input_shape}, Output: {model.output_shape}, Layers: {len(model.layers)}")
    except Exception as e:
        row["Status"] = f"❌ Load Error: {str(e)}"
        log_lines.append(f"   ❌ Gagal load model: {str(e)}")
    log_lines.append("-" * 60)
    keras_rows.append(row)

# Simpan .log
with open(log_path, "w") as f:
    f.write('\n'.join(log_lines))

# Simpan .html dua tabel
html_out = "<h2>Validasi Preprocessing (.npz + .pkl)</h2>" + pd.DataFrame(table_rows).to_html(index=False)
html_out += "<br><h2>Validasi Model (.keras)</h2>" + pd.DataFrame(keras_rows).to_html(index=False)
with open(html_path, "w") as f:
    f.write(html_out)

print("✅ Validasi selesai.")
print("📄 Log:", log_path)
print("🌐 HTML:", html_path)
