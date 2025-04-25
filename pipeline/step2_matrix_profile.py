import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
import stumpy
import json
import sys

# -------- Load Config --------
with open("config.json", "r") as f:
    cfg = json.load(f)

cfg1 = cfg["step1_preprocess"]
cfg2 = cfg["step2_matrix_profile"]
cfg_gen = cfg["general"]

INPUT_FILE = "data/processed/processed_trace.sac"
WINDOW_SEC = cfg2["window_sec"]
TOP_N_MOTIFS = cfg2["top_n_motifs"]
USE_GPU = cfg_gen["use_gpu"]
PLOT_MP = cfg2["plot_matrix_profile"]
FORCE_RECOMPUTE = cfg2["force_recompute"]

# -------- Compute folder name --------
def format_duration_name(seconds):
    if seconds < 60:
        return f"{seconds:02d}Seconds"
    elif seconds < 3600:
        return f"{seconds // 60:02d}Minutes"
    else:
        return f"{seconds // 3600:03d}Hours"

duration_str = format_duration_name(cfg1["trim_duration"])
engine_str = "GPU" if USE_GPU else "CPU"
window_str = f"{WINDOW_SEC:.1f}s"
rate_str = f"{int(cfg_gen['sampling_rate'])}Hz"
folder_name = f"{duration_str}-{rate_str}-{window_str}-{engine_str}"
base_dir = os.path.join("results", folder_name)

# -------- Paths --------
OUTPUT_IMAGE_DIR = os.path.join(base_dir, "templates", "images")
OUTPUT_TRACE_DIR = os.path.join(base_dir, "templates", "traces")
OUTPUT_METADATA = os.path.join(base_dir, "templates", "metadata.csv")
MP_SAVE_PATH = os.path.join(base_dir, "matrix_profile.npy")

# -------- Logging --------
os.makedirs(base_dir, exist_ok=True)
sys.stdout = open(os.path.join(base_dir, "log_pipeline.txt"), "a", encoding="utf-8")

# -------- Clean template output dir --------
template_root = os.path.dirname(OUTPUT_METADATA)
if os.path.exists(template_root):
    shutil.rmtree(template_root)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_TRACE_DIR, exist_ok=True)

# -------- Load waveform --------
print("🔄 Loading waveform...")
st = read(INPUT_FILE)
tr = st[0]
fs = tr.stats.sampling_rate
data = tr.data.astype(np.float64)
m = int(WINDOW_SEC * fs)
print(f"✅ Loaded {tr.id}, duration = {len(data)/fs:.1f}s, sampling rate = {fs} Hz")

# -------- Matrix Profile: Compute or Load --------
if os.path.exists(MP_SAVE_PATH) and not FORCE_RECOMPUTE:
    print(f"📁 Loading cached matrix profile from: {MP_SAVE_PATH}")
    mp = np.load(MP_SAVE_PATH, allow_pickle=True)
else:
    print(f"🧠 Computing Matrix Profile (window size = {m})...")
    try:
        if USE_GPU:
            mp = stumpy.gpu_stump(data, m)
            print("⚡ Using GPU acceleration.")
        else:
            raise RuntimeError("Forcing CPU")
    except Exception as e:
        print(f"⚠️ GPU failed or disabled: {e}")
        mp = stumpy.stump(data, m)
        print("💡 Using CPU fallback.")

    np.save(MP_SAVE_PATH, mp)
    print(f"💾 Matrix profile saved to {MP_SAVE_PATH}")

# -------- Motif Selection --------
peak_indices = np.argsort(mp[:, 0])
visited_pairs = set()
motifs_saved = 0
motif_info = []

for idx_a in peak_indices:
    if motifs_saved >= TOP_N_MOTIFS:
        break

    idx_b = int(mp[idx_a, 1])
    if (idx_b, idx_a) in visited_pairs or (idx_a, idx_b) in visited_pairs:
        continue
    visited_pairs.add((idx_a, idx_b))

    for j, idx in enumerate([idx_a, idx_b]):
        start_time = tr.stats.starttime + (idx / fs)
        end_time = start_time + (m / fs)
        segment = tr.copy().trim(starttime=start_time, endtime=end_time)

        id_str = f"motif{motifs_saved + 1}_{'a' if j == 0 else 'b'}"
        trace_path = os.path.join(OUTPUT_TRACE_DIR, f"{id_str}.sac")
        img_path = os.path.join(OUTPUT_IMAGE_DIR, f"{id_str}.png")

        segment.write(trace_path, format="SAC")

        fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
        ax.plot(segment.times(), segment.data, color='black')
        ax.set_title(f"{id_str} | {start_time} to {end_time}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

        motif_info.append({
            "template_id": id_str,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "station": tr.stats.station,
            "network": tr.stats.network,
            "channel": tr.stats.channel,
            "latitude": tr.stats.sac.stla if "sac" in tr.stats and "stla" in tr.stats.sac else "N/A",
            "longitude": tr.stats.sac.stlo if "sac" in tr.stats and "stlo" in tr.stats.sac else "N/A",
            "sampling_rate": fs,
            "start_index": idx
        })

    motifs_saved += 1

# -------- Optional Matrix Profile Plot --------
if PLOT_MP:
    print("📊 Plotting matrix profile...")
    highlight_colors = ['blue', 'red', 'green', 'purple', 'orange']
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.plot(mp[:, 0], color='black', linewidth=1.5)
    ax.set_title("Matrix Profile with Highlighted Motifs")
    ax.set_xlabel("Index (Start Time)")
    ax.set_ylabel("Distance")
    ax.grid(True)

    for i, (idx_a, idx_b) in enumerate(visited_pairs):
        color = highlight_colors[i % len(highlight_colors)]
        ax.axvline(idx_a, color=color, linestyle='--', linewidth=1.5)
        ax.axvline(idx_b, color=color, linestyle='-', linewidth=1.5)

    plt.tight_layout()
    plot_path = os.path.join(template_root, "matrix_profile.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Matrix profile plot saved to {plot_path}")
else:
    print("⚠️ Matrix profile plotting skipped (disabled in config).")

# -------- Save Metadata --------
df = pd.DataFrame(motif_info)
df.to_csv(OUTPUT_METADATA, index=False)
print(f"✅ Saved {2 * motifs_saved} templates and metadata to:")
print(f"    → {OUTPUT_TRACE_DIR}")
print(f"    → {OUTPUT_IMAGE_DIR}")
print(f"    → {OUTPUT_METADATA}")
