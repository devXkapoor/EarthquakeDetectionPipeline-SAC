import os
import shutil
import pandas as pd
import numpy as np
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt
import json
import sys

# --- Load Config ---
with open("config.json", "r") as f:
    cfg = json.load(f)

cfg1 = cfg["step1_preprocess"]
cfg2 = cfg["step2_matrix_profile"]
cfg6 = cfg["step6_generate_outputs"]
cfg_gen = cfg["general"]

# --- Compute Output Folder ---
def format_duration_name(seconds):
    if seconds < 60:
        return f"{seconds:02d}Seconds"
    elif seconds < 3600:
        return f"{seconds // 60:02d}Minutes"
    else:
        return f"{seconds // 3600:03d}Hours"

duration_str = format_duration_name(cfg1["trim_duration"])
engine_str = "GPU" if cfg_gen["use_gpu"] else "CPU"
window_str = f"{cfg2['window_sec']:.1f}s"
rate_str = f"{int(cfg_gen['sampling_rate'])}Hz"
folder_name = f"{duration_str}-{rate_str}-{window_str}-{engine_str}"
base_dir = os.path.join("results", folder_name)

# --- Redirect stdout to log file ---
sys.stdout = open(os.path.join(base_dir, "log_pipeline.txt"), "a", encoding="utf-8")

# --- Paths ---
WAVEFORM_FILE = "data/processed/processed_trace.sac"
CATALOG_FILE = os.path.join(base_dir, "detections", "catalog_filtered.csv")
TEMPLATE_DIR = os.path.join(base_dir, "templates", "traces")
DETECTION_METADATA = os.path.join(base_dir, "detections", "metadata.csv")

EXTRACT_OUT_DIR = os.path.join(base_dir, "detections", "traces")
IMG_RAW_DIR = os.path.join(base_dir, "detections", "images")
IMG_NORM_DIR = os.path.join(base_dir, "detections", "images_normalized")
IMG_NORM_OVERLAY_DIR = os.path.join(base_dir, "detections", "images_norm_overlays")

EVENT_WINDOW_SEC = cfg6["event_window_sec"]
PLOT_WINDOW_SEC = cfg6["plot_window_sec"]

# --- Clean old outputs ---
for folder in [EXTRACT_OUT_DIR, IMG_RAW_DIR, IMG_NORM_DIR, IMG_NORM_OVERLAY_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

if os.path.exists(DETECTION_METADATA):
    os.remove(DETECTION_METADATA)

# --- Load catalog ---
if not os.path.exists(CATALOG_FILE) or os.stat(CATALOG_FILE).st_size == 0:
    print(f"⚠️ No detections found to generate outputs. Skipping step.")
    sys.exit(0)

catalog = pd.read_csv(CATALOG_FILE)
print(f"📖 Loaded detection catalog: {len(catalog)} entries")

# --- Load waveform ---
print("🔄 Loading processed waveform...")
st = read(WAVEFORM_FILE)
tr = st[0]
fs = tr.stats.sampling_rate

event_meta = []

# --- Iterate over detections ---
for i, row in catalog.iterrows():
    match_time = UTCDateTime(row["match_time"])
    template_id = row["template_id"]
    match_index = row["match_index"]

    start = match_time - EVENT_WINDOW_SEC / 2
    end = match_time + EVENT_WINDOW_SEC / 2

    try:
        segment = tr.copy().trim(starttime=start, endtime=end)
    except Exception:
        print(f"⚠️ Skipped event {i}: Out of bounds")
        continue

    seg_id = f"{template_id}_match{i}"
    seg_file = os.path.join(EXTRACT_OUT_DIR, f"{seg_id}.sac")
    tpl_file = os.path.join(TEMPLATE_DIR, f"{template_id}.sac")
    segment.write(seg_file, format="SAC")

    t = np.linspace(-EVENT_WINDOW_SEC/2, EVENT_WINDOW_SEC/2, len(segment.data))
    det_norm = (segment.data - np.mean(segment.data)) / np.std(segment.data)
    duration_hours = cfg1["trim_duration"] / 3600

    # ------------------ PLOT 1: Raw Detection ------------------
    img_raw_path = os.path.join(IMG_RAW_DIR, f"{seg_id}.png")
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), dpi=300, sharey=True)
    for ax in axs:
        ax.plot(t, segment.data, color='black')
        ax.axvline(0, color='blue', linestyle='--', linewidth=1.5, label='Start / Match Time')
        ax.axvline(1, color='green', linestyle='--', linewidth=1.5, label='End Time')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)
    axs[0].set_xlim(-EVENT_WINDOW_SEC / 2, EVENT_WINDOW_SEC / 2)
    axs[0].set_title(f"Zoomed-In View ({EVENT_WINDOW_SEC}s)")
    axs[1].set_xlim(-PLOT_WINDOW_SEC / 2, PLOT_WINDOW_SEC / 2)
    axs[1].set_title(f"Zoomed-Out View ({PLOT_WINDOW_SEC}s)")
    fig.suptitle(f"Raw Detected Signal", fontsize=14, fontweight='bold')
    fig.text(0.5, 0.935, f"Matching Template: {template_id} - Match No.: {i} | Match Index: {match_index} | UTC Match Time: {match_time} | Corr: {row['correlation']:.5f} | Duration of Raw Waveform used: {duration_hours} Hours", ha='center', va='top', fontsize=12, style='italic')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(img_raw_path)
    plt.close()

    # ------------------ PLOT 2: Normalized Detection ------------------
    img_norm_path = os.path.join(IMG_NORM_DIR, f"{seg_id}_norm.png")
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), dpi=300, sharey=True)
    for ax in axs:
        ax.plot(t, det_norm, color='black')
        ax.axvline(0, color='blue', linestyle='--', linewidth=1.5, label='Start / Match Time')
        ax.axvline(1, color='green', linestyle='--', linewidth=1.5, label='End Time')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Z-Norm Amplitude")
        ax.grid(True)
        ax.legend()

    axs[0].set_xlim(-EVENT_WINDOW_SEC / 2, EVENT_WINDOW_SEC / 2)
    axs[0].set_title(f"Zoomed-In View ({EVENT_WINDOW_SEC}s)")
    axs[1].set_xlim(-PLOT_WINDOW_SEC / 2, PLOT_WINDOW_SEC / 2)
    axs[1].set_title(f"Zoomed-Out View ({PLOT_WINDOW_SEC}s)")
    fig.suptitle(f"Z-Normalized Detected Signal")
    fig.text(0.5, 0.935, f"Matching Template: {template_id} - Match No.: {i} | Match Index: {match_index} | UTC Match Time: {match_time} | Corr: {row['correlation']:.5f} | Duration of Raw Waveform used: {duration_hours} Hours", ha='center', va='top', fontsize=12, style='italic')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(img_norm_path)
    plt.close()

    # ------------------ PLOT 3: Overlay ------------------
    img_overlay_path = os.path.join(IMG_NORM_OVERLAY_DIR, f"{seg_id}_overlay.png")
    try:
        tr_tpl = read(tpl_file)[0]
        tpl_norm = (tr_tpl.data - np.mean(tr_tpl.data)) / np.std(tr_tpl.data)
        tpl_duration = len(tpl_norm) / fs
        tpl_t_centered = np.linspace(0, tpl_duration, len(tpl_norm))

        fig, axs = plt.subplots(2, 1, figsize=(16, 9), dpi=300, sharey=True)
        for ax in axs:
            ax.plot(t, det_norm, label="Detection", color='black')
            ax.plot(tpl_t_centered, tpl_norm, label="Template", color='red', alpha=0.7)
            ax.axvline(0, color='blue', linestyle='--', linewidth=1.5, label='Start / Match Time')
            ax.axvline(1, color='green', linestyle='--', linewidth=1.5, label='End Time')
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Z-Norm Amplitude")
            ax.grid(True)
            ax.legend()

        axs[0].set_xlim(-EVENT_WINDOW_SEC / 2, EVENT_WINDOW_SEC / 2)
        axs[0].set_title("Zoomed-In Overlay")

        axs[1].set_xlim(-PLOT_WINDOW_SEC / 2, PLOT_WINDOW_SEC / 2)
        axs[1].set_title("Zoomed-Out Overlay")

        fig.suptitle(f"Overlay of Template and Detected Signal (Both Normalized)", fontsize=14, fontweight='bold')
        fig.text(0.5, 0.935, f"Matching Template: {template_id} - Match No.: {i} | Match Index: {match_index} | UTC Match Time: {match_time} | Corr: {row['correlation']:.5f} | Duration of Raw Waveform used: {duration_hours} Hours", ha='center', va='top', fontsize=12, style='italic')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(img_overlay_path)
        plt.close()
    except Exception as e:
        print(f"⚠️ Overlay failed for {seg_id}: {e}")
        img_overlay_path = ""


    # ------------------ Metadata Entry ------------------
    event_meta.append({
        "event_id": seg_id,
        "template_id": template_id,
        "match_time": row["match_time"],
        "match_index": match_index,
        "correlation": row["correlation"],
        "station": row["station"],
        "network": row["network"],
        "channel": row["channel"],
        "image_path": img_raw_path.replace("\\", "/"),
        "image_norm_path": img_norm_path.replace("\\", "/"),
        "image_overlay_path": img_overlay_path.replace("\\", "/"),
        "sac_path": seg_file.replace("\\", "/"),
        "template_path": tpl_file.replace("\\", "/")
    })

# --- Save metadata ---
df_meta = pd.DataFrame(event_meta)
df_meta.to_csv(DETECTION_METADATA, index=False)

print(f"✅ Metadata saved to: {DETECTION_METADATA}")
print(f"🖼️ Detection plots saved to: {IMG_RAW_DIR}, {IMG_NORM_DIR}, {IMG_NORM_OVERLAY_DIR}")
