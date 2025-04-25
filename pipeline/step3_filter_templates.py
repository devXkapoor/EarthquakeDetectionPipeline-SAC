import os
import shutil
import pandas as pd
import numpy as np
from obspy import read
import json
import sys

# --- Load Config ---
with open("config.json", "r") as f:
    cfg = json.load(f)

cfg1 = cfg["step1_preprocess"]
cfg2 = cfg["step2_matrix_profile"]
cfg3 = cfg["step3_filter_templates"]
cfg_gen = cfg["general"]

# --- Compute dynamic folder name ---
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

# --- Redirect stdout to log ---
sys.stdout = open(os.path.join(base_dir, "log_pipeline.txt"), "a", encoding="utf-8")

# --- Config Paths ---
TEMPLATE_METADATA = os.path.join(base_dir, "templates", "metadata.csv")
TRACE_DIR = os.path.join(base_dir, "templates", "traces")
IMAGE_DIR = os.path.join(base_dir, "templates", "images")
OUTPUT_METADATA = os.path.join(base_dir, "templates", "metadata_filtered.csv")
OUTPUT_TRACE_DIR = os.path.join(base_dir, "templates", "traces_filtered")
OUTPUT_IMAGE_DIR = os.path.join(base_dir, "templates", "images_filtered")
GROUP_GAP_THRESHOLD = cfg3["group_gap_threshold"]

# --- Clean output directories ---
if os.path.exists(OUTPUT_TRACE_DIR):
    shutil.rmtree(OUTPUT_TRACE_DIR)
if os.path.exists(OUTPUT_IMAGE_DIR):
    shutil.rmtree(OUTPUT_IMAGE_DIR)

os.makedirs(OUTPUT_TRACE_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# --- Load metadata ---
df = pd.read_csv(TEMPLATE_METADATA)
df["start_index"] = df["start_index"].astype(int)
df = df.sort_values(by="start_index").reset_index(drop=True)

# --- Helper: Signal energy (RMS) ---
def signal_energy(trace_path):
    try:
        tr = read(trace_path)[0]
        data = tr.data.astype(np.float64)
        return np.sqrt(np.mean(np.square(data)))
    except Exception:
        return -np.inf

# --- Filter motifs ---
filtered_rows = []
group = [df.iloc[0]]

for i in range(1, len(df)):
    curr = df.iloc[i]
    prev = df.iloc[i - 1]

    if curr["start_index"] - prev["start_index"] <= GROUP_GAP_THRESHOLD:
        group.append(curr)
    else:
        best_row = max(group, key=lambda row: signal_energy(os.path.join(TRACE_DIR, f"{row['template_id']}.sac")))
        filtered_rows.append(best_row)
        group = [curr]

if group:
    best_row = max(group, key=lambda row: signal_energy(os.path.join(TRACE_DIR, f"{row['template_id']}.sac")))
    filtered_rows.append(best_row)

# --- Save filtered metadata ---
filtered_df = pd.DataFrame(filtered_rows)
filtered_df.to_csv(OUTPUT_METADATA, index=False)

# --- Copy corresponding trace and image files ---
for row in filtered_df.itertuples(index=False):
    template_id = row.template_id
    src_trace = os.path.join(TRACE_DIR, f"{template_id}.sac")
    dst_trace = os.path.join(OUTPUT_TRACE_DIR, f"{template_id}.sac")

    src_img = os.path.join(IMAGE_DIR, f"{template_id}.png")
    dst_img = os.path.join(OUTPUT_IMAGE_DIR, f"{template_id}.png")

    if os.path.exists(src_trace):
        shutil.copy(src_trace, dst_trace)
    else:
        print(f"⚠️ Missing trace: {src_trace}")

    if os.path.exists(src_img):
        shutil.copy(src_img, dst_img)
    else:
        print(f"⚠️ Missing image: {src_img}")

# --- Summary ---
print(f"✅ Filtered metadata saved to: {OUTPUT_METADATA}")
print(f"✅ Filtered traces saved to:   {OUTPUT_TRACE_DIR}")
print(f"✅ Filtered images saved to:   {OUTPUT_IMAGE_DIR}")
print(f"📉 Reduced from {len(df)} to {len(filtered_df)} motif templates")
