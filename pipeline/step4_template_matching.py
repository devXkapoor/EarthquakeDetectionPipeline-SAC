import os
import shutil
import numpy as np
import pandas as pd
from obspy import read, UTCDateTime
import json
import sys

# -------- Load Config --------
with open("config.json", "r") as f:
    cfg = json.load(f)

cfg1 = cfg["step1_preprocess"]
cfg2 = cfg["step2_matrix_profile"]
cfg4 = cfg["step4_template_matching"]
cfg_gen = cfg["general"]

# -------- Compute folder name from config --------
def format_duration_name(seconds):
    if seconds < 60:
        return f"{seconds:02d}Seconds"
    elif seconds < 3600:
        return f"{seconds//60:02d}Minutes"
    else:
        return f"{seconds//3600:03d}Hours"

duration_str = format_duration_name(cfg1["trim_duration"])
engine_str = "GPU" if cfg_gen["use_gpu"] else "CPU"
window_str = f"{cfg2['window_sec']:.1f}s"
rate_str = f"{int(cfg_gen['sampling_rate'])}Hz"
folder_name = f"{duration_str}-{rate_str}-{window_str}-{engine_str}"
base_dir = os.path.join("results", folder_name)

# -------- Redirect stdout to log --------
sys.stdout = open(os.path.join(base_dir, "log_pipeline.txt"), "a", encoding="utf-8")

# -------- Path Setup --------
TEMPLATE_DIR = os.path.join(base_dir, "templates", "traces_filtered")
TEMPLATE_METADATA_FILE = os.path.join(base_dir, "templates", "metadata_filtered.csv")
PROCESSED_TRACE = os.path.join("data", "processed", "processed_trace.sac")
DETECTIONS_DIR = os.path.join(base_dir, "detections")
OUTPUT_CSV = os.path.join(DETECTIONS_DIR, "catalog.csv")

CORR_THRESHOLD = cfg4["correlation_threshold"]
MAX_MATCHES_PER_TEMPLATE = cfg4["max_matches_per_template"]

# -------- Load waveform --------
print("🔄 Loading preprocessed waveform...")
st = read(PROCESSED_TRACE)
tr = st[0]
fs = tr.stats.sampling_rate
data = tr.data.astype(np.float64)
print(f"✅ Trace loaded: {tr.id} | Duration = {len(data)/fs:.1f}s")

# -------- Load metadata --------
template_metadata = pd.read_csv(TEMPLATE_METADATA_FILE)

# -------- Prepare output directory --------
if os.path.exists(DETECTIONS_DIR):
    shutil.rmtree(DETECTIONS_DIR)
os.makedirs(DETECTIONS_DIR, exist_ok=True)

detections = []

# -------- Match templates --------
print(f"🔍 Scanning templates in {TEMPLATE_DIR} ...")
template_files = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith(".sac")]

for tpl_file in sorted(template_files):
    tpl_path = os.path.join(TEMPLATE_DIR, tpl_file)
    tpl_id = os.path.splitext(tpl_file)[0]
    tpl = read(tpl_path)[0]
    tpl_data = tpl.data.astype(np.float64)

    m = len(tpl_data)
    if m > len(data):
        print(f"⚠️ Skipping {tpl_id}: longer than input trace")
        continue

    meta_row = template_metadata[template_metadata["template_id"] == tpl_id]
    template_start_index = int(meta_row["start_index"].values[0]) if not meta_row.empty else None

    print(f"📈 Matching template {tpl_id} ...")

    template_norm = (tpl_data - np.mean(tpl_data)) / np.std(tpl_data)

    corr_vals = []
    indices = []

    for i in range(len(data) - m):
        if template_start_index is not None and abs(i - template_start_index) < m:
            continue

        window = data[i:i + m]
        if np.std(window) == 0:
            continue

        window_norm = (window - np.mean(window)) / np.std(window)
        corr = np.dot(template_norm, window_norm) / m
        corr_vals.append(corr)
        indices.append(i)

    if not corr_vals:
        print(f"⚠️ No valid comparisons for {tpl_id}")
        continue

    corr_vals = np.array(corr_vals)
    indices = np.array(indices)

    match_idxs = np.where(corr_vals > CORR_THRESHOLD)[0]
    match_times = [tr.stats.starttime + (indices[i] / fs) for i in match_idxs]

    for i, idx in enumerate(match_idxs[:MAX_MATCHES_PER_TEMPLATE]):
        detections.append({
            "template_id": tpl_id,
            "match_index": int(indices[idx]),
            "match_time": str(match_times[i]),
            "correlation": float(corr_vals[idx]),
            "station": tr.stats.station,
            "network": tr.stats.network,
            "channel": tr.stats.channel
        })

    print(f"✅ {len(match_idxs)} matches found for {tpl_id}")

# -------- Save Detection Catalog --------
df = pd.DataFrame(detections)

if df.empty:
    print("⚠️ No matches found in any template. Saving empty catalog with header only.")
    df = pd.DataFrame(columns=[
        "template_id", "match_index", "match_time",
        "correlation", "station", "network", "channel"
    ])

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n📁 Saved detection catalog: {OUTPUT_CSV}")
