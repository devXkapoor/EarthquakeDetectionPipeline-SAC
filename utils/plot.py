import os
import shutil
import json
import sys
from datetime import timedelta, timezone
from obspy import read, UTCDateTime
from obspy.signal.filter import bandpass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.ticker import FuncFormatter

# -------- Load Config --------
def load_config():
    with open("config_plots.json", "r") as f:
        return json.load(f)

cfg = load_config()
cfg1 = cfg["step1_preprocess"]
cfg_gen = cfg["general"]
cfg2 = cfg["step2_matrix_profile"]

# -------- Compute Derived Values --------
trim_duration = int(cfg1["trim_duration"])
sampling_rate = int(cfg_gen["sampling_rate"])
window_sec = float(cfg2["window_sec"])
use_gpu = cfg_gen["use_gpu"]

# -------- Format Output Folder Name --------
def format_duration_name(seconds):
    if seconds < 60:
        return f"{seconds:02d}Seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes:02d}Minutes"
    else:
        hours = seconds // 3600
        return f"{hours:03d}Hours"

duration_str = format_duration_name(trim_duration)
engine_str = "GPU" if use_gpu else "CPU"
folder_name = f"{duration_str}-{int(sampling_rate)}Hz-{window_sec:.1f}s-{engine_str}"
output_root = os.path.join("processed_plots", folder_name)
processed_dir = os.path.join(output_root, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)

# -------- Logging Setup --------
log_path = os.path.join(output_root, "log_pipeline.txt")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
sys.stdout = open(log_path, 'w', encoding='utf-8')

# -------- STEP 1: LOAD RAW --------
print("🔄 Loading raw waveform...")
st = read(cfg1["raw_path"])
tr = st[0]
fs = tr.stats.sampling_rate
start_time = tr.stats.starttime

print(f"✅ Loaded: {tr.stats.starttime}  - {tr.stats.endtime}")
print(f"⏱️ Duration: {tr.stats.starttime} to {tr.stats.endtime} ({len(tr.data)} samples)")

# -------- STEP 2: TRIM --------
trim_start = start_time + cfg1["trim_start"]
trim_end = trim_start + trim_duration
print(f"✂️ Trimming to: {trim_start} to {trim_end} ...")
tr.trim(starttime=trim_start, endtime=trim_end)

# -------- STEP 3: FILTER --------
print("⚙️ Bandpass filtering...")
data = bandpass(tr.data, freqmin=cfg1["freqmin"], freqmax=cfg1["freqmax"], df=fs, corners=4, zerophase=True)
tr.data = data.astype(np.float64)

# -------- STEP 4: SAVE --------
processed_trace_path = os.path.join(processed_dir, "processed_trace.sac")
print(f"💾 Saving processed trace to {processed_trace_path}...")
tr.write(processed_trace_path, format="SAC")
print("✅ Preprocessing complete.")

# -------- STEP 5: PLOT AND SAVE --------
print("📈 Plotting and saving processed trace...")

# Load the CSV file with the template metadata
csv_path = "processed_plots/001Hours-250Hz-1.0s-GPU/metadata_filtered.csv"
df = pd.read_csv(csv_path)

# Convert the start_time column to datetime objects
df['start_time'] = pd.to_datetime(df['start_time'])

# Read the processed trace
st_processed = read(processed_trace_path)
tr_processed = st_processed[0]

# Ensure timezone-aware datetime in UTC
start_utc = tr_processed.stats.starttime.datetime.replace(tzinfo=timezone.utc)
npts = tr_processed.stats.npts
delta = 1.0 / tr_processed.stats.sampling_rate
utc_times = [start_utc + timedelta(seconds=i * delta) for i in range(npts)]

plt.figure(figsize=(16, 9))
plt.plot(utc_times, tr_processed.data, color='black')
plt.title(f"Processed Trace: {tr_processed.stats.starttime} - {tr_processed.stats.endtime}")
plt.xlabel("UTC Time")
plt.ylabel("Amplitude")
plt.grid(True)

# Format with milliseconds (3 decimal places), in UTC
def time_with_milliseconds(x, pos):
    dt = mdates.num2date(x, tz=timezone.utc)
    return dt.strftime('%H:%M:%S.%f')[:-3]

plt.gca().xaxis.set_major_formatter(FuncFormatter(time_with_milliseconds))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()

# Highlight the start_times from the CSV file
for start_time in df['start_time']:
    # Convert CSV start_time to matplotlib date format
    start_time_mpl = mdates.date2num(start_time)
    plt.axvline(x=start_time_mpl, color='red', linestyle='--', label="Start Time")

plt.tight_layout()

# Save the plot and show it
processed_plot_path = os.path.join(processed_dir, "processed_trace_plot.png")
plt.savefig(processed_plot_path)
plt.show()

print(f"🖼️ Processed trace plot saved to: {processed_plot_path}")
