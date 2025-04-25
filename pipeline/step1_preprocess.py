import os
import shutil
from obspy import read, UTCDateTime
from obspy.signal.filter import bandpass
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import json
import sys

# -------- Load Config --------
with open("config.json", "r") as f:
    cfg = json.load(f)

cfg1 = cfg["step1_preprocess"]
cfg_gen = cfg["general"]

RAW_PATH = cfg1["raw_input_path"]
OUTPUT_PATH = cfg1["output_path"]
OUTPUT_PLOT = cfg1["output_plot"]
TRIM_START = cfg1["trim_start"]
TRIM_DURATION = cfg1["trim_duration"]
FREQMIN = cfg1["freqmin"]
FREQMAX = cfg1["freqmax"]
SAMPLING_RATE = cfg_gen["sampling_rate"]

# -------- Compute dynamic folder name --------
def format_duration_name(seconds):
    if seconds < 60:
        return f"{seconds:02d}Seconds"
    elif seconds < 3600:
        return f"{seconds//60:02d}Minutes"
    else:
        return f"{seconds//3600:03d}Hours"

duration_str = format_duration_name(TRIM_DURATION)
engine_str = "GPU" if cfg_gen["use_gpu"] else "CPU"
window_str = f"{cfg['step2_matrix_profile']['window_sec']:.1f}s"
rate_str = f"{int(SAMPLING_RATE)}Hz"
folder_name = f"{duration_str}-{rate_str}-{window_str}-{engine_str}"
base_dir = os.path.join("results", folder_name)

# -------- Redirect stdout to log --------
os.makedirs(base_dir, exist_ok=True)
sys.stdout = open(os.path.join(base_dir, "log_pipeline.txt"), "a", encoding="utf-8")

# -------- Output Path --------
NEW_RAW_PATH = os.path.join(base_dir, "data", "raw")
NEW_RAW_PATH_FILE = os.path.join(base_dir, "data", "raw", "SAC_Data.sac")
NEW_OUTPUT_PATH = os.path.join("data", "processed")
NEW_OUTPUT_PATH_FILE = os.path.join(NEW_OUTPUT_PATH, "processed_trace.sac")
PROCESSED_PATH = os.path.join(base_dir, "data", "processed")
PROCESSED_FILE = os.path.join(PROCESSED_PATH, "processed_trace.sac")
PROCESSED_PLOT = os.path.join(base_dir, "data", "processed", "processed_trace_plot.png")


if os.path.exists(PROCESSED_PATH):
    shutil.rmtree(PROCESSED_PATH)
os.makedirs(PROCESSED_PATH, exist_ok=True)

if os.path.exists(NEW_RAW_PATH):
    shutil.rmtree(NEW_RAW_PATH)
os.makedirs(NEW_RAW_PATH, exist_ok=True)

if os.path.exists(NEW_OUTPUT_PATH):
    shutil.rmtree(NEW_OUTPUT_PATH)
os.makedirs(NEW_OUTPUT_PATH, exist_ok=True)

# -------- STEP 1: LOAD RAW --------
print("🔄 Loading raw waveform...")
st = read(RAW_PATH)
tr = st[0]
tr.write(NEW_RAW_PATH_FILE, format="SAC")
fs = tr.stats.sampling_rate
start_time = tr.stats.starttime
print(f"✅ Loaded: {tr.id}")
print(f"⏱️ Duration: {tr.stats.starttime} to {tr.stats.endtime} ({len(tr.data)} samples)")

# -------- STEP 2: TRIM --------
trim_start = start_time + TRIM_START
trim_end = trim_start + TRIM_DURATION
print(f"✂️ Trimming to: {trim_start} to {trim_end} ...")
tr.trim(starttime=trim_start, endtime=trim_end)

# -------- STEP 3: FILTER --------
print(f"⚙️ Bandpass filtering {FREQMIN} - {FREQMAX} Hz...")
filtered_data = bandpass(tr.data, freqmin=FREQMIN, freqmax=FREQMAX, df=fs, corners=4, zerophase=True)
tr.data = filtered_data.astype(np.float64)

# -------- STEP 4: SAVE --------
print(f"💾 Saving processed trace to {PROCESSED_PATH}...")
tr.write(PROCESSED_FILE, format="SAC")
tr.write(NEW_OUTPUT_PATH_FILE, format="SAC")

# -------- STEP 5: PLOT --------
print("📈 Plotting processed trace...")
st_processed = read(PROCESSED_FILE)
tr_processed = st_processed[0]

start_utc = tr_processed.stats.starttime.datetime
npts = tr_processed.stats.npts
delta = 1.0 / tr_processed.stats.sampling_rate
utc_times = [start_utc + timedelta(seconds=i * delta) for i in range(npts)]

plt.figure(figsize=(16, 9))
plt.plot(utc_times, tr_processed.data, color='black')
plt.title(f"Processed Trace: {tr_processed.id}")
plt.xlabel("UTC Time")
plt.ylabel("Amplitude")
plt.grid(True)

def precise_formatter(x, pos):
    dt = mdates.num2date(x)
    return dt.strftime('%H:%M:%S.') + f"{dt.microsecond:06d}Z"

plt.gca().xaxis.set_major_formatter(FuncFormatter(precise_formatter))
plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig(PROCESSED_PLOT)
plt.savefig(OUTPUT_PLOT)
plt.show()

print(f"🖼️ Processed trace plot saved to: {PROCESSED_PLOT}")
print("✅ Preprocessing complete.")
