import subprocess
import json
import os
import sys
import datetime
import time
from pathlib import Path

# Load configuration
with open("config.json") as f:
    config = json.load(f)

# Extract parameters for naming result folder
duration_sec = config["step1_preprocess"]["trim_duration"]
sampling_rate = config["general"]["sampling_rate"]
window_sec = config["step2_matrix_profile"]["window_sec"]
use_gpu = config["general"]["use_gpu"]

# Duration formatting
def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:02.0f}Seconds"
    elif seconds < 3600:
        return f"{int(seconds // 60):02}Minutes"
    else:
        return f"{int(seconds // 3600):03}Hours"

duration_str = format_duration(duration_sec)
engine = "GPU" if use_gpu else "CPU"
result_string = f"{duration_str}-{sampling_rate}Hz-{window_sec:.1f}s-{engine}"
result_folder = os.path.join("results", result_string)

# Paths
log_file = os.path.join(result_folder, "log_time.txt")
os.makedirs(result_folder, exist_ok=True)

# Redirect stdout and stderr to log file and console
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.stdout, open(log_file, "w", encoding="utf-8"))
sys.stderr = sys.stdout

# Step scripts
steps = [
    "pipeline/step1_preprocess.py",
    "pipeline/step2_matrix_profile.py",
    "pipeline/step3_filter_templates.py",
    "pipeline/step4_template_matching.py",
    "pipeline/step5_filter_detections.py",
    "pipeline/step6_generate_outputs.py"
]

# Track timing
step_timings = []
pipeline_start = time.time()
print("🚀 Starting Motif-Based Earthquake Detection Pipeline...\n")

for step in steps:
    step_name = Path(step).stem
    print(f"\n▶️ Running: {step_name}")
    start_time = datetime.datetime.now()
    print(f"⏰ Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    step_start = time.time()
    subprocess.run(["python", step], check=True)
    step_end = time.time()

    end_time = datetime.datetime.now()
    elapsed = step_end - step_start
    total_elapsed = step_end - pipeline_start

    print(f"✅ Completed: {step_name}")
    print(f"⏱️ End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🕓 Time Taken: {elapsed:.2f} seconds")
    print(f"🧭 Total Elapsed: {total_elapsed:.2f} seconds\n")

    step_timings.append({
        "step": step_name,
        "start": start_time,
        "end": end_time,
        "duration": elapsed,
        "total_elapsed": total_elapsed
    })

# Final summary
print("\n✅ Motif-Based Pipeline Completed Successfully!\n")
print("📋 Execution Summary:")
for step in step_timings:
    print(f" - {step['step']}: {step['duration']:.2f}s | Start: {step['start'].strftime('%H:%M:%S')} → End: {step['end'].strftime('%H:%M:%S')}")

total_time = time.time() - pipeline_start
print(f"\n⏳ Total Pipeline Time: {total_time:.2f} seconds")
print(f"📝 Logs saved to: {log_file}")
print(f"📂 Results stored in: {result_folder}")
