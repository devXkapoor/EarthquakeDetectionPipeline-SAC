import os
import pandas as pd
import json
import sys

# --- Load Config ---
with open("config.json", "r") as f:
    cfg = json.load(f)

cfg1 = cfg["step1_preprocess"]
cfg2 = cfg["step2_matrix_profile"]
cfg_gen = cfg["general"]

# --- Compute Results Folder Name ---
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

# --- Redirect stdout to log_pipeline.txt ---
sys.stdout = open(os.path.join(base_dir, "log_pipeline.txt"), "a", encoding="utf-8")

# --- Paths ---
TEMPLATE_METADATA = os.path.join(base_dir, "templates", "metadata.csv")
INPUT_CATALOG = os.path.join(base_dir, "detections", "catalog.csv")
OUTPUT_CATALOG = os.path.join(base_dir, "detections", "catalog_filtered.csv")

MATCH_INDEX_COLUMN = "match_index"
CORRELATION_COLUMN = "correlation"
TEMPLATE_COLUMN = "template_id"

# --- Check if catalog exists and is non-empty ---
if not os.path.exists(INPUT_CATALOG) or os.stat(INPUT_CATALOG).st_size == 0:
    print("⚠️ Detection catalog is missing or empty. Skipping filtering step.")
    pd.DataFrame(columns=[
        "template_id", "match_index", "match_time",
        "correlation", "station", "network", "channel"
    ]).to_csv(OUTPUT_CATALOG, index=False)
    print(f"📁 Empty filtered catalog written to: {OUTPUT_CATALOG}")
    sys.exit(0)

# --- Load metadata and catalog ---
templates_df = pd.read_csv(TEMPLATE_METADATA)
catalog_df = pd.read_csv(INPUT_CATALOG)

# Get unique template IDs from metadata
unique_templates = templates_df[TEMPLATE_COLUMN].unique()

# Sort catalog by template_id and match_index
catalog_df = catalog_df.sort_values(by=[TEMPLATE_COLUMN, MATCH_INDEX_COLUMN]).reset_index(drop=True)

# Final filtered rows
filtered_rows = []

# Process each template_id
for template_id in unique_templates:
    group = catalog_df[catalog_df[TEMPLATE_COLUMN] == template_id].copy()
    group = group.sort_values(by=MATCH_INDEX_COLUMN).reset_index(drop=True)

    i = 0
    while i < len(group):
        current_row = group.iloc[i]
        cluster = [current_row]

        # Cluster together consecutive match_index rows
        j = i + 1
        while j < len(group) and (group.iloc[j][MATCH_INDEX_COLUMN] - group.iloc[j - 1][MATCH_INDEX_COLUMN]) <= 1:
            cluster.append(group.iloc[j])
            j += 1

        # Select the best one (highest correlation)
        best_row = max(cluster, key=lambda x: x[CORRELATION_COLUMN])
        filtered_rows.append(best_row)

        # Move to the next unprocessed row
        i = j

# --- Save filtered catalog ---
if filtered_rows:
    filtered_df = pd.DataFrame(filtered_rows)
else:
    # Ensure the expected columns exist even if no rows are kept
    filtered_df = pd.DataFrame(columns=[TEMPLATE_COLUMN, MATCH_INDEX_COLUMN, CORRELATION_COLUMN])

filtered_df = filtered_df.sort_values(by=[TEMPLATE_COLUMN, CORRELATION_COLUMN], ascending=[True, False]
).reset_index(drop=True)

os.makedirs(os.path.dirname(OUTPUT_CATALOG), exist_ok=True)
filtered_df.to_csv(OUTPUT_CATALOG, index=False)

print(f"✅ Filtered catalog saved to: {OUTPUT_CATALOG}")
print(f"📉 Reduced from {len(catalog_df)} to {len(filtered_df)} detections")

