import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = SCRIPT_DIR.parent.parent     

data_dir = PROJECT_ROOT / "data" / "pvdaq_processed"
output_file = data_dir / "pvdaq_all_systems.csv"

COLUMN_RENAME_MAP = {
    "timestamp": "measured_on",
    "systemid": "system_id"
}

dfs = []

for csv_file in data_dir.glob("*.csv"):
    try:
        df = pd.read_csv(csv_file)

        if df.empty:
            continue

        dfs.append(df)

    except pd.errors.EmptyDataError:
        print(f"Skipping empty file: {csv_file.name}")

# Combine all files
combined_df = pd.concat(dfs, ignore_index=True)
combined_df = combined_df.rename(columns=COLUMN_RENAME_MAP)

#ensure datetime type
combined_df["measured_on"] = pd.to_datetime(combined_df["measured_on"])
# Force system_id to string
combined_df["system_id"] = combined_df["system_id"].astype(str)

#keep only relevant columns and sort
combined_df = combined_df[
    ["measured_on", "ac_power_norm", "system_id"]
]
combined_df = combined_df.sort_values(
    by=["measured_on", "system_id"]
)
combined_df = combined_df.drop_duplicates(
    subset=["measured_on", "system_id"]
)
combined_df.to_csv(output_file, index=False)
print(f"Combined file written to: {output_file}")
