import os
import sys
import csv
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = SCRIPT_DIR.parent.parent       


def extract_ac_power_from_csv(filepath, ac_col_name, system_id):
    """
    Reads a messy CSV file line-by-line and extracts:
    - First column (assumed timestamp)
    - AC power column

    Returns only rows that:
    - Fall exactly on the hour
    - Are from year 2019 or later
    """

    rows = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)

        header = None
        ac_col_idx = None
        first_col_name = None

        for row in reader:
            if not row:
                continue

            # Detect header
            if header is None:
                header = row
                first_col_name = header[0]

                if ac_col_name not in header:
                    return None

                ac_col_idx = header.index(ac_col_name)
                continue

            if ac_col_idx >= len(row) or len(row) == 0:
                continue

            first_col_value = row[0]
            value = row[ac_col_idx]

            try:
                value = float(value)
            except (ValueError, TypeError):
                continue

            rows.append({
                first_col_name: first_col_value,
                "ac_power": value,
                "system_id": system_id
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    #Parse timestamps
    df[first_col_name] = pd.to_datetime(df[first_col_name], errors="coerce")

    # Drop rows with invalid timestamps
    df = df.dropna(subset=[first_col_name])

    # keep only rows exactly on the hour
    df = df[
        (df[first_col_name].dt.minute == 0) &
        (df[first_col_name].dt.second == 0)
    ]

    # kep only year >= 2019
    df = df[df[first_col_name].dt.year >= 2019]

    if df.empty:
        return None

    return df


def main():
    if len(sys.argv) != 4:
        print("Usage: python process_individual_systems.py <ac_power_column_name> <system_id> <system_capacity [Watts]>")
        sys.exit(1)

    ac_col_name = sys.argv[1]
    system_id = sys.argv[2]

    try:
        system_capacity = float(sys.argv[3])
        if system_capacity <= 0:
            raise ValueError
    except ValueError:
        print("system_capacity must be a positive number")
        sys.exit(1)

    input_dir = PROJECT_ROOT / "data" / "pvdaq_raw" / ("system" + (system_id))
    output_dir = PROJECT_ROOT / "data" / "pvdaq_processed" 
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"system{system_id}_combined.csv")

    all_dfs = []

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(".csv"):
                continue

            filepath = os.path.join(root, fname)
            print(f"Processing {filepath}")

            df = extract_ac_power_from_csv(filepath, ac_col_name, system_id)

            if df is not None:
                all_dfs.append(df)

    if not all_dfs:
        print("No valid data found.")
        sys.exit(1)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    #Sort time series chronologically 
    timestamp_col = combined_df.columns[0]
    combined_df = combined_df.sort_values(by=timestamp_col).reset_index(drop=True)

    combined_df["ac_power_norm"] = combined_df["ac_power"] / system_capacity
    combined_df = combined_df.drop(columns=["ac_power"])


    combined_df.to_csv(output_path, index=False)

    print(f"Combined CSV written to: {output_path}")
    print(f"Total rows: {len(combined_df)}")



if __name__ == "__main__":
    main()
