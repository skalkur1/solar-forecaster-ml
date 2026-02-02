import os
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "horizons_processed"    


HORIZON_RANGE = range(1, 13)

TIMESTAMP_COL = "valid_time_mst"
# Training: Jan 2019 - Dec 2021
TRAIN_START = "2019-01-01"
TRAIN_END = "2021-12-31"
# Validation: Jan 2022 - Aug 2022
VAL_START = "2022-01-01"
VAL_END = "2022-08-31"
# Testing: Sep 2022 - Feb 2023
TEST_START = "2022-09-01"
TEST_END = "2023-02-28"


def split_horizon_data(horizon_int):
    horizon_dir = os.path.join(DATA_DIR, f"horizon{horizon_int}")
    input_file = os.path.join(horizon_dir, f"horizon{horizon_int}_data.csv")
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}, skipping horizon {horizon_int}")
        return

    print(f"Processing horizon {horizon_int}")
    df = pd.read_csv(input_file)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"{TIMESTAMP_COL} column not found in {input_file}")
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors='coerce')
    df = df.dropna(subset=[TIMESTAMP_COL])

  
    # Identify all systems
    all_systems = sorted(df["system_id"].astype(str).unique())
    system_cols = [f"system_{s}" for s in all_systems]


    # Split by date ranges
    train_df = df[(df[TIMESTAMP_COL] >= TRAIN_START) & (df[TIMESTAMP_COL] <= TRAIN_END)].copy()
    val_df = df[(df[TIMESTAMP_COL] >= VAL_START) & (df[TIMESTAMP_COL] <= VAL_END)].copy()
    test_df = df[(df[TIMESTAMP_COL] >= TEST_START) & (df[TIMESTAMP_COL] <= TEST_END)].copy()

    drop_cols = ["run_time_utc", "horizon_hr","forecast_issue_time_mst"]
    if "valid_time_utc" in df.columns:
        drop_cols.append("valid_time_utc")

    train_df = train_df.drop(columns=drop_cols)
    val_df = val_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    def one_hot_encode_systems(split_df):
        split_df["system_id"] = split_df["system_id"].astype(str)
        dummies = pd.get_dummies(split_df["system_id"], prefix="system")
        # Ensure all system columns exist
        dummies = dummies.reindex(columns=system_cols, fill_value=0)
        split_df = pd.concat([split_df.drop(columns=["system_id"]), dummies], axis=1)
        return split_df

    train_df = one_hot_encode_systems(train_df)
    val_df = one_hot_encode_systems(val_df)
    test_df = one_hot_encode_systems(test_df)
    splits_dir = os.path.join(horizon_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

    print(f"Saved splits for horizon {horizon_int}:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val:   {len(val_df)} rows")
    print(f"  Test:  {len(test_df)} rows")


def main():
    for h in HORIZON_RANGE:
        split_horizon_data(h)


if __name__ == "__main__":
    main()
