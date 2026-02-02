import os
import json
import argparse
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "horizons_processed"
RESULTS_DIR = PROJECT_ROOT / "results"
HORIZON_RANGE = range(1, 13)

TARGET_COL = "ac_power_norm"
SYSTEM_COLS = ["system_10", "system_4", "system_50", "system_51"]


def compute_per_system_rmse(df, preds):
    """Return dict of per-system RMSE if system columns exist."""
    rmse_dict = {}

    for sys_col in SYSTEM_COLS:
        if sys_col not in df.columns:
            continue

        mask = df[sys_col] == 1
        if mask.sum() == 0:
            continue

        rmse = mean_squared_error(
            df.loc[mask, TARGET_COL],
            preds[mask]
        ) ** 0.5

        rmse_dict[sys_col] = rmse

    return rmse_dict


def test_horizon(horizon, model_dir, output_dir):
    print(f"\n=== Testing Horizon {horizon} ===")

    split_dir = DATA_DIR / f"horizon{horizon}" / "splits"
    test_path = split_dir / "test.csv"
    model_path = model_dir / f"horizon{horizon}_baseline.json"

    if not test_path.exists():
        print(f"Missing test data for horizon {horizon}, skipping.")
        return None

    if not model_path.exists():
        print(f"Missing model for horizon {horizon}, skipping.")
        return None


    test_df = pd.read_csv(test_path)
    test_df.drop(columns=["valid_time_mst"], errors="ignore", inplace=True)

    #Load model FIRST to get feature names
    bst = xgb.Booster()
    bst.load_model(model_path)

    model_features = bst.feature_names
    if model_features is None:
        raise ValueError(
            f"Model at {model_path} has no feature names. "
            "Was it trained using pandas / DMatrix?"
        )

   
    missing = set(model_features) - set(test_df.columns)
    if missing:
        raise ValueError(
            f"Horizon {horizon}: missing features in test data: {missing}"
        )

    X_test = test_df[model_features]
    y_test = test_df[TARGET_COL]

    dtest = xgb.DMatrix(X_test, label=y_test)

    #Predict
    preds = bst.predict(dtest)

    test_rmse = mean_squared_error(y_test, preds) ** 0.5
    test_mae = mean_absolute_error(y_test, preds)

    per_system_rmse = compute_per_system_rmse(test_df, preds)

    print(
        f"Horizon {horizon} | TEST RMSE: {test_rmse:.4f} | "
        f"TEST MAE: {test_mae:.4f}"
    )
    results = {
        "horizon": horizon,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "per_system_rmse": per_system_rmse,
        "features_used": model_features
    }

    results_path = output_dir / f"horizon{horizon}_test_metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    row = {
        "horizon": horizon,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
    }

    for sys_col in SYSTEM_COLS:
        row[f"{sys_col}_rmse"] = per_system_rmse.get(sys_col, None)

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved XGBoost models on test data"
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Directory containing saved models (relative to /models)"
    )

    args = parser.parse_args()
    model_name = args.model_dir
    model_dir = PROJECT_ROOT / "models" / model_name
    output_dir = RESULTS_DIR / model_name
    os.makedirs(output_dir, exist_ok=True)

    if not model_dir.is_dir():
        raise ValueError(f"Provided model_dir does not exist: {model_dir}")

    summary_rows = []

    for h in HORIZON_RANGE:
        row = test_horizon(h, model_dir, output_dir)
        if row is not None:
            summary_rows.append(row)

    #write summary csv
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("horizon")
        summary_csv_path = output_dir / "test_metrics_all_horizons.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSaved summary CSV to: {summary_csv_path}")


if __name__ == "__main__":
    main()
