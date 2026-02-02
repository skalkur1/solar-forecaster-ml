import os
import json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "horizons_processed"    
MODEL_DIR = PROJECT_ROOT / "models" / "xgboost_main"
HORIZON_RANGE = range(1, 13)

TARGET_COL = "ac_power_norm"

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist"
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50
TIME_FEATURES = ["hour_sin", "hour_cos",
    "doy_sin", "doy_cos", "ac_power_on_forecast_issue",
    "cos_zenith", "dswrf", "t2m", "tcc", "system_10","system_4","system_50","system_51"] #Use all time features except valid_time_mst and target ac_power_norm


def train_horizon_baseline(horizon):
    print(f"\nTraining Horizon {horizon} (XGBoost Main Model)")

    split_dir = os.path.join(DATA_DIR, f"horizon{horizon}", "splits")
    train_path = os.path.join(split_dir, "train.csv")
    val_path = os.path.join(split_dir, "val.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Missing train/val for horizon {horizon}, skipping.")
        return
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    for df in [train_df, val_df]:
        df.drop(columns=["valid_time_mst"], errors="ignore", inplace=True)

    X_train = train_df[TIME_FEATURES]
    y_train = train_df[TARGET_COL]
    X_val = val_df[TIME_FEATURES]
    y_val = val_df[TARGET_COL]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    bst = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False
    )

    val_preds = bst.predict(dval)
    val_rmse = mean_squared_error(y_val, val_preds) ** 0.5
    val_mae = mean_absolute_error(y_val, val_preds)

    print(f"Horizon {horizon} | VAL RMSE: {val_rmse:.4f} | VAL MAE: {val_mae:.4f}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"horizon{horizon}_baseline.json")
    bst.save_model(model_path)
    metrics = {
        "horizon": horizon,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "best_iteration": bst.best_iteration
    }

    metrics_path = os.path.join(
        MODEL_DIR, f"horizon{horizon}_baseline_metrics.json"
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

def main():
    for h in HORIZON_RANGE:
        train_horizon_baseline(h)

if __name__ == "__main__":
    main()
