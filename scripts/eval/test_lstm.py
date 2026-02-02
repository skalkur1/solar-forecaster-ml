import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "horizons_processed"
RESULTS_DIR = PROJECT_ROOT / "results"
HORIZON_RANGE = range(1, 13)

TARGET_COL = "ac_power_norm"
DROP_COLS = ["ac_power_on_forecast_issue"]
TIME_COL = "valid_time_mst"
SYSTEM_COLS = ["system_10", "system_4", "system_50", "system_51"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PVSequenceDataset(Dataset):
    def __init__(self, df, lookback, feature_cols, target_col):
        self.X, self.y, self.system_ids = self.build_sequences(
            df, lookback, feature_cols, target_col
        )

    def build_sequences(self, df, lookback, feature_cols, target_col):
        X, y, sys = [], [], []

        df = df.sort_values([TIME_COL, "system_id"])

        for system_id, g in df.groupby("system_id"):
            g = g.sort_values(TIME_COL).reset_index(drop=True)
            times = pd.to_datetime(g[TIME_COL])

            for i in range(lookback, len(g)):
                window_times = times.iloc[i - lookback:i + 1]

                # strict 1-hour continuity
                if not all(
                    (window_times.iloc[j] - window_times.iloc[j - 1]).total_seconds() == 3600
                    for j in range(1, len(window_times))
                ):
                    continue

                X.append(g.iloc[i - lookback:i][feature_cols].values)
                y.append(g.iloc[i][target_col])
                sys.append(system_id)

        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32),
            torch.tensor(np.array(sys), dtype=torch.long),
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.system_ids[idx]


class LSTMModel(nn.Module):
    def __init__(self, num_features, num_systems, emb_dim=4, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_systems, emb_dim)
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + emb_dim, 1)

    def forward(self, x, system_id):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        emb = self.embedding(system_id)
        out = torch.cat([last_hidden, emb], dim=1)
        return self.fc(out).squeeze(1)


def build_system_mapping(df):
    """Return a mapping from system column names to integer IDs."""
    system_cols = [c for c in df.columns if c.startswith("system_")]
    mapping = {name: i for i, name in enumerate(system_cols)}
    return mapping, system_cols


def extract_system_id(df, system_cols, mapping):
    """Return the integer system_id for each row."""
    if "system_id" in df.columns:
        return df["system_id"].values
    system_ids = []
    for _, row in df.iterrows():
        active_col = row[system_cols].idxmax()
        system_ids.append(mapping[active_col])
    return np.array(system_ids)


def eval_per_system(model, df, loader):
    """Compute per-system RMSE."""
    model.eval()
    preds_dict = {sys_id: [] for sys_id in df['system_id'].unique()}
    trues_dict = {sys_id: [] for sys_id in df['system_id'].unique()}

    with torch.no_grad():
        for X, y, sys in loader:
            X, sys = X.to(DEVICE), sys.to(DEVICE)
            p = model(X, sys).cpu().numpy()
            y_np = y.numpy()
            for i, sys_id in enumerate(sys.cpu().numpy()):
                preds_dict[sys_id].append(p[i])
                trues_dict[sys_id].append(y_np[i])

    rmse_dict = {}
    for sys_id in preds_dict:
        if len(preds_dict[sys_id]) == 0:
            continue
        rmse = mean_squared_error(trues_dict[sys_id], preds_dict[sys_id]) ** 0.5
        rmse_dict[sys_id] = rmse
    
    return rmse_dict


def load_model_from_json(model_path):
    """Load model architecture and weights from JSON file."""
    with open(model_path, 'r') as f:
        model_data = json.load(f)
 
    num_features = model_data['num_features']
    num_systems = model_data['num_systems']
    emb_dim = model_data.get('emb_dim', 4)
    hidden_dim = model_data.get('hidden_dim', 64)
    system_mapping = model_data['system_mapping']
    
    model = LSTMModel(
        num_features=num_features,
        num_systems=num_systems,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim
    )
    
    state_dict = {}
    for key, value in model_data['model_state_dict'].items():
        state_dict[key] = torch.tensor(value)
    
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    return model, model_data


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

    model, model_data = load_model_from_json(model_path)
    
    lookback = model_data.get('lookback', 6)
    system_mapping = model_data['system_mapping']
    
    feature_cols = [
        "dswrf", "tcc", "t2m", "cos_zenith",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "ac_power_norm"
    ]

    test_df = pd.read_csv(test_path)
    

    for c in DROP_COLS:
        if c in test_df.columns:
            test_df.drop(columns=c, inplace=True)

    system_cols = list(system_mapping.keys())
    test_df["system_id"] = extract_system_id(test_df, system_cols, system_mapping)

    
    test_ds = PVSequenceDataset(test_df, lookback, feature_cols, TARGET_COL)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y, sys in test_loader:
            X, sys = X.to(DEVICE), sys.to(DEVICE)
            p = model(X, sys).cpu().numpy()
            preds.extend(p)
            trues.extend(y.numpy())

    test_rmse = mean_squared_error(trues, preds) ** 0.5
    test_mae = mean_absolute_error(trues, preds)

    per_system_rmse = eval_per_system(model, test_df, test_loader)

    print(
        f"Horizon {horizon} | TEST RMSE: {test_rmse:.4f} | "
        f"TEST MAE: {test_mae:.4f}"
    )

    results = {
        "horizon": horizon,
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "per_system_rmse": {int(k): float(v) for k, v in per_system_rmse.items()},
        "features_used": feature_cols,
        "lookback": lookback
    }

    results_path = output_dir / f"horizon{horizon}_test_metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    row = {
        "horizon": horizon,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
    }

    id_to_name = {v: k for k, v in system_mapping.items()}
    for sys_id, rmse_val in per_system_rmse.items():
        sys_name = id_to_name.get(sys_id, f"system_{sys_id}")
        row[f"{sys_name}_rmse"] = rmse_val

    return row

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved LSTM models on test data"
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
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("horizon")
        summary_csv_path = output_dir / "test_metrics_all_horizons.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSaved summary CSV to: {summary_csv_path}")


if __name__ == "__main__":
    main()