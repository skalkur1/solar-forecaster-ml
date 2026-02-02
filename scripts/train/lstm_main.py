import os
import json
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
MODEL_DIR = PROJECT_ROOT / "models" / "lstm_main"
HORIZON_RANGE = range(1, 13)

LOOKBACK = 6
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_COL = "ac_power_norm"
DROP_COLS = ["ac_power_on_forecast_issue"]

TIME_COL = "valid_time_mst"

FEATURE_COLS = [
    "dswrf", "tcc", "t2m", "cos_zenith",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "ac_power_norm"
]

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


#Helper functions
def build_system_mapping(df):
    #Return a mapping from system column names to integer IDs.
    system_cols = [c for c in df.columns if c.startswith("system_")]
    mapping = {name: i for i, name in enumerate(system_cols)}
    return mapping, system_cols

def extract_system_id(df, system_cols, mapping):
    #Return the integer system_id for each row.
    if "system_id" in df.columns:
        return df["system_id"].values
    system_ids = []
    for _, row in df.iterrows():
        active_col = row[system_cols].idxmax()
        system_ids.append(mapping[active_col])
    return np.array(system_ids)




#Eval per system
def eval_per_system(model, df, loader):
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

def train_horizon(horizon):
    print(f"\nTraining Horizon {horizon} (LSTM)")
    split_dir = os.path.join(DATA_DIR, f"horizon{horizon}", "splits")
    train_path = os.path.join(split_dir, "train.csv")
    val_path = os.path.join(split_dir, "val.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Missing train/val for horizon {horizon}, skipping.")
        return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # drop forbidden column
    for df in [train_df, val_df]:
        for c in DROP_COLS:
            if c in df.columns:
                df.drop(columns=c, inplace=True)

    mapping, system_cols = build_system_mapping(train_df)
    for df in [train_df, val_df]:
        df["system_id"] = extract_system_id(df, system_cols, mapping)

    num_systems = len(mapping)

    train_ds = PVSequenceDataset(train_df, LOOKBACK, FEATURE_COLS, TARGET_COL)
    val_ds = PVSequenceDataset(val_df, LOOKBACK, FEATURE_COLS, TARGET_COL)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = LSTMModel(num_features=len(FEATURE_COLS), num_systems=num_systems).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    #training loop
    for epoch in range(EPOCHS):
        model.train()
        for X, y, sys in train_loader:
            X, y, sys = X.to(DEVICE), y.to(DEVICE), sys.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X, sys)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
    #validation
    def eval_loader(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, y, sys in loader:
                X, sys = X.to(DEVICE), sys.to(DEVICE)
                p = model(X, sys).cpu().numpy()
                preds.extend(p)
                trues.extend(y.numpy())
        return mean_squared_error(trues, preds) ** 0.5, mean_absolute_error(trues, preds)

    val_rmse, val_mae = eval_loader(val_loader)

    print(f"Horizon {horizon} | VAL RMSE: {val_rmse:.4f} | VAL MAE: {val_mae:.4f}")

    #per-system rmse
    val_system_rmse = eval_per_system(model, val_df, val_loader)

    print(f"Per-system VAL RMSE: {val_system_rmse}")

    #create out file
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_state = {
        'model_state_dict': {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()},
        'num_features': len(FEATURE_COLS),
        'num_systems': num_systems,
        'system_mapping': mapping,
        'emb_dim': 4,
        'hidden_dim': 64,
        'lookback': LOOKBACK
    }
    model_path = os.path.join(MODEL_DIR, f"horizon{horizon}_baseline.json")
    with open(model_path, "w") as f:
        json.dump(model_state, f, indent=2)

    # Save metrics
    metrics = {
        "horizon": horizon,
        "val_rmse": float(val_rmse),
        "val_mae": float(val_mae),
        "per_system_val_rmse": {int(k): float(v) for k, v in val_system_rmse.items()}
    }

    metrics_path = os.path.join(MODEL_DIR, f"horizon{horizon}_baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")

def main():
    for h in HORIZON_RANGE:
        train_horizon(h)


if __name__ == "__main__":
    main()