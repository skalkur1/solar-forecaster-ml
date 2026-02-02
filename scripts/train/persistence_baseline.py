import os
import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

PV_FILE = PROJECT_ROOT / "data" / "pvdaq_processed" / "pvdaq_all_systems.csv"
HORIZONS = range(1, 13)  
OUTPUT_DIR = PROJECT_ROOT / "results" / "persistence_baseline"
OUTPUT_FILE = OUTPUT_DIR / "persistence_baseline.csv"

def main():
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not PV_FILE.exists():
        raise FileNotFoundError(f"PV data file not found: {PV_FILE}")
    
    df = pd.read_csv(PV_FILE)

    df['measured_on'] = pd.to_datetime(df['measured_on'], errors='coerce')
    df = df.dropna(subset=['measured_on', 'ac_power_norm', 'system_id'])

    df['system_id'] = df['system_id'].astype(str)

 
    df = df.sort_values(['system_id', 'measured_on'])

    results = []

    for horizon in HORIZONS:
        print(f"Processing horizon {horizon}h...")

        df[f'ac_power_prev_{horizon}h'] = (
            df.groupby('system_id')['ac_power_norm']
              .shift(horizon)
        )
  
        valid_rows = df.dropna(subset=[f'ac_power_prev_{horizon}h'])
        
        corr = valid_rows['ac_power_norm'].corr(valid_rows[f'ac_power_prev_{horizon}h'])
        #checks how well previous value at (target - horizon) predicts current ac power
        rmse = np.sqrt(np.mean((valid_rows['ac_power_norm'] - valid_rows[f'ac_power_prev_{horizon}h'])**2))
        
        results.append({
            'horizon': horizon,
            'test_rmse': rmse,
            'correlation': corr,
            'num_points': len(valid_rows)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nPersistence baseline saved to {OUTPUT_FILE}") #output to csv in results/persistence_baseline
    print(results_df)


if __name__ == "__main__":
    main()