import os
import pandas as pd
import numpy as np
from datetime import timedelta
from astral import LocationInfo
from astral.sun import elevation, zenith
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = SCRIPT_DIR.parent.parent       

HRRR_FILE = PROJECT_ROOT / "data" / "hrrr_raw" /"hrrr_forecasts_full.csv"
PV_FILE = PROJECT_ROOT / "data" / "pvdaq_processed" / "pvdaq_all_systems.csv"  # now contains all systems
OUT_DIR = PROJECT_ROOT / "data" / "horizons_processed"

#Site Coordinates
BOULDER_LAT = 39.7406
BOULDER_LON = -105.1774
TIMEZONE = "UTC"

SOLAR_ELEVATION_THRESHOLD = 5  

LOCATION = LocationInfo(
    name="Boulder",
    region="USA",
    timezone=TIMEZONE,
    latitude=BOULDER_LAT,
    longitude=BOULDER_LON,
)

def solar_geometry(dt_mst):
    dt_utc = dt_mst + timedelta(hours=7)
    elev = elevation(LOCATION.observer, dt_utc)
    zen = zenith(LOCATION.observer, dt_utc)
    cos_zen = np.cos(np.deg2rad(zen))
    return elev, zen, cos_zen

def is_daylight_mst(dt_mst):
    dt_utc = dt_mst + timedelta(hours=7)
    return elevation(LOCATION.observer, dt_utc) > SOLAR_ELEVATION_THRESHOLD

def main():
    #Load all pv data
    pv = pd.read_csv(PV_FILE)
    pv_time_col = pv.columns[0]
    pv[pv_time_col] = pd.to_datetime(pv[pv_time_col], errors="coerce")
    pv = pv.dropna(subset=[pv_time_col])
    pv = pv.rename(columns={pv_time_col: "valid_time_mst"})
    pv["valid_time_mst"] = pv["valid_time_mst"].dt.floor("h")
    pv["system_id"] = pv["system_id"].astype(str)
    # Load HRRR
    hrrr = pd.read_csv(HRRR_FILE)
    hrrr["run_time_utc"] = pd.to_datetime(hrrr["run_time_utc"], errors="coerce")
    hrrr = hrrr.dropna(subset=["run_time_utc"])
    # Loop automatically over horizons 1-12
    for horizon_hr in range(1, 13):
        print(f"Processing horizon {horizon_hr}...")
        output_dir = OUT_DIR / ("horizon" + str(horizon_hr))
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"horizon{horizon_hr}_data.csv")
        
        # Fileter horizons
        hrrr_h = hrrr[hrrr["horizon_hr"] == horizon_hr].copy()
        if hrrr_h.empty:
            continue

        # Convert utc to mst and compute valid time
        hrrr_h["valid_time_utc"] = hrrr_h["run_time_utc"] + pd.to_timedelta(hrrr_h["horizon_hr"], unit="h")
        hrrr_h["valid_time_mst"] = (hrrr_h["valid_time_utc"] - timedelta(hours=7)).dt.floor("h")

        # Daylight filter
        hrrr_h = hrrr_h[hrrr_h["valid_time_mst"].apply(is_daylight_mst)]
        if hrrr_h.empty:
            continue

        # Restrict HRRR to PV availability
        pv_start = pv["valid_time_mst"].min()
        pv_end = pv["valid_time_mst"].max()
        hrrr_h = hrrr_h[(hrrr_h["valid_time_mst"] >= pv_start) & (hrrr_h["valid_time_mst"] <= pv_end)]

        # Merge HRRR across all systems
        systems = pv["system_id"].unique()
        hrrr_h = (
            hrrr_h.assign(key=1)
            .merge(pd.DataFrame({"system_id": systems, "key": 1}), on="key")
            .drop(columns="key")
        )

        # Merge PV power (time + system)
        merged = hrrr_h.merge(
            pv[["valid_time_mst", "system_id", "ac_power_norm"]],
            on=["valid_time_mst", "system_id"],
            how="left"
        )

        # Drop rows without AC power
        merged = merged.dropna(subset=["ac_power_norm"])
        # Compute forecast issue time (run_time - 7h)
        merged["forecast_issue_time_mst"] = (merged["run_time_utc"] - timedelta(hours=7)).dt.floor("h")
        # Create a lookup table: (system_id, valid_time_mst) -> ac_power_norm
        pv_lookup = pv.set_index(["system_id", "valid_time_mst"])["ac_power_norm"]

        def get_prev_ac(row):
            key = (row["system_id"], row["forecast_issue_time_mst"])
            return pv_lookup.get(key, np.nan)  # np.nan if not available

        # Apply
        merged["ac_power_on_forecast_issue"] = merged.apply(get_prev_ac, axis=1)

       
        # Add solar geometry features
        geom = merged["valid_time_mst"].apply(solar_geometry)
        #merged["solar_elevation"] = geom.apply(lambda x: x[0])
        #merged["solar_zenith"] = geom.apply(lambda x: x[1])
        merged["cos_zenith"] = geom.apply(lambda x: x[2])

        # Time encodings
        hour = merged["valid_time_mst"].dt.hour
        doy = merged["valid_time_mst"].dt.dayofyear
        merged["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        merged["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        merged["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        merged["doy_cos"] = np.cos(2 * np.pi * doy / 365)

        # Target cleanup
        merged["ac_power_norm"] = merged["ac_power_norm"].clip(lower=0.0, upper=1.05)
        merged["ac_power_on_forecast_issue"] = merged["ac_power_on_forecast_issue"].clip(lower=0.0, upper=1.05)

        # Weather cleaning
        merged = merged[
            (merged["dswrf"] >= 0) &
            (merged["tcc"].between(0, 100)) &
            (merged["cos_zenith"] > 0)
        ]

        # Ensure categorical
        merged["system_id"] = merged["system_id"].astype(str)

        # Sort by timestamp then system
        merged = merged.sort_values(["valid_time_mst", "system_id"]).reset_index(drop=True)

        # Save CSV (keep timestamp for later splitting)
        merged.to_csv(output_file, index=False)
        print(f"Horizon {horizon_hr} done: {len(merged)} rows written to {output_file}")

if __name__ == "__main__":
    main()
