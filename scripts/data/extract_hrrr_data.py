import os
import gc
import time
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import wraps
from herbie import Herbie
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = SCRIPT_DIR.parent.parent
data_dir = PROJECT_ROOT / "data" / "hrrr_raw"    

LAT, LON = 39.7404, -105.1772
START, END = "2019-01-01 00:00", "2023-02-28 23:00"  
FORECAST_HORIZONS = range(1, 13)
OUTFILE = data_dir / "hrrr_forecasts_full.csv"
ERROR_LOG = data_dir / "hrrr_errors.log"
PROGRESS_LOG = data_dir / "hrrr_progress.log"
# EC2-optimized settings (ajust max workers if on local machine)
BUFFER_SIZE = 50        
MAX_WORKERS = 48       
MAX_RETRIES = 3         
RETRY_DELAY = 2         
CHECKPOINT_HOURS = 24   



points_df = pd.DataFrame(
    {"longitude": [LON], "latitude": [LAT]},
    index=["site_1"]
)


if os.path.exists(OUTFILE):
    try:
        existing = pd.read_csv(OUTFILE, parse_dates=["run_time_utc"])
        completed_keys = set(zip(existing.run_time_utc.astype(str), existing.horizon_hr))
        print(f"📥 Resuming: {len(completed_keys)} records already completed")
    except Exception:
        print("⚠️ Output file corrupted or empty. Restarting fresh.")
        completed_keys = set()
else:
    completed_keys = set()


def log_error(run_time_str, fxx, error_type, error_msg):
    """Log errors to CSV file"""
    with open(ERROR_LOG, "a") as f:
        f.write(f"{run_time_str},{fxx},{error_type},{error_msg}\n")

def log_progress(message):
    """Log progress milestones"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(PROGRESS_LOG, "a") as f:
        f.write(log_msg + "\n")

def retry_on_failure(max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Retry decorator for handling transient S3/network errors"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except FileNotFoundError as e:
                    # Don't retry if GRIB file doesn't exist
                    if attempt == 0:
                        log_error(args[0], args[1], "FileNotFound", str(e))
                    return None
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Final attempt failed
                        log_error(args[0], args[1], f"{type(e).__name__}_after_retries", str(e))
                        print(f"⚠️ Failed after {max_retries} retries: {args[0]} f{args[1]:02d} → {e}")
                        return None
                    wait_time = delay * (2 ** attempt)
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

@retry_on_failure(max_retries=MAX_RETRIES)
def fetch_hrrr(run_time_str, fxx):
    """
    Fetch HRRR data for a specific run time and forecast horizon.
    Returns a dict with data (or NaN for missing variables), or None if already completed.
    """
    key = (run_time_str, fxx)
    if key in completed_keys:
        return None
   
    row_data = {
        "run_time_utc": run_time_str,
        "horizon_hr": fxx,
        "dswrf": np.nan,
        "tcc": np.nan,
        "t2m": np.nan,
    }
    
    H = Herbie(
        run_time_str,
        model="hrrr",
        product="sfc",
        fxx=fxx,
        save=False,
        grib="pygrib"
    )

    ds = H.xarray(
        search="DSWRF:surface|TCDC:entire atmosphere|TMP:2 m above ground"
    )
    
    if isinstance(ds, list):
        ds = xr.merge(ds)

    # Pick points
    pt = ds.herbie.pick_points(points=points_df)
    
    # Map of output names to possible variable names in the dataset
    var_mappings = {
        'dswrf': ['dswrf', 'sdswrf', 'DSWRF'],
        'tcc': ['tcc', 'tcdc', 'TCC', 'TCDC'],
        't2m': ['t2m', 'tmp', 'TMP', 't']
    }
    
    # Try to extract each variable independently
    missing_vars = []
    for output_name, possible_names in var_mappings.items():
        found = False
        for var_name in possible_names:
            if var_name in pt:
                try:
                    row_data[output_name] = pt[var_name].isel(point=0).item()
                    found = True
                    break
                except Exception:
                    continue
        
        if not found:
            missing_vars.append(output_name)
    
    # Log if any variables were missing (but still return the row!)
    if missing_vars:
        log_error(run_time_str, fxx, "missing_variables", ",".join(missing_vars))
    # Clean up
    ds.close()
    del ds, pt
    # Delete the GRIB file to save disk space
    try:
        if hasattr(H, 'grib'):
            grib_path = H.get_localFilePath()
            if os.path.exists(grib_path):
                os.remove(grib_path)
    except Exception:
        pass  
    
    gc.collect()
    
    return row_data

def cleanup_data_directory(): #clean up for RAM overflow
    try:
        data_dir = os.path.join(os.getcwd(), 'data')
        if os.path.exists(data_dir):
            # Get all GRIB files older than 1 hour
            current_time = time.time()
            deleted_count = 0
            
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith('.grib2') or file.endswith('.idx'):
                        file_path = os.path.join(root, file)
                        # Delete files older than 1 hour
                        if current_time - os.path.getmtime(file_path) > 3600:
                            try:
                                os.remove(file_path)
                                deleted_count += 1
                            except Exception:
                                pass
            
            if deleted_count > 0:
                log_progress(f"🗑️  Cleaned up {deleted_count} old GRIB files")
    except Exception as e:
        pass  


def calculate_progress(completed_count, total_count, start_time):
    """Calculate and return progress statistics"""
    elapsed = time.time() - start_time
    progress_pct = (completed_count / total_count) * 100
    
    if completed_count > 0:
        avg_time_per_record = elapsed / completed_count
        remaining_records = total_count - completed_count
        eta_seconds = avg_time_per_record * remaining_records
        eta_hours = eta_seconds / 3600
    else:
        eta_hours = 0
    
    return progress_pct, eta_hours


def main():
    # Initialize logs
    if not os.path.exists(ERROR_LOG):
        with open(ERROR_LOG, "w") as f:
            f.write("run_time,horizon,error_type,error_msg\n")
    
    if not os.path.exists(PROGRESS_LOG):
        with open(PROGRESS_LOG, "w") as f:
            f.write("HRRR Data Collection Progress Log\n")
            f.write("=" * 60 + "\n")
    
    log_progress(f"Starting HRRR data collection")
    log_progress(f"Period: {START} to {END}")
    log_progress(f"Workers: {MAX_WORKERS}")
    log_progress(f"Forecast horizons: {list(FORECAST_HORIZONS)}")
    
    rows_buffer = []
    run_times = pd.date_range(START, END, freq="1H").strftime("%Y-%m-%d %H:%M")
    
    total_expected = len(run_times) * len(FORECAST_HORIZONS)
    completed = len(completed_keys)
    start_time = time.time()
    
    log_progress(f"Total expected records: {total_expected:,}")
    log_progress(f"Already completed: {completed:,}")
    log_progress(f"Remaining: {(total_expected - completed):,}")
    
    checkpoint_counter = 0
    last_checkpoint_time = datetime.now()
    last_progress_update = 0
    
    #Create a single pool and submit ALL tasks upfront
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        all_futures = {}
        for run_time_str in run_times:
            for fxx in FORECAST_HORIZONS:
                future = executor.submit(fetch_hrrr, run_time_str, fxx)
                all_futures[future] = (run_time_str, fxx)
        
        log_progress(f"Submitted {len(all_futures):,} tasks to {MAX_WORKERS} workers")
        

        for future in as_completed(all_futures):
            result = future.result()
            if result: 
                rows_buffer.append(result)
                completed += 1
            
            # Progress logging every 1000 records
            if completed - last_progress_update >= 1000:
                progress_pct, eta_hours = calculate_progress(completed, total_expected, start_time)
                log_progress(
                    f"Progress: {completed:,}/{total_expected:,} ({progress_pct:.2f}%) | "
                    f"ETA: {eta_hours:.1f} hours"
                )
                last_progress_update = completed
            
            # Flush buffer periodically
            if len(rows_buffer) >= BUFFER_SIZE:
                pd.DataFrame(rows_buffer).to_csv(
                    OUTFILE,
                    mode="a",
                    header=not os.path.exists(OUTFILE),
                    index=False
                )
                rows_buffer.clear()
                gc.collect()
        
        # Final cleanup
        cleanup_data_directory()
    
    # Final flush
    if rows_buffer:
        pd.DataFrame(rows_buffer).to_csv(
            OUTFILE,
            mode="a",
            header=not os.path.exists(OUTFILE),
            index=False
        )
        log_progress(f"Final flush: {len(rows_buffer)} rows")
    
    # Final statistics
    elapsed_hours = (time.time() - start_time) / 3600
    log_progress("=" * 60)
    log_progress("🚀 HRRR download complete!")
    log_progress(f"Total time: {elapsed_hours:.2f} hours")
    log_progress(f"Total records: {completed:,}/{total_expected:,}")
    if os.path.exists(OUTFILE):
        df = pd.read_csv(OUTFILE)
        log_progress(f"\n📊 Data Completeness:")
        log_progress(f"   Total rows: {len(df):,}")
        log_progress(f"   DSWRF missing: {df['dswrf'].isna().sum():,} ({df['dswrf'].isna().mean()*100:.1f}%)")
        log_progress(f"   TCC missing: {df['tcc'].isna().sum():,} ({df['tcc'].isna().mean()*100:.1f}%)")
        log_progress(f"   T2M missing: {df['t2m'].isna().sum():,} ({df['t2m'].isna().mean()*100:.1f}%)")
    
    log_progress(f"\n   Check {ERROR_LOG} for details on failures")
    log_progress("=" * 60)
if __name__ == "__main__":
    main()