"""
Optimized CMEMS Fetch — Date-grouped, resumable, with progress logging.

Strategy: Group profiles by unique date. For each date, make ONE spatial query
covering all profile locations on that date. Extract individual values locally
from the downloaded xarray dataset. This is faster than per-profile queries
when multiple profiles share a date.

Usage:
    python scripts/cmems_fetch_optimized.py --limit 50     # fetch 50 dates
    python scripts/cmems_fetch_optimized.py --limit 0      # fetch ALL remaining dates
    python scripts/cmems_fetch_optimized.py --status       # show status only
"""
import argparse
import sys
import io
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# CMEMS product
DATASET_ID = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
BUFFER_DEG = 0.25

# Cache directory
CACHE_DIR = Path('data/raw/cmems_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Target depths for GLORYS temperature extraction
TARGET_DEPTHS = [10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTH_TOLERANCE = 5.0

# Progress log
PROGRESS_LOG = CACHE_DIR / '_progress.json'


def get_cache_key(lat, lon, date_str):
    return f"cmems_{date_str}_{lat:.2f}_{lon:.2f}"


def load_cached_keys():
    return {f.stem for f in CACHE_DIR.glob('cmems_*.json')}


def load_progress():
    if PROGRESS_LOG.exists():
        with open(PROGRESS_LOG) as f:
            return json.load(f)
    return {"dates_processed": 0, "profiles_cached": 0, "profiles_failed": 0,
            "start_time": None, "last_update": None}


def save_progress(prog):
    prog["last_update"] = datetime.now().isoformat()
    with open(PROGRESS_LOG, 'w') as f:
        json.dump(prog, f, indent=2)


def fetch_date_batch(date_str, profiles_for_date):
    """
    Fetch SSS + currents + temperature for ALL profiles on a single date
    in ONE spatial query. Returns dict mapping profile index to extracted values.
    """
    import copernicusmarine
    import xarray as xr

    lats = [p['latitude'] for p in profiles_for_date]
    lons = [p['longitude'] for p in profiles_for_date]

    # Bounding box covering all profiles on this date
    lat_min = min(lats) - BUFFER_DEG
    lat_max = max(lats) + BUFFER_DEG
    lon_min = min(lons) - BUFFER_DEG
    lon_max = max(lons) + BUFFER_DEG

    try:
        result = copernicusmarine.subset(
            dataset_id=DATASET_ID,
            variables=['so', 'uo', 'vo', 'thetao'],
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_latitude=lat_min, maximum_latitude=lat_max,
            start_datetime=date_str, end_datetime=date_str,
            minimum_depth=0.4, maximum_depth=1050.0,
        )

        ds = xr.open_dataset(result.file_path)

        extracted = {}
        for p in profiles_for_date:
            idx = p['idx']
            lat, lon = p['latitude'], p['longitude']

            # Find nearest grid cell
            lat_idx = np.argmin(np.abs(ds.latitude.values - lat))
            lon_idx = np.argmin(np.abs(ds.longitude.values - lon))

            # Extract SSS
            sss_val = None
            if 'so' in ds:
                v = float(ds.so.isel(latitude=lat_idx, longitude=lon_idx, depth=0).values)
                if not np.isnan(v):
                    sss_val = round(v, 4)

            # Extract currents
            cu_val, cv_val = None, None
            if 'uo' in ds:
                v = float(ds.uo.isel(latitude=lat_idx, longitude=lon_idx, depth=0).values)
                if not np.isnan(v):
                    cu_val = round(v, 4)
            if 'vo' in ds:
                v = float(ds.vo.isel(latitude=lat_idx, longitude=lon_idx, depth=0).values)
                if not np.isnan(v):
                    cv_val = round(v, 4)

            # Extract temperature at target depths
            temp_profile = {}
            if 'thetao' in ds:
                cmems_depths = ds.depth.values
                for target_d in TARGET_DEPTHS:
                    depth_idx = np.argmin(np.abs(cmems_depths - target_d))
                    actual_depth = cmems_depths[depth_idx]
                    if abs(actual_depth - target_d) <= DEPTH_TOLERANCE:
                        v = float(ds.thetao.isel(
                            latitude=lat_idx, longitude=lon_idx, depth=depth_idx
                        ).values)
                        if not np.isnan(v):
                            temp_profile[str(target_d)] = round(v, 4)

            extracted[idx] = {
                'sss': sss_val,
                'current_u': cu_val,
                'current_v': cv_val,
                'temp_profile': temp_profile,
                'source': 'cmems',
                'timestamp': datetime.now().isoformat(),
            }

        ds.close()
        try:
            os.remove(str(result.file_path))
        except:
            pass

        return extracted

    except Exception as e:
        # If batch fails, return empty for all profiles
        return {p['idx']: {
            'sss': None, 'current_u': None, 'current_v': None,
            'temp_profile': {}, 'source': 'cmems_failed',
            'error': str(e), 'timestamp': datetime.now().isoformat(),
        } for p in profiles_for_date}


def show_status():
    df = pd.read_parquet('data/processed/oceanembed_real_dataset.parquet')
    df['date_str'] = df['date'].astype(str).str[:10]
    cached = load_cached_keys()

    cached_count = 0
    for _, row in df.iterrows():
        key = get_cache_key(row['latitude'], row['longitude'], row['date_str'])
        if key in cached:
            cached_count += 1

    prog = load_progress()
    print(f"Dataset: {len(df)} profiles")
    print(f"Cached: {cached_count}/{len(df)} ({100*cached_count/len(df):.1f}%)")
    print(f"Dates processed: {prog.get('dates_processed', 0)}")
    print(f"Last update: {prog.get('last_update', 'never')}")


def main():
    parser = argparse.ArgumentParser(description='Optimized CMEMS fetch (date-grouped)')
    parser.add_argument('--limit', type=int, default=0, help='Max dates to process (0=all)')
    parser.add_argument('--status', action='store_true', help='Show status only')
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    # Load dataset
    df = pd.read_parquet('data/processed/oceanembed_real_dataset.parquet')
    df['date_str'] = df['date'].astype(str).str[:10]
    cached = load_cached_keys()

    print(f"Dataset: {len(df)} profiles, {len(cached)} cached")

    # Group uncached profiles by date
    date_groups = {}
    for idx, row in df.iterrows():
        key = get_cache_key(row['latitude'], row['longitude'], row['date_str'])
        if key not in cached:
            d = row['date_str']
            if d not in date_groups:
                date_groups[d] = []
            date_groups[d].append({
                'idx': idx,
                'latitude': row['latitude'],
                'longitude': row['longitude'],
            })

    # Sort dates by number of profiles (most first)
    sorted_dates = sorted(date_groups.items(), key=lambda x: -len(x[1]))

    if args.limit > 0:
        sorted_dates = sorted_dates[:args.limit]

    print(f"Dates to process: {len(sorted_dates)} ({sum(len(v) for v in date_groups.values())} profiles)")

    # Progress tracking
    prog = load_progress()
    if not prog.get("start_time"):
        prog["start_time"] = datetime.now().isoformat()

    total_success = 0
    total_fail = 0
    start_time = time.time()

    for di, (date_str, profiles) in enumerate(sorted_dates):
        elapsed = time.time() - start_time
        if di > 0:
            rate = di / elapsed * 60  # dates per minute
            remaining = len(sorted_dates) - di
            eta_min = remaining / max(rate, 0.01)
            print(f"\n[{di+1}/{len(sorted_dates)}] {date_str} ({len(profiles)} profiles) | "
                  f"OK={total_success} fail={total_fail} | "
                  f"{elapsed:.0f}s elapsed, ~{eta_min:.0f}min remaining", end='', flush=True)
        else:
            print(f"\n[{di+1}/{len(sorted_dates)}] {date_str} ({len(profiles)} profiles)...", end='', flush=True)

        extracted = fetch_date_batch(date_str, profiles)

        for idx, data in extracted.items():
            row = df.iloc[idx]
            cache_key = get_cache_key(row['latitude'], row['longitude'], date_str)
            cache_path = CACHE_DIR / f"{cache_key}.json"
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)

            if data.get('sss') is not None or data.get('temp_profile'):
                total_success += 1
            else:
                total_fail += 1

        # Checkpoint progress
        prog["dates_processed"] = prog.get("dates_processed", 0) + 1
        prog["profiles_cached"] = len(load_cached_keys())
        prog["profiles_failed"] = total_fail
        save_progress(prog)

        # Log every 25 dates
        if (di + 1) % 25 == 0:
            total_cached = len(load_cached_keys())
            pct = 100 * total_cached / len(df)
            print(f"\n  === CHECKPOINT: {total_cached}/{len(df)} ({pct:.1f}%) cached, "
                  f"{total_success} ok, {total_fail} failed ===")

    total_time = time.time() - start_time
    total_cached = len(load_cached_keys())

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Dates processed: {len(sorted_dates)}")
    print(f"New successes: {total_success}, New failures: {total_fail}")
    print(f"Total cached: {total_cached}/{len(df)} ({100*total_cached/len(df):.1f}%)")
    print(f"Time: {total_time:.0f}s ({total_time/max(len(sorted_dates),1):.1f}s per date)")

    save_progress(prog)


if __name__ == '__main__':
    main()
