"""
Batch CMEMS Fetch — SSS + Currents + GLORYS Temperature Profiles
Fetches real data from Copernicus Marine for every Argo profile location/date.
Caches per-profile; resumes from cache if interrupted.

Usage:
    python scripts/fetch_cmems_batch.py --limit 100     # fetch first 100 uncached profiles
    python scripts/fetch_cmems_batch.py --limit 0        # fetch ALL uncached profiles
    python scripts/fetch_cmems_batch.py --status          # show cache status only
"""
import argparse, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime

# CMEMS product
DATASET_ID = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
BUFFER_DEG = 0.25  # spatial buffer around each profile

# Cache directory
CACHE_DIR = Path('data/raw/cmems_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Target depths for GLORYS temperature extraction
TARGET_DEPTHS = [10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTH_TOLERANCE = 5.0  # meters — accept CMEMS depth within this tolerance


def get_cache_key(lat, lon, date_str):
    """Generate a cache filename for a profile."""
    return f"cmems_{date_str}_{lat:.2f}_{lon:.2f}"


def load_cache_status():
    """Check which profiles are already cached."""
    cached = set()
    for f in CACHE_DIR.glob('*.json'):
        cached.add(f.stem)
    return cached


def fetch_cmems_one_profile(lat, lon, date_str):
    """
    Fetch SSS, surface currents, and GLORYS temperature profile for one profile.
    Returns dict with values or None on failure.
    """
    import copernicusmarine
    
    lat_min = lat - BUFFER_DEG
    lat_max = lat + BUFFER_DEG
    lon_min = lon - BUFFER_DEG
    lon_max = lon + BUFFER_DEG
    
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
        
        # Extract SSS (surface, ~0.5m)
        sss_val = float(np.nanmean(ds.so.isel(depth=0).values)) if 'so' in ds else None
        if sss_val is not None and np.isnan(sss_val):
            sss_val = None
        
        # Extract surface currents (depth ~0.5m)
        uo_val = float(np.nanmean(ds.uo.isel(depth=0).values)) if 'uo' in ds else None
        vo_val = float(np.nanmean(ds.vo.isel(depth=0).values)) if 'vo' in ds else None
        if uo_val is not None and np.isnan(uo_val):
            uo_val = None
        if vo_val is not None and np.isnan(vo_val):
            vo_val = None
        
        # Extract temperature at target depths
        if 'thetao' in ds:
            cmems_depths = ds.depth.values
            temp_profile = {}
            for target_d in TARGET_DEPTHS:
                # Find nearest depth level
                depth_idx = np.argmin(np.abs(cmems_depths - target_d))
                actual_depth = cmems_depths[depth_idx]
                if abs(actual_depth - target_d) <= DEPTH_TOLERANCE:
                    val = float(np.nanmean(ds.thetao.isel(depth=depth_idx).values))
                    if not np.isnan(val):
                        temp_profile[str(target_d)] = round(val, 4)
        else:
            temp_profile = {}
        
        ds.close()
        
        # Clean up downloaded file
        try:
            os.remove(str(result.file_path))
        except:
            pass
        
        return {
            'sss': round(sss_val, 4) if sss_val is not None else None,
            'current_u': round(uo_val, 4) if uo_val is not None else None,
            'current_v': round(vo_val, 4) if vo_val is not None else None,
            'temp_profile': temp_profile,
            'source': 'cmems',
            'timestamp': datetime.now().isoformat(),
        }
        
    except Exception as e:
        return {
            'sss': None,
            'current_u': None,
            'current_v': None,
            'temp_profile': {},
            'source': 'cmems_failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }


def save_cache(cache_key, data):
    """Save one profile's data to cache."""
    path = CACHE_DIR / f"{cache_key}.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def show_status():
    """Show cache status."""
    cached = load_cache_status()
    df = pd.read_parquet('data/processed/oceanembed_real_dataset.parquet')
    df['date_str'] = df['date'].astype(str).str[:10]
    
    total = len(df)
    cached_count = 0
    for _, row in df.iterrows():
        key = get_cache_key(row['latitude'], row['longitude'], row['date_str'])
        if key in cached:
            cached_count += 1
    
    print(f"Cache status: {cached_count}/{total} profiles cached ({100*cached_count/total:.1f}%)")
    print(f"Cache directory: {CACHE_DIR}")
    
    # Sample a cached file
    if cached:
        sample_key = list(cached)[0]
        sample_path = CACHE_DIR / f"{sample_key}.json"
        with open(sample_path) as f:
            sample = json.load(f)
        print(f"\nSample cached profile ({sample_key}):")
        print(f"  SSS: {sample.get('sss')}")
        print(f"  Currents: u={sample.get('current_u')}, v={sample.get('current_v')}")
        print(f"  Temp depths: {len(sample.get('temp_profile', {}))}")


def main():
    parser = argparse.ArgumentParser(description='Batch CMEMS fetch')
    parser.add_argument('--limit', type=int, default=0, help='Max profiles to fetch (0=all)')
    parser.add_argument('--status', action='store_true', help='Show cache status only')
    args = parser.parse_args()
    
    if args.status:
        show_status()
        return
    
    # Load dataset
    df = pd.read_parquet('data/processed/oceanembed_real_dataset.parquet')
    df['date_str'] = df['date'].astype(str).str[:10]
    
    cached = load_cache_status()
    print(f"Dataset: {len(df)} profiles, {len(cached)} already cached")
    
    # Find uncached profiles
    uncached = []
    for idx, row in df.iterrows():
        key = get_cache_key(row['latitude'], row['longitude'], row['date_str'])
        if key not in cached:
            uncached.append((idx, row))
    
    if args.limit > 0:
        uncached = uncached[:args.limit]
    
    print(f"Fetching {len(uncached)} profiles...")
    
    # Fetch with progress tracking
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    for i, (idx, row) in enumerate(uncached):
        lat = row['latitude']
        lon = row['longitude']
        date_str = row['date_str']
        cache_key = get_cache_key(lat, lon, date_str)
        
        elapsed = time.time() - start_time
        if i > 0:
            eta = elapsed / i * (len(uncached) - i)
            print(f"  [{i+1}/{len(uncached)}] ({lat:.2f}, {lon:.2f}) {date_str} | "
                  f"OK={success_count} fail={fail_count} | "
                  f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s", end='')
        else:
            print(f"  [{i+1}/{len(uncached)}] ({lat:.2f}, {lon:.2f}) {date_str}", end='')
        
        data = fetch_cmems_one_profile(lat, lon, date_str)
        save_cache(cache_key, data)
        
        if data.get('sss') is not None or data.get('temp_profile'):
            success_count += 1
            n_depths = len(data.get('temp_profile', {}))
            print(f" -> SSS={data.get('sss')}, depths={n_depths}")
        else:
            fail_count += 1
            print(f" -> FAILED: {data.get('error', 'unknown')[:60]}")
    
    total_time = time.time() - start_time
    total_cached = len(load_cache_status())
    
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Fetched: {success_count} ok, {fail_count} failed")
    print(f"Total cached: {total_cached}/{len(df)} ({100*total_cached/len(df):.1f}%)")
    print(f"Time: {total_time:.0f}s ({total_time/len(uncached):.1f}s per profile)")


if __name__ == '__main__':
    main()
