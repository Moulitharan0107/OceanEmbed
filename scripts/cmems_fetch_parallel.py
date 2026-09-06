"""
Parallel CMEMS Fetch — Stratified subset, 3 concurrent workers.
Resumable: skips already-cached profiles.
"""
import sys, io, json, time, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

DATASET_ID = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
CACHE_DIR = Path('data/raw/cmems_cache')
TARGET_DEPTHS = [10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTH_TOLERANCE = 5.0
MAX_WORKERS = 3

lock = threading.Lock()
counter = {"ok": 0, "fail": 0}


def fetch_one(lat, lon, date_str):
    import copernicusmarine, xarray as xr
    try:
        result = copernicusmarine.subset(
            dataset_id=DATASET_ID,
            variables=['so', 'uo', 'vo', 'thetao'],
            minimum_longitude=lon - 0.25, maximum_longitude=lon + 0.25,
            minimum_latitude=lat - 0.25, maximum_latitude=lat + 0.25,
            start_datetime=date_str, end_datetime=date_str,
            minimum_depth=0.4, maximum_depth=1050.0,
        )
        ds = xr.open_dataset(result.file_path)

        sss = None
        if 'so' in ds:
            v = float(ds.so.isel(depth=0).values)
            if not np.isnan(v):
                sss = round(v, 4)

        cu, cv = None, None
        if 'uo' in ds:
            v = float(ds.uo.isel(depth=0).values)
            if not np.isnan(v):
                cu = round(v, 4)
        if 'vo' in ds:
            v = float(ds.vo.isel(depth=0).values)
            if not np.isnan(v):
                cv = round(v, 4)

        temps = {}
        if 'thetao' in ds:
            depths = ds.depth.values
            for td in TARGET_DEPTHS:
                di = np.argmin(np.abs(depths - td))
                if abs(depths[di] - td) <= DEPTH_TOLERANCE:
                    v = float(ds.thetao.isel(depth=di).values)
                    if not np.isnan(v):
                        temps[str(td)] = round(v, 4)

        ds.close()
        try:
            os.remove(str(result.file_path))
        except:
            pass

        return {
            'sss': sss, 'current_u': cu, 'current_v': cv,
            'temp_profile': temps, 'source': 'cmems',
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            'sss': None, 'current_u': None, 'current_v': None,
            'temp_profile': {}, 'source': 'cmems_failed',
            'error': str(e)[:200], 'timestamp': datetime.now().isoformat(),
        }


def main():
    subset = pd.read_csv('data/raw/cmems_cache/stratified_subset.csv')
    cached = {f.stem for f in CACHE_DIR.glob('cmems_*.json')}
    uncached = subset[~subset['cache_key'].isin(cached)].copy()

    print(f"Stratified subset: {len(subset)} total, {len(subset)-len(uncached)} already cached, {len(uncached)} to fetch")
    print(f"Using {MAX_WORKERS} parallel workers")

    start_time = time.time()
    futures_map = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for _, row in uncached.iterrows():
            lat, lon = row['latitude'], row['longitude']
            date_str = row['date_str']
            cache_key = row['cache_key']
            fut = executor.submit(fetch_one, lat, lon, date_str)
            futures_map[fut] = (cache_key, lat, lon, date_str)

        done_count = 0
        for fut in as_completed(futures_map):
            cache_key, lat, lon, date_str = futures_map[fut]
            data = fut.result()
            done_count += 1

            cache_path = CACHE_DIR / f"{cache_key}.json"
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)

            with lock:
                if data.get('sss') is not None or data.get('temp_profile'):
                    counter["ok"] += 1
                else:
                    counter["fail"] += 1

            elapsed = time.time() - start_time
            rate = done_count / max(elapsed, 1) * 60
            remaining = len(uncached) - done_count
            eta_min = remaining / max(rate, 0.01)

            if done_count % 10 == 0 or done_count == len(uncached):
                total_cached = len(list(CACHE_DIR.glob('cmems_*.json')))
                pct = 100 * total_cached / len(subset)
                print(f"  [{done_count}/{len(uncached)}] OK={counter['ok']} fail={counter['fail']} | "
                      f"{rate:.1f}/min | ETA {eta_min:.0f}min | total cached: {total_cached} ({pct:.0f}%)",
                      flush=True)

    total_time = time.time() - start_time
    total_cached = len(list(CACHE_DIR.glob('cmems_*.json')))
    print(f"\n{'='*60}")
    print(f"COMPLETE: {counter['ok']} ok, {counter['fail']} failed")
    print(f"Total cached: {total_cached} files")
    print(f"Time: {total_time:.0f}s ({total_time/max(done_count,1):.1f}s per profile)")


if __name__ == '__main__':
    main()
