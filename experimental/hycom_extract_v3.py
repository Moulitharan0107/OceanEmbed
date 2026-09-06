"""
HYCOM GOFS extraction v3: Resilient, saves after each time group, handles corruption.
Uses xarray OPeNDAP to batch-fetch all points at each time step.
"""

import json, time, sys, numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
import xarray as xr
import os
import warnings
warnings.filterwarnings('ignore')

BASE_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
CACHE_DIR = Path("data/raw/surface_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LAT_START = -80.0
LAT_STEP = 0.04
LON_STEP = 0.08
N_LON = 4500
N_TIME = 16809
TIME_REF = datetime(2000, 1, 1)

def lat2i(lat): return int(round((lat - LAT_START) / LAT_STEP))
def lon2i(lon): return int(round((lon % 360) / LON_STEP)) % N_LON

def date2ti(ds):
    dt = datetime.strptime(str(ds)[:10], "%Y-%m-%d")
    h = (dt - TIME_REF).total_seconds() / 3600.0
    t0 = (datetime(2019,1,1) - TIME_REF).total_seconds() / 3600.0
    t1 = (datetime(2024,12,31) - TIME_REF).total_seconds() / 3600.0
    return max(0, min(int(round((h - t0)/(t1 - t0) * (N_TIME-1))), N_TIME-1))

def safe_save(path, data):
    """Save JSON atomically for Windows compatibility."""
    tmp_path = str(path) + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp_path, str(path))

def load_cache(fname):
    path = CACHE_DIR / fname
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except:
        # Try to recover partial data
        try:
            with open(path) as f:
                content = f.read()
            # Find last complete key-value pair
            # Try truncating at different points
            for i in range(len(content)-1, max(0, len(content)-10000), -1):
                if content[i] == '}':
                    try:
                        d = json.loads(content[:i+1])
                        return d
                    except:
                        continue
        except:
            pass
        return {}

# Load data
df = pd.read_parquet("data/processed/oceanembed_real_dataset.parquet")
print(f"Loaded {len(df)} Argo profiles")

sss_c = load_cache("hycom_sss_cache.json")
cur_c = load_cache("hycom_current_cache.json")
tmp_c = load_cache("hycom_temp_profile_cache.json")

# Find uncached
uncached = []
for i in range(len(df)):
    k = str(i)
    if (sss_c.get(k) is None or 
        not isinstance(cur_c.get(k), dict) or cur_c[k].get("u") is None or
        tmp_c.get(k) is None):
        uncached.append(i)

print(f"Cached: {len(df) - len(uncached)}, Uncached: {len(uncached)}")
if not uncached:
    print("All cached!")
    sys.exit(0)

# Group by time index
time_groups = {}
for idx in uncached:
    t_idx = date2ti(str(df.iloc[idx]["date"])[:10])
    if t_idx not in time_groups:
        time_groups[t_idx] = []
    time_groups[t_idx].append(idx)

sorted_times = sorted(time_groups.keys())
print(f"Unique time indices: {len(sorted_times)}")

# Open dataset once
print("Opening HYCOM GOFS...")
ds = xr.open_dataset(BASE_URL, engine='netcdf4', decode_times=False, mask_and_scale=True)
print(f"Ready. Processing {len(sorted_times)} time groups...")

total_done = 0
total = len(uncached)
start = time.time()

for t_idx in sorted_times:
    indices = time_groups[t_idx]
    
    lat_idx_arr = np.array([lat2i(float(df.iloc[idx]["latitude"])) for idx in indices])
    lon_idx_arr = np.array([lon2i(float(df.iloc[idx]["longitude"])) for idx in indices])
    
    try:
        sss_vals = ds['salinity'].isel(time=t_idx, depth=0, lat=lat_idx_arr, lon=lon_idx_arr).values
        u_vals = ds['water_u'].isel(time=t_idx, depth=0, lat=lat_idx_arr, lon=lon_idx_arr).values
        v_vals = ds['water_v'].isel(time=t_idx, depth=0, lat=lat_idx_arr, lon=lon_idx_arr).values
        temp_vals = ds['water_temp'].isel(time=t_idx, lat=lat_idx_arr, lon=lon_idx_arr).values
        
        for i, idx in enumerate(indices):
            k = str(idx)
            val = sss_vals[i]
            sss_c[k] = float(val) if not np.isnan(val) else None
            u_val, v_val = u_vals[i], v_vals[i]
            cur_c[k] = {"u": float(u_val) if not np.isnan(u_val) else None,
                        "v": float(v_val) if not np.isnan(v_val) else None}
            profile = temp_vals[i]
            tmp_c[k] = [float(x) if not np.isnan(x) else None for x in profile]
    except Exception as e:
        for idx in indices:
            k = str(idx)
            sss_c[k] = None
            cur_c[k] = {"u": None, "v": None}
            tmp_c[k] = None
    
    total_done += len(indices)
    
    if total_done % 200 == 0 or total_done == total:
        sv = sum(1 for v in sss_c.values() if v is not None)
        cv = sum(1 for v in cur_c.values() if isinstance(v, dict) and v.get("u") is not None)
        tv = sum(1 for v in tmp_c.values() if v is not None)
        elapsed = time.time() - start
        rate = total_done / elapsed if elapsed > 0 else 0
        eta = (total - total_done) / rate / 60 if rate > 0 else 0
        print(f"  [{total_done}/{total}] SSS={sv} Cur={cv} Temp={tv} ({rate:.1f}/s, ETA {eta:.1f}min)")
        sys.stdout.flush()
        
        safe_save(CACHE_DIR/"hycom_sss_cache.json", sss_c)
        safe_save(CACHE_DIR/"hycom_current_cache.json", cur_c)
        safe_save(CACHE_DIR/"hycom_temp_profile_cache.json", tmp_c)

ds.close()

safe_save(CACHE_DIR/"hycom_sss_cache.json", sss_c)
safe_save(CACHE_DIR/"hycom_current_cache.json", cur_c)
safe_save(CACHE_DIR/"hycom_temp_profile_cache.json", tmp_c)

sv = sum(1 for v in sss_c.values() if v is not None)
cv = sum(1 for v in cur_c.values() if isinstance(v, dict) and v.get("u") is not None)
tv = sum(1 for v in tmp_c.values() if v is not None)
print(f"\nDone! {time.time()-start:.0f}s")
print(f"SSS: {sv}/{len(df)} | Currents: {cv}/{len(df)} | TempProf: {tv}/{len(df)}")
print(f"Source: HYCOM GOFS GLBy0.08/expt_93.0 via THREDDS OPeNDAP")
