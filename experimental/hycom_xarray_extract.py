"""
Efficient HYCOM GOFS extraction using xarray OPeNDAP.
Groups profiles by time index, loads one time slice at a time.
"""

import json, time, sys, numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
import xarray as xr
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

# Load Argo profiles
df = pd.read_parquet("data/processed/oceanembed_real_dataset.parquet")
print(f"Loaded {len(df)} Argo profiles")

# Load existing caches
sss_c = json.load(open(CACHE_DIR/"hycom_sss_cache.json")) if (CACHE_DIR/"hycom_sss_cache.json").exists() else {}
cur_c = json.load(open(CACHE_DIR/"hycom_current_cache.json")) if (CACHE_DIR/"hycom_current_cache.json").exists() else {}
tmp_c = json.load(open(CACHE_DIR/"hycom_temp_profile_cache.json")) if (CACHE_DIR/"hycom_temp_profile_cache.json").exists() else {}

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

print(f"Unique time indices: {len(time_groups)}")
print(f"Opening HYCOM dataset via OPeNDAP...")

ds = xr.open_dataset(BASE_URL, engine='netcdf4', decode_times=False, 
                     mask_and_scale=True)
print(f"Dataset loaded. Variables: {list(ds.data_vars)[:8]}")

# Process each time group
total_done = 0
total = len(uncached)
start = time.time()

for t_idx in sorted(time_groups.keys()):
    indices = time_groups[t_idx]
    
    # Collect all (lat_idx, lon_idx) pairs for this time
    lat_idx_list = []
    lon_idx_list = []
    for idx in indices:
        row = df.iloc[idx]
        lat_idx_list.append(lat2i(float(row["latitude"])))
        lon_idx_list.append(lon2i(float(row["longitude"])))
    
    lat_idx_arr = np.array(lat_idx_list)
    lon_idx_arr = np.array(lon_idx_list)
    
    try:
        # Extract SSS for all points at this time
        sss_vals = ds['salinity'].isel(time=t_idx, depth=0, lat=lat_idx_arr, lon=lon_idx_arr).values
        
        # Extract current U for all points at this time
        u_vals = ds['water_u'].isel(time=t_idx, depth=0, lat=lat_idx_arr, lon=lon_idx_arr).values
        
        # Extract current V for all points at this time
        v_vals = ds['water_v'].isel(time=t_idx, depth=0, lat=lat_idx_arr, lon=lon_idx_arr).values
        
        # Extract temperature profiles for all points at this time
        temp_vals = ds['water_temp'].isel(time=t_idx, lat=lat_idx_arr, lon=lon_idx_arr).values
        # temp_vals shape: (n_profiles, n_depths)
        
        for i, idx in enumerate(indices):
            k = str(idx)
            
            # SSS
            val = sss_vals[i]
            sss_c[k] = float(val) if not np.isnan(val) else None
            
            # Currents
            u_val = u_vals[i]
            v_val = v_vals[i]
            cur_c[k] = {
                "u": float(u_val) if not np.isnan(u_val) else None,
                "v": float(v_val) if not np.isnan(v_val) else None
            }
            
            # Temperature profile
            profile = temp_vals[i]
            valid_profile = [float(x) if not np.isnan(x) else None for x in profile]
            tmp_c[k] = valid_profile if any(x is not None for x in valid_profile) else None
        
        total_done += len(indices)
        
    except Exception as e:
        print(f"  Error at time {t_idx}: {e}")
        for idx in indices:
            k = str(idx)
            sss_c[k] = None
            cur_c[k] = {"u": None, "v": None}
            tmp_c[k] = None
        total_done += len(indices)
        
        # Save progress periodically
        if total_done % 200 == 0 or total_done == total:
            sv = sum(1 for v in sss_c.values() if v is not None)
            cv = sum(1 for v in cur_c.values() if isinstance(v,dict) and v.get("u") is not None)
            tv = sum(1 for v in tmp_c.values() if v is not None)
            elapsed = time.time() - start
            rate = total_done / elapsed if elapsed > 0 else 0
            eta = (total - total_done) / rate / 60 if rate > 0 else 0
            
            print(f"  [{total_done}/{total}] SSS={sv} Cur={cv} Temp={tv} "
                  f"({rate:.1f}/s, ETA {eta:.1f}min)")
            sys.stdout.flush()
            
            for fname, data in [("hycom_sss_cache.json", sss_c), ("hycom_current_cache.json", cur_c), ("hycom_temp_profile_cache.json", tmp_c)]:
                tmp = CACHE_DIR / f"_{fname}.tmp"
                with open(tmp, "w") as f: json.dump(data, f)
                tmp.rename(CACHE_DIR / fname)

ds.close()

# Final save
with open(CACHE_DIR/"hycom_sss_cache.json","w") as f: json.dump(sss_c, f)
with open(CACHE_DIR/"hycom_current_cache.json","w") as f: json.dump(cur_c, f)
with open(CACHE_DIR/"hycom_temp_profile_cache.json","w") as f: json.dump(tmp_c, f)

print(f"\nDone! Total time: {time.time()-start:.0f}s")
sv = sum(1 for v in sss_c.values() if v is not None)
cv = sum(1 for v in cur_c.values() if isinstance(v,dict) and v.get("u") is not None)
tv = sum(1 for v in tmp_c.values() if v is not None)
print(f"SSS: {sv}/{len(df)} | Currents: {cv}/{len(df)} | TempProfiles: {tv}/{len(df)}")
print(f"Source: HYCOM GOFS GLBy0.08/expt_93.0")
