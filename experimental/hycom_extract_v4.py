"""
HYCOM GOFS extraction v4: Split SSS+currents from temperature profiles.
Temperature profiles are large (40 values each) so save them separately
in smaller batches to avoid corruption.
"""

import json, time, sys, numpy as np, os
from pathlib import Path
from datetime import datetime
import pandas as pd
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

BASE_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
CACHE_DIR = Path("data/raw/surface_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LAT_START, LAT_STEP, LON_STEP, N_LON, N_TIME = -80.0, 0.04, 0.08, 4500, 16809
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
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    if os.path.exists(path): os.remove(path)
    os.rename(tmp, str(path))

def load_cache(fname):
    path = CACHE_DIR / fname
    if not path.exists(): return {}
    try:
        with open(path) as f: return json.load(f)
    except:
        # Truncate to last valid JSON
        with open(path, 'r') as f: content = f.read()
        for i in range(len(content)-1, max(0, len(content)-100000), -1):
            if content[i] in ('}', ']'):
                try: return json.loads(content[:i+1])
                except: continue
        return {}

# Load data
df = pd.read_parquet("data/processed/oceanembed_real_dataset.parquet")
print(f"Loaded {len(df)} profiles")

sss_c = load_cache("hycom_sss_cache.json")
cur_c = load_cache("hycom_current_cache.json")
tmp_c = load_cache("hycom_temp_profile_cache.json")

print(f"SSS cache: {sum(1 for v in sss_c.values() if v is not None)}/{len(sss_c)} valid")
print(f"Cur cache: {sum(1 for v in cur_c.values() if isinstance(v,dict) and v.get('u') is not None)}/{len(cur_c)} valid")
print(f"Tmp cache: {sum(1 for v in tmp_c.values() if v is not None)}/{len(tmp_c)} valid")

# Find what needs work
needs_sss = [i for i in range(len(df)) if sss_c.get(str(i)) is None]
needs_cur = [i for i in range(len(df)) if not isinstance(cur_c.get(str(i)), dict) or cur_c[str(i)].get("u") is None]
needs_tmp = [i for i in range(len(df)) if tmp_c.get(str(i)) is None]

print(f"\nNeeds SSS: {len(needs_sss)}, Cur: {len(needs_cur)}, Temp: {len(needs_tmp)}")
need_all = sorted(set(needs_sss) | set(needs_cur))
print(f"Combined surface need: {len(need_all)}")

# Group by time
time_groups = {}
for idx in need_all:
    t_idx = date2ti(str(df.iloc[idx]["date"])[:10])
    if t_idx not in time_groups: time_groups[t_idx] = []
    time_groups[t_idx].append(idx)

# Also group temp-only needs
tmp_groups = {}
for idx in needs_tmp:
    if idx not in need_all:  # Only if not already in surface groups
        t_idx = date2ti(str(df.iloc[idx]["date"])[:10])
        if t_idx not in tmp_groups: tmp_groups[t_idx] = []
        tmp_groups[t_idx].append(idx)

print(f"Surface time groups: {len(time_groups)}, Temp-only groups: {len(tmp_groups)}")
print(f"\nOpening HYCOM GOFS...")

ds = xr.open_dataset(BASE_URL, engine='netcdf4', decode_times=False, mask_and_scale=True)
print("Ready!\n")

# Phase 1: Extract SSS + currents
total_done = 0
start = time.time()
for t_idx in sorted(time_groups.keys()):
    indices = time_groups[t_idx]
    lat_arr = np.array([lat2i(float(df.iloc[i]["latitude"])) for i in indices])
    lon_arr = np.array([lon2i(float(df.iloc[i]["longitude"])) for i in indices])
    
    try:
        sss_v = ds['salinity'].isel(time=t_idx, depth=0, lat=lat_arr, lon=lon_arr).values
        u_v = ds['water_u'].isel(time=t_idx, depth=0, lat=lat_arr, lon=lon_arr).values
        v_v = ds['water_v'].isel(time=t_idx, depth=0, lat=lat_arr, lon=lon_arr).values
        
        for i, idx in enumerate(indices):
            k = str(idx)
            s = sss_v[i]
            sss_c[k] = float(s) if not np.isnan(s) else None
            cu, cv = u_v[i], v_v[i]
            cur_c[k] = {"u": float(cu) if not np.isnan(cu) else None,
                        "v": float(cv) if not np.isnan(cv) else None}
    except Exception as e:
        for idx in indices:
            sss_c[str(idx)] = None
            cur_c[str(idx)] = {"u": None, "v": None}
    
    total_done += len(indices)
    if total_done % 500 == 0:
        sv = sum(1 for v in sss_c.values() if v is not None)
        cv = sum(1 for v in cur_c.values() if isinstance(v,dict) and v.get("u") is not None)
        print(f"  Surface [{total_done}] SSS={sv} Cur={cv} ({total_done/(time.time()-start):.1f}/s)")
        sys.stdout.flush()
        safe_save(CACHE_DIR/"hycom_sss_cache.json", sss_c)
        safe_save(CACHE_DIR/"hycom_current_cache.json", cur_c)

sv = sum(1 for v in sss_c.values() if v is not None)
cv = sum(1 for v in cur_c.values() if isinstance(v,dict) and v.get("u") is not None)
print(f"\nSurface done: SSS={sv}, Currents={cv} ({time.time()-start:.0f}s)")
safe_save(CACHE_DIR/"hycom_sss_cache.json", sss_c)
safe_save(CACHE_DIR/"hycom_current_cache.json", cur_c)

# Phase 2: Extract temperature profiles (separate to avoid corruption)
all_tmp_needs = sorted(set(needs_tmp))
tmp_time_groups = {}
for idx in all_tmp_needs:
    t_idx = date2ti(str(df.iloc[idx]["date"])[:10])
    if t_idx not in tmp_time_groups: tmp_time_groups[t_idx] = []
    tmp_time_groups[t_idx].append(idx)

print(f"\nPhase 2: Temperature profiles ({len(all_tmp_needs)} need, {len(tmp_time_groups)} groups)")
t2_start = time.time()
t2_done = 0

for t_idx in sorted(tmp_time_groups.keys()):
    indices = tmp_time_groups[t_idx]
    lat_arr = np.array([lat2i(float(df.iloc[i]["latitude"])) for i in indices])
    lon_arr = np.array([lon2i(float(df.iloc[i]["longitude"])) for i in indices])
    
    try:
        temp_v = ds['water_temp'].isel(time=t_idx, lat=lat_arr, lon=lon_arr).values
        for i, idx in enumerate(indices):
            k = str(idx)
            profile = temp_v[i]
            tmp_c[k] = [float(x) if not np.isnan(x) else None for x in profile]
    except:
        for idx in indices:
            tmp_c[str(idx)] = None
    
    t2_done += len(indices)
    if t2_done % 200 == 0:
        tv = sum(1 for v in tmp_c.values() if v is not None)
        print(f"  Temp [{t2_done}/{len(all_tmp_needs)}] valid={tv}")
        sys.stdout.flush()
        safe_save(CACHE_DIR/"hycom_temp_profile_cache.json", tmp_c)

ds.close()
safe_save(CACHE_DIR/"hycom_temp_profile_cache.json", tmp_c)

tv = sum(1 for v in tmp_c.values() if v is not None)
print(f"\nTemp done: {tv}/{len(df)} ({time.time()-t2_start:.0f}s)")

print(f"\n{'='*60}")
print(f"FINAL: SSS={sv}/{len(df)} ({100*sv/len(df):.1f}%)")
print(f"       Cur={cv}/{len(df)} ({100*cv/len(df):.1f}%)") 
print(f"       Tmp={tv}/{len(df)} ({100*tv/len(df):.1f}%)")
print(f"Total: {time.time()-start:.0f}s")
print(f"Source: HYCOM GOFS GLBy0.08/expt_93.0")
