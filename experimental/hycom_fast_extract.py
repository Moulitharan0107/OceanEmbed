"""
Fast HYCOM GOFS extraction: batch all profiles with the same time index together.
Instead of 4 requests per profile, batch into ~744 time slices.
"""

import urllib.request
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
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

SCALE = {
    "salinity": {"scale": 0.001, "offset": 20.0, "fill": -30000},
    "water_u": {"scale": 0.001, "offset": 0.0, "fill": -30000},
    "water_v": {"scale": 0.001, "offset": 0.0, "fill": -30000},
    "water_temp": {"scale": 0.001, "offset": 20.0, "fill": -30000},
}

def lat_to_idx(lat):
    return int(round((lat - LAT_START) / LAT_STEP))

def lon_to_idx(lon):
    return int(round((lon % 360) / LON_STEP)) % N_LON

def date_to_time_idx(date_str):
    dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    hours = (dt - TIME_REF).total_seconds() / 3600.0
    t0 = (datetime(2019, 1, 1) - TIME_REF).total_seconds() / 3600.0
    t_end = (datetime(2024, 12, 31) - TIME_REF).total_seconds() / 3600.0
    idx = int(round((hours - t0) / (t_end - t0) * (N_TIME - 1)))
    return max(0, min(idx, N_TIME - 1))

def raw_to_phys(val, var_name):
    s = SCALE[var_name]
    if val == s["fill"]:
        return None
    return val * s["scale"] + s["offset"]

def fetch_point_value(url):
    """Fetch a single value from an OPeNDAP Grid query."""
    try:
        resp = urllib.request.urlopen(url, timeout=15)
        text = resp.read().decode("utf-8")
        for line in text.split("\n"):
            line = line.strip()
            if "," in line and "[" in line:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        return float(parts[-1].strip())
                    except ValueError:
                        continue
    except Exception:
        pass
    return None

def fetch_profile(url):
    """Fetch a temperature profile from OPeNDAP."""
    try:
        resp = urllib.request.urlopen(url, timeout=20)
        text = resp.read().decode("utf-8")
        values = []
        for line in text.split("\n"):
            line = line.strip()
            if "," in line and "[" in line:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        values.append(float(parts[-1].strip()))
                    except ValueError:
                        continue
        return values
    except Exception:
        return []


def extract_batch(profiles, time_idx, var_name, depth_slice=None):
    """Extract a variable for multiple profiles at the same time index.
    Returns dict mapping profile_index -> value."""
    results = {}
    
    for idx, (lat, lon) in profiles:
        lat_i = lat_to_idx(lat)
        lon_i = lon_to_idx(lon)
        
        if depth_slice is not None:
            url = (f"{BASE_URL}.ascii?"
                   f"{var_name}[{time_idx}:{time_idx}][{depth_slice}][{lat_i}:{lat_i}][{lon_i}:{lon_i}]")
            vals = fetch_profile(url)
            if vals:
                s = SCALE[var_name]
                phys = []
                for v in vals:
                    if v == s["fill"]:
                        phys.append(None)
                    else:
                        phys.append(v * s["scale"] + s["offset"])
                results[idx] = phys
            else:
                results[idx] = None
        else:
            url = (f"{BASE_URL}.ascii?"
                   f"{var_name}[{time_idx}:{time_idx}][0:0][{lat_i}:{lat_i}][{lon_i}:{lon_i}]")
            val = fetch_point_value(url)
            if val is not None:
                results[idx] = raw_to_phys(int(val), var_name)
            else:
                results[idx] = None
        
        time.sleep(0.02)
    
    return results


if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 60)
    print("HYCOM GOFS Fast Batch Extraction")
    print("=" * 60)
    
    # Load Argo profiles
    df = pd.read_parquet("data/processed/oceanembed_real_dataset.parquet")
    print(f"Loaded {len(df)} Argo profiles")
    
    # Load existing caches
    sss_file = CACHE_DIR / "hycom_sss_cache.json"
    curr_file = CACHE_DIR / "hycom_current_cache.json"
    temp_file = CACHE_DIR / "hycom_temp_profile_cache.json"
    
    sss_cache = json.load(open(sss_file)) if sss_file.exists() else {}
    curr_cache = json.load(open(curr_file)) if curr_file.exists() else {}
    temp_cache = json.load(open(temp_file)) if temp_file.exists() else {}
    
    # Find uncached profiles
    uncached = []
    for i in range(len(df)):
        key = str(i)
        if (sss_cache.get(key) is None or 
            not isinstance(curr_cache.get(key), dict) or curr_cache[key].get("u") is None or
            temp_cache.get(key) is None):
            uncached.append(i)
    
    print(f"Cached: {len(df) - len(uncached)}, Uncached: {len(uncached)}")
    
    if not uncached:
        print("All cached!")
        sys.exit(0)
    
    # Group uncached profiles by time index
    time_groups = {}
    for idx in uncached:
        row = df.iloc[idx]
        t_idx = date_to_time_idx(str(row["date"])[:10])
        if t_idx not in time_groups:
            time_groups[t_idx] = []
        time_groups[t_idx].append(idx)
    
    print(f"Unique time indices: {len(time_groups)}")
    
    # Process each time group
    total_done = 0
    total = len(uncached)
    start_time = time.time()
    
    for t_idx, profile_indices in sorted(time_groups.items()):
        # Get lat/lon for each profile
        profiles = []
        for idx in profile_indices:
            row = df.iloc[idx]
            profiles.append((idx, (float(row["latitude"]), float(row["longitude"]))))
        
        # Extract SSS (one request per profile)
        sss_results = extract_batch(profiles, t_idx, "salinity", depth_slice="0:0")
        
        # Extract current_u (one request per profile)
        u_results = extract_batch(profiles, t_idx, "water_u", depth_slice="0:0")
        
        # Extract current_v (one request per profile)
        v_results = extract_batch(profiles, t_idx, "water_v", depth_slice="0:0")
        
        # Extract temperature profile (one request per profile, all depths)
        temp_results = extract_batch(profiles, t_idx, "water_temp", depth_slice="0:39")
        
        # Save results
        for idx in profile_indices:
            key = str(idx)
            sss_cache[key] = sss_results.get(idx)
            curr_cache[key] = {"u": u_results.get(idx), "v": v_results.get(idx)}
            temp_cache[key] = temp_results.get(idx)
        
        total_done += len(profile_indices)
        elapsed = time.time() - start_time
        rate = total_done / elapsed if elapsed > 0 else 0
        remaining = (total - total_done) / rate if rate > 0 else 0
        
        if total_done % 100 == 0 or total_done == total:
            sss_valid = sum(1 for v in sss_cache.values() if v is not None)
            curr_valid = sum(1 for v in curr_cache.values() if isinstance(v, dict) and v.get("u") is not None)
            temp_valid = sum(1 for v in temp_cache.values() if v is not None)
            
            print(f"  [{total_done}/{total}] SSS={sss_valid} Cur={curr_valid} Temp={temp_valid} "
                  f"({rate:.1f}/s, ETA {remaining/60:.1f}min)")
            sys.stdout.flush()
            
            # Save progress
            with open(sss_file, "w") as f:
                json.dump(sss_cache, f)
            with open(curr_file, "w") as f:
                json.dump(curr_cache, f)
            with open(temp_file, "w") as f:
                json.dump(temp_cache, f)
    
    # Final save
    with open(sss_file, "w") as f:
        json.dump(sss_cache, f)
    with open(curr_file, "w") as f:
        json.dump(curr_cache, f)
    with open(temp_file, "w") as f:
        json.dump(temp_cache, f)
    
    # Summary
    print("\n" + "=" * 60)
    sss_valid = sum(1 for v in sss_cache.values() if v is not None)
    curr_valid = sum(1 for v in curr_cache.values() if isinstance(v, dict) and v.get("u") is not None)
    temp_valid = sum(1 for v in temp_cache.values() if v is not None)
    
    print(f"SSS:       {sss_valid}/{len(df)} ({100*sss_valid/len(df):.1f}%)")
    print(f"Currents:  {curr_valid}/{len(df)} ({100*curr_valid/len(df):.1f}%)")
    print(f"TempProf:  {temp_valid}/{len(df)} ({100*temp_valid/len(df):.1f}%)")
    print(f"Time: {time.time()-start_time:.0f}s")
    print(f"Source: HYCOM GOFS GLBy0.08/expt_93.0 (THREDDS OPeNDAP)")
