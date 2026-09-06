"""
HYCOM GOFS OPeNDAP batch extraction for SSS, currents, and temperature profiles.
Uses GLBy0.08/expt_93.0 (1/12° global, 40 depth levels, 2019-2024+).
All data accessed via OPeNDAP point queries — no full-grid downloads.
"""

import urllib.request
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

BASE_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
CACHE_DIR = Path("data/raw/surface_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# HYCOM depth levels (from the dataset)
HYCOM_DEPTHS = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
                125, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000]

# HYCOM coordinate parameters
LAT_START = -80.0
LAT_STEP = 0.04
LON_START = 0.0
LON_STEP = 0.08  # approximately
N_LON = 4500

# Time: "hours since 2000-01-01 00:00:00", 16809 steps, approx daily
TIME_REF = datetime(2000, 1, 1)
N_TIME = 16809

# Scale factors from DAS
SCALE = {
    "salinity": {"scale": 0.001, "offset": 20.0, "fill": -30000},
    "water_u": {"scale": 0.001, "offset": 0.0, "fill": -30000},
    "water_v": {"scale": 0.001, "offset": 0.0, "fill": -30000},
    "water_temp": {"scale": 0.001, "offset": 20.0, "fill": -30000},
}


def lat_to_idx(lat):
    return int(round((lat - LAT_START) / LAT_STEP))

def lon_to_idx(lon):
    """Convert east longitude to 0-360 index."""
    lon360 = lon % 360
    return int(round(lon360 / LON_STEP)) % N_LON

def date_to_time_idx(date_str):
    """Convert date string to nearest time index."""
    dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    hours = (dt - TIME_REF).total_seconds() / 3600.0
    t0_hours = (datetime(2019, 1, 1) - TIME_REF).total_seconds() / 3600.0
    t_end_hours = (datetime(2024, 12, 31) - TIME_REF).total_seconds() / 3600.0
    idx = int(round((hours - t0_hours) / (t_end_hours - t0_hours) * (N_TIME - 1)))
    return max(0, min(idx, N_TIME - 1))

def raw_to_phys(raw, var_name):
    """Convert raw Int16 to physical value."""
    s = SCALE[var_name]
    if raw == s["fill"]:
        return None
    return raw * s["scale"] + s["offset"]

def parse_grid_response(text, var_name):
    """Parse OPeNDAP Grid variable ASCII response to get data values."""
    in_data = False
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("Dataset") or "---" in line:
            continue
        # The data line format: [i][j][k][l], value
        if "," in line and "[" in line:
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    val = float(parts[-1].strip())
                    return val
                except ValueError:
                    continue
    return None

def parse_grid_profile(text, var_name):
    """Parse OPeNDAP Grid profile response to get all depth values."""
    values = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("Dataset") or "---" in line:
            continue
        if "," in line and "[" in line:
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    val = float(parts[-1].strip())
                    values.append(val)
                except ValueError:
                    continue
    return values

def fetch_single_point(lat, lon, date_str):
    """Fetch SSS, current_u, current_v, and temperature profile for a single point."""
    lat_idx = lat_to_idx(lat)
    lon_idx = lon_to_idx(lon)
    time_idx = date_to_time_idx(date_str)
    
    results = {}
    
    # Query 1: SSS (salinity at surface, depth=0)
    url_sss = (f"{BASE_URL}.ascii?"
               f"salinity[{time_idx}:{time_idx}][0:0][{lat_idx}:{lat_idx}][{lon_idx}:{lon_idx}]")
    
    # Query 2: Currents (surface, depth=0)
    url_u = (f"{BASE_URL}.ascii?"
             f"water_u[{time_idx}:{time_idx}][0:0][{lat_idx}:{lat_idx}][{lon_idx}:{lon_idx}]")
    url_v = (f"{BASE_URL}.ascii?"
             f"water_v[{time_idx}:{time_idx}][0:0][{lat_idx}:{lat_idx}][{lon_idx}:{lon_idx}]")
    
    # Query 3: Temperature profile (all depths)
    url_temp = (f"{BASE_URL}.ascii?"
                f"water_temp[{time_idx}:{time_idx}][0:39][{lat_idx}:{lat_idx}][{lon_idx}:{lon_idx}]")
    
    for var_name, url in [("sss", url_sss), ("current_u", url_u), 
                           ("current_v", url_v), ("temp_profile", url_temp)]:
        try:
            resp = urllib.request.urlopen(url, timeout=20)
            text = resp.read().decode("utf-8")
            
            if var_name == "temp_profile":
                vals = parse_grid_profile(text, "water_temp")
                if vals:
                    # Apply scale factors
                    s = SCALE["water_temp"]
                    phys_vals = []
                    for v in vals:
                        if v == s["fill"]:
                            phys_vals.append(None)
                        else:
                            phys_vals.append(v * s["scale"] + s["offset"])
                    results["temp_profile"] = phys_vals
                else:
                    results["temp_profile"] = None
            else:
                var_scale = {"sss": "salinity", "current_u": "water_u", "current_v": "water_v"}[var_name]
                val = parse_grid_response(text, var_scale)
                if val is not None:
                    phys = raw_to_phys(int(val), var_scale)
                    results[var_name] = phys
                else:
                    results[var_name] = None
        except Exception as e:
            results[var_name] = None
    
    return results


if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 60)
    print("HYCOM GOFS OPeNDAP Extraction")
    print("Dataset: GLBy0.08/expt_93.0")
    print("=" * 60)
    
    # Test single point first
    print("\n--- Test: Bay of Bengal (15N, 85E, 2022-06-15) ---")
    test = fetch_single_point(15.0, 85.0, "2022-06-15")
    for k, v in test.items():
        if isinstance(v, list):
            print(f"  {k}: {len(v)} values")
            print(f"    First 5: {[round(x, 3) if x is not None else None for x in v[:5]]}")
        else:
            print(f"  {k}: {v}")
    
    # Load Argo profiles
    print("\n--- Loading Argo profiles ---")
    df = pd.read_parquet("data/processed/oceanembed_real_dataset.parquet")
    print(f"Loaded {len(df)} profiles")
    
    # Cache files
    sss_cache_file = CACHE_DIR / "hycom_sss_cache.json"
    current_cache_file = CACHE_DIR / "hycom_current_cache.json"
    temp_cache_file = CACHE_DIR / "hycom_temp_profile_cache.json"
    
    # Load existing caches
    sss_cache = json.load(open(sss_cache_file)) if sss_cache_file.exists() else {}
    current_cache = json.load(open(current_cache_file)) if current_cache_file.exists() else {}
    temp_cache = json.load(open(temp_cache_file)) if temp_cache_file.exists() else {}
    
    # Find what needs extraction (combine SSS + currents + temp profiles)
    needs_extract = []
    for i in range(len(df)):
        key = str(i)
        sss_ok = key in sss_cache and sss_cache[key] is not None
        curr_ok = key in current_cache and isinstance(current_cache.get(key), dict) and current_cache[key].get("u") is not None
        temp_ok = key in temp_cache and temp_cache[key] is not None
        
        if not (sss_ok and curr_ok and temp_ok):
            needs_extract.append(i)
    
    print(f"Already cached: {len(df) - len(needs_extract)}")
    print(f"Need extraction: {len(needs_extract)}")
    
    if not needs_extract:
        print("All done!")
    else:
        # Batch extract
        batch_size = 20
        success = 0
        failed = 0
        total = len(needs_extract)
        
        for batch_start in range(0, total, batch_size):
            batch = needs_extract[batch_start:batch_start + batch_size]
            
            for idx in batch:
                row = df.iloc[idx]
                lat_val = float(row["latitude"])
                lon_val = float(row["longitude"])
                date_val = str(row["date"])[:10]
                
                result = fetch_single_point(lat_val, lon_val, date_val)
                
                # Update caches
                key = str(idx)
                if result.get("sss") is not None:
                    sss_cache[key] = result["sss"]
                    success += 1
                else:
                    sss_cache[key] = None
                
                if result.get("current_u") is not None and result.get("current_v") is not None:
                    current_cache[key] = {"u": result["current_u"], "v": result["current_v"]}
                else:
                    current_cache[key] = {"u": None, "v": None}
                
                if result.get("temp_profile") is not None:
                    temp_cache[key] = result["temp_profile"]
                else:
                    temp_cache[key] = None
                
                time.sleep(0.05)  # Rate limit
            
            # Save progress
            with open(sss_cache_file, "w") as f:
                json.dump(sss_cache, f)
            with open(current_cache_file, "w") as f:
                json.dump(current_cache, f)
            with open(temp_cache_file, "w") as f:
                json.dump(temp_cache, f)
            
            processed = min(batch_start + batch_size, total)
            sss_count = sum(1 for v in sss_cache.values() if v is not None)
            curr_count = sum(1 for v in current_cache.values() if isinstance(v, dict) and v.get("u") is not None)
            temp_count = sum(1 for v in temp_cache.values() if v is not None)
            
            print(f"  [{processed}/{total}] SSS={sss_count} | Currents={curr_count} | TempProfiles={temp_count}")
            sys.stdout.flush()
    
    # Final summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    
    sss_valid = sum(1 for v in sss_cache.values() if v is not None)
    curr_valid = sum(1 for v in current_cache.values() if isinstance(v, dict) and v.get("u") is not None)
    temp_valid = sum(1 for v in temp_cache.values() if v is not None)
    total_profiles = len(df)
    
    print(f"SSS:       {sss_valid}/{total_profiles} ({100*sss_valid/total_profiles:.1f}%)")
    print(f"Currents:  {curr_valid}/{total_profiles} ({100*curr_valid/total_profiles:.1f}%)")
    print(f"TempProf:  {temp_valid}/{total_profiles} ({100*temp_valid/total_profiles:.1f}%)")
    print(f"\nSource: HYCOM GOFS GLBy0.08/expt_93.0 via THREDDS OPeNDAP")
    print(f"SSS: sea_water_salinity (psu), scale=0.001, offset=20.0")
    print(f"Currents: eastward/northward water velocity (m/s), scale=0.001")
    print(f"Temperature: 40 depth levels from 0-5000m")
