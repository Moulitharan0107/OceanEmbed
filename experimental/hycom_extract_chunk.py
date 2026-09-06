"""Extract a chunk of HYCOM profiles. Resumes from cache."""

import json, time, sys, urllib.request, numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
CACHE_DIR = Path("data/raw/surface_cache")
LAT_START, LAT_STEP, LON_STEP, N_LON, N_TIME = -80.0, 0.04, 0.08, 4500, 16809
TIME_REF = datetime(2000, 1, 1)
SCALE = {
    "salinity": (0.001, 20.0, -30000),
    "water_u": (0.001, 0.0, -30000),
    "water_v": (0.001, 0.0, -30000),
    "water_temp": (0.001, 20.0, -30000),
}

def lat2i(lat): return int(round((lat - LAT_START) / LAT_STEP))
def lon2i(lon): return int(round((lon % 360) / LON_STEP)) % N_LON
def date2ti(ds):
    dt = datetime.strptime(str(ds)[:10], "%Y-%m-%d")
    h = (dt - TIME_REF).total_seconds() / 3600.0
    t0 = (datetime(2019,1,1) - TIME_REF).total_seconds() / 3600.0
    t1 = (datetime(2024,12,31) - TIME_REF).total_seconds() / 3600.0
    return max(0, min(int(round((h - t0)/(t1 - t0) * (N_TIME-1))), N_TIME-1))

def fetch_val(url):
    try:
        r = urllib.request.urlopen(url, timeout=15)
        for line in r.read().decode("utf-8").split("\n"):
            l = line.strip()
            if "," in l and "[" in l:
                p = l.split(",")
                if len(p) >= 2:
                    return float(p[-1].strip())
    except: pass
    return None

def fetch_vals(url):
    try:
        r = urllib.request.urlopen(url, timeout=20)
        vals = []
        for line in r.read().decode("utf-8").split("\n"):
            l = line.strip()
            if "," in l and "[" in l:
                p = l.split(",")
                if len(p) >= 2:
                    vals.append(float(p[-1].strip()))
        return vals
    except: pass
    return []

def to_phys(raw, var):
    sc, off, fill = SCALE[var]
    if raw == fill: return None
    return raw * sc + off

def to_phys_list(raws, var):
    sc, off, fill = SCALE[var]
    return [None if v == fill else v * sc + off for v in raws]

# Load data
df = pd.read_parquet("data/processed/oceanembed_real_dataset.parquet")
sss_c = json.load(open(CACHE_DIR/"hycom_sss_cache.json")) if (CACHE_DIR/"hycom_sss_cache.json").exists() else {}
cur_c = json.load(open(CACHE_DIR/"hycom_current_cache.json")) if (CACHE_DIR/"hycom_current_cache.json").exists() else {}
tmp_c = json.load(open(CACHE_DIR/"hycom_temp_profile_cache.json")) if (CACHE_DIR/"hycom_temp_profile_cache.json").exists() else {}

# Find uncached
MAX_PROFILES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
uncached = []
for i in range(len(df)):
    k = str(i)
    if (sss_c.get(k) is None or 
        not isinstance(cur_c.get(k), dict) or cur_c[k].get("u") is None or
        tmp_c.get(k) is None):
        uncached.append(i)
uncached = uncached[:MAX_PROFILES]

print(f"Extracting {len(uncached)} profiles (cache: SSS={len(sss_c)}, Cur={len(cur_c)}, Temp={len(tmp_c)})")
t0 = time.time()

for n, i in enumerate(uncached):
    row = df.iloc[i]
    lat, lon, ds = float(row["latitude"]), float(row["longitude"]), str(row["date"])[:10]
    ti = date2ti(ds)
    li, oi = lat2i(lat), lon2i(lon)
    k = str(i)
    
    # SSS
    u = f"{BASE_URL}.ascii?salinity[{ti}:{ti}][0:0][{li}:{li}][{oi}:{oi}]"
    v = fetch_val(u)
    sss_c[k] = to_phys(int(v), "salinity") if v is not None else None
    
    # Current U
    u2 = f"{BASE_URL}.ascii?water_u[{ti}:{ti}][0:0][{li}:{li}][{oi}:{oi}]"
    v2 = fetch_val(u2)
    cu = to_phys(int(v2), "water_u") if v2 is not None else None
    
    # Current V
    u3 = f"{BASE_URL}.ascii?water_v[{ti}:{ti}][0:0][{li}:{li}][{oi}:{oi}]"
    v3 = fetch_val(u3)
    cv = to_phys(int(v3), "water_v") if v3 is not None else None
    cur_c[k] = {"u": cu, "v": cv}
    
    # Temp profile
    u4 = f"{BASE_URL}.ascii?water_temp[{ti}:{ti}][0:39][{li}:{li}][{oi}:{oi}]"
    raws = fetch_vals(u4)
    tmp_c[k] = to_phys_list(raws, "water_temp") if raws else None
    
    time.sleep(0.03)
    
    if (n+1) % 50 == 0 or n+1 == len(uncached):
        sv = sum(1 for v in sss_c.values() if v is not None)
        cv2 = sum(1 for v in cur_c.values() if isinstance(v,dict) and v.get("u") is not None)
        tv = sum(1 for v in tmp_c.values() if v is not None)
        el = time.time()-t0
        print(f"  [{n+1}/{len(uncached)}] SSS={sv} Cur={cv2} Temp={tv} ({(n+1)/el:.1f}/s)")
        with open(CACHE_DIR/"hycom_sss_cache.json","w") as f: json.dump(sss_c,f)
        with open(CACHE_DIR/"hycom_current_cache.json","w") as f: json.dump(cur_c,f)
        with open(CACHE_DIR/"hycom_temp_profile_cache.json","w") as f: json.dump(tmp_c,f)

print(f"Done in {time.time()-t0:.0f}s")
