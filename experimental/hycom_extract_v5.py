"""
HYCOM GOFS extraction v5: corruption-proof per-profile caching.

Each profile is saved as its own small JSON file (atomic write: tmp + rename),
so a timeout kill can never corrupt the whole cache. Profiles sharing the same
HYCOM time index are fetched in one OPeNDAP request. A modest thread pool
speeds up the many small requests. Fully resumable.

Source: HYCOM GOFS 3.1 reanalysis, GLBy0.08/expt_93.0 (free, no login).
Variables: water_temp (3D), water_u (surface), water_v (surface), salinity (surface).
"""

import json, os, sys, time, threading, random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

BASE_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
CACHE_DIR = Path("data/raw/surface_cache/hycom_v5")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LAT_START, LAT_STEP, LON_STEP, N_LON, N_TIME = -80.0, 0.04, 0.08, 4500, 16809
TIME_REF = datetime(2000, 1, 1)

def lat2i(lat): return int(round((lat - LAT_START) / LAT_STEP))
def lon2i(lon): return int(round((lon % 360) / LON_STEP)) % N_LON

def date2ti(d):
    dt = d.to_pydatetime() if hasattr(d, 'to_pydatetime') else datetime.strptime(str(d)[:10], "%Y-%m-%d")
    h = (dt - TIME_REF).total_seconds() / 3600.0
    t0 = (datetime(2019, 1, 1) - TIME_REF).total_seconds() / 3600.0
    t1 = (datetime(2024, 12, 31) - TIME_REF).total_seconds() / 3600.0
    return max(0, min(int(round((h - t0) / (t1 - t0) * (N_TIME - 1))), N_TIME - 1))

def profile_path(idx):
    return CACHE_DIR / f"profile_{idx:05d}.json"

def atomic_save(idx, data):
    path = profile_path(idx)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f)
    if path.exists():
        path.unlink()
    os.rename(str(tmp), str(path))

def load_done():
    done = {}
    for p in CACHE_DIR.glob("profile_*.json"):
        try:
            with open(p) as f:
                d = json.load(f)
            idx = int(p.stem.split('_')[1])
            done[idx] = d
        except Exception:
            pass  # skip corrupt/incomplete per-profile file; it will be re-fetched
    return done

# ---------------- load dataframe ----------------
df = pd.read_parquet("data/processed/oceanembed_real_dataset.parquet")
print(f"Loaded {len(df)} profiles", flush=True)

done = load_done()
print(f"Already cached: {len(done)} profiles", flush=True)

# Recover earlier SSS entries from the old cache (bonus continuity)
old_sss_path = Path("data/raw/surface_cache/hycom_sss_cache.json")
if old_sss_path.exists():
    try:
        with open(old_sss_path) as f:
            content = f.read()
        for i in range(len(content) - 1, max(0, len(content) - 200000), -1):
            if content[i] in ('}', ']'):
                try:
                    old = json.loads(content[:i + 1])
                    break
                except Exception:
                    continue
        else:
            old = {}
        merged = 0
        for k, v in old.items():
            idx = int(k)
            if idx not in done and v is not None:
                done[idx] = {"sss": v, "u": None, "v": None, "temp": None, "source": "recovered-old-cache"}
                merged += 1
        print(f"Recovered {merged} profiles from old SSS cache", flush=True)
    except Exception as e:
        print(f"Old cache recovery skipped: {e}", flush=True)

# Which indices still need full extraction (need u/v/temp, not just sss)
todo = []
for i in range(len(df)):
    d = done.get(i)
    needs = (d is None or d.get("temp") is None or d.get("u") is None or d.get("v") is None)
    if needs:
        todo.append(i)
print(f"Need extraction: {len(todo)} profiles", flush=True)

if not todo:
    print("All profiles already extracted. Nothing to do.", flush=True)
    sys.exit(0)

# Group by time index to batch requests
time_groups = {}
for idx in todo:
    ti = date2ti(df.iloc[idx]['date'])
    time_groups.setdefault(ti, []).append(idx)
print(f"Unique time indices to fetch: {len(time_groups)}", flush=True)

# ---------------- extraction ----------------
NCONC = int(os.environ.get('HYCOM_CONC', '2'))
print(f"Using concurrency {NCONC}", flush=True)

print("Opening datasets per worker...", flush=True)

lock = threading.Lock()
_tls = threading.local()

def get_ds():
    if not hasattr(_tls, 'ds'):
        _tls.ds = xr.open_dataset(BASE_URL, engine='netcdf4', decode_times=False, mask_and_scale=True)
    return _tls.ds

# Pre-compute keep-depth indices (only depths <= 1000m)
_ds0 = xr.open_dataset(BASE_URL, engine='netcdf4', decode_times=False, mask_and_scale=True)
_dep_vals = _ds0['depth'].values.astype(float)
_KEEP_D = [0.0,2.0,4.0,6.0,8.0,10.0,12.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,
           60.0,70.0,80.0,90.0,100.0,125.0,150.0,200.0,250.0,300.0,350.0,400.0,
           500.0,600.0,700.0,800.0,900.0,1000.0]
_KEEP_IDX = [int(np.argmin(np.abs(_dep_vals - d))) for d in _KEEP_D]
_ds0.close()
print(f"Depth trim: {len(_KEEP_D)} depths (was 40) — indices {min(_KEEP_IDX)}..{max(_KEEP_IDX)}", flush=True)

# warm up one dataset
get_ds()
print("Dataset open. Extracting...", flush=True)
counters = {"ok": 0, "fail": 0}
start_t = time.time()

def fetch_group(args):
    t_idx, indices = args
    lat_arr = np.array([lat2i(float(df.iloc[i]['latitude'])) for i in indices])
    lon_arr = np.array([lon2i(float(df.iloc[i]['longitude'])) for i in indices])
    # polite pacing: avoid hammering the server
    time.sleep(random.uniform(0.05, 0.25))
    sss_v = u_v = v_v = t_raw = None
    for attempt in range(3):
        try:
            ds = get_ds()
            z = np.arange(len(indices))
            li = xr.DataArray(lat_arr, dims='z')
            oi = xr.DataArray(lon_arr, dims='z')
            surf = ds[['salinity','water_u','water_v']].isel(time=t_idx, depth=0).isel(lat=li, lon=oi).load()
            sss_v = surf['salinity'].values
            u_v = surf['water_u'].values
            v_v = surf['water_v'].values
            t_raw = ds['water_temp'].isel(time=t_idx, depth=_KEEP_IDX).isel(lat=li, lon=oi).load().values
            break
        except Exception as e:
            if attempt == 2:
                return None, (t_idx, f"{type(e).__name__}: {e}")
            time.sleep(1.0 + attempt * 1.5)
    if sss_v is None:
        return None, (t_idx, "failed after retries")
    out = []
    for j, idx in enumerate(indices):
        s = sss_v[j]
        rec = {
            "sss": float(s) if not np.isnan(s) else None,
            "u": float(u_v[j]) if not np.isnan(u_v[j]) else None,
            "v": float(v_v[j]) if not np.isnan(v_v[j]) else None,
            "temp": [float(x) if not np.isnan(x) else None for x in t_raw[:, j]],
            "source": "HYCOM GOFS GLBy0.08/expt_93.0",
        }
        out.append((idx, rec))
    return out, None

with ThreadPoolExecutor(max_workers=NCONC) as ex:
    futs = {ex.submit(fetch_group, (t, idxs)): t for t, idxs in time_groups.items()}
    for fut in as_completed(futs):
        res, err = fut.result()
        if err:
            t_idx, msg = err
            with lock:
                counters['fail'] += 1
                if counters['fail'] <= 5:
                    print(f"  group t={t_idx} FAIL: {msg}", flush=True)
            continue
        for idx, rec in res:
            try:
                atomic_save(idx, rec)
                with lock:
                    counters['ok'] += 1
            except Exception as e:
                with lock:
                    counters['fail'] += 1
        with lock:
            done_n = counters['ok']
            if done_n % 100 == 0:
                rate = done_n / max(time.time() - start_t, 1e-9)
                print(f"  [{done_n} cached, {rate:.1f}/s, {time.time()-start_t:.0f}s elapsed]", flush=True)

elapsed = time.time() - start_t
done = load_done()
n_sss = sum(1 for d in done.values() if d.get('sss') is not None)
n_uv = sum(1 for d in done.values() if d.get('u') is not None and d.get('v') is not None)
n_temp = sum(1 for d in done.values() if d.get('temp') is not None)
print(f"\n{'='*60}", flush=True)
print(f"Profiles cached: {len(done)}/{len(df)}", flush=True)
print(f"SSS valid: {n_sss} | Currents valid: {n_uv} | Temp profiles valid: {n_temp}", flush=True)
print(f"Elapsed: {elapsed:.0f}s", flush=True)