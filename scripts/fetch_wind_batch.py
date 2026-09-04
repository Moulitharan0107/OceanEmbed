#!/usr/bin/env python3
"""Fetch wind data from Open-Meteo ERA5 API in batches. Properly caches to disk."""
import warnings; warnings.filterwarnings('ignore')
import requests, json, time, numpy as np, pandas as pd
from pathlib import Path

CACHE = Path('data/raw/surface_cache/wind_values.json')
profiles = pd.read_parquet('data/raw/argo/argo_profiles_qc.parquet')
profiles['date'] = pd.to_datetime(profiles['time']).dt.tz_localize(None).dt.normalize()

# Load or init cache
if CACHE.exists():
    wind = json.load(open(CACHE))
    # Normalize keys to str
    wind['u10'] = {str(k): v for k, v in wind['u10'].items()}
    wind['v10'] = {str(k): v for k, v in wind['v10'].items()}
else:
    wind = {'u10': {}, 'v10': {}}

need = [str(i) for i in profiles.index if str(i) not in wind['u10']]
print(f'Cached: {len(wind["u10"])}, Need: {len(need)}')

t0 = time.time()
fetched = 0
for i, key in enumerate(need):
    idx = int(key)
    row = profiles.loc[idx]
    url = f'https://archive-api.open-meteo.com/v1/archive?latitude={row["latitude"]:.4f}&longitude={row["longitude"]:.4f}&start_date={row["date"].strftime("%Y-%m-%d")}&end_date={row["date"].strftime("%Y-%m-%d")}&hourly=wind_speed_10m,wind_direction_10m&timezone=UTC'
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            d = r.json().get('hourly', {})
            ws = d.get('wind_speed_10m', [])
            wd = d.get('wind_direction_10m', [])
            w12 = ws[12] if len(ws) > 12 else (ws[0] if ws else None)
            d12 = wd[12] if len(wd) > 12 else (wd[0] if wd else None)
            if w12 is not None and d12 is not None:
                rad = np.radians(d12)
                wind['u10'][key] = round(-w12 * np.sin(rad), 4)
                wind['v10'][key] = round(-w12 * np.cos(rad), 4)
            else:
                wind['u10'][key] = None; wind['v10'][key] = None
        else:
            wind['u10'][key] = None; wind['v10'][key] = None
    except:
        wind['u10'][key] = None; wind['v10'][key] = None
    fetched += 1

    if fetched % 50 == 0:
        n = sum(1 for v in wind['u10'].values() if v is not None)
        print(f'  {fetched}/{len(need)}, {n} valid, {time.time()-t0:.0f}s')
        with open(CACHE, 'w') as f:
            json.dump(wind, f)

    if time.time() - t0 > 540:
        print(f'  Time limit at {fetched}')
        break

with open(CACHE, 'w') as f:
    json.dump(wind, f)

n = sum(1 for v in wind['u10'].values() if v is not None)
print(f'FINAL: {n} valid u10 out of {len(wind["u10"])} total')
