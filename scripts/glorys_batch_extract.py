"""GLORYS batch extraction by date.
For each unique date, fetch ONE full-field subset covering the Indian Ocean domain,
then extract temperature at all Argo profile locations for that date locally with xarray.

HARD CAP: 30 minutes total."""
import time
import json
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import copernicusmarine

# Configuration
CACHE_DIR = Path('data/raw/surface_cache/glorys')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / 'glorys_profiles.json'
DATASET_ID = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
TIMEOUT = 30 * 60  # 30 minutes

# Domain
LAT_MIN, LAT_MAX = -30, 30
LON_MIN, LON_MAX = 30, 120
TARGET_DEPTHS = [10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# Load existing cache
cache = {}
if CACHE_FILE.exists():
    cache = json.load(open(CACHE_FILE))
print(f'Existing cache: {len(cache)} profiles')

# Load Argo dataset
df = pd.read_parquet('data/processed/oceanembed_real_dataset.parquet')
df['date_str'] = df['date'].astype(str).str[:10]

# Build date->profile_indices mapping
date_profiles = {}
for idx, row in df.iterrows():
    d = row['date_str']
    if d not in date_profiles:
        date_profiles[d] = []
    date_profiles[d].append(idx)

# Sort dates by profile count (most first)
sorted_dates = sorted(date_profiles.keys(), key=lambda d: -len(date_profiles[d]))
print(f'Unique dates: {len(sorted_dates)}')

# Check which dates are already cached
uncached_dates = []
for d in sorted_dates:
    profiles = date_profiles[d]
    all_cached = all(str(p) in cache for p in profiles)
    if not all_cached:
        uncached_dates.append(d)

print(f'Dates needing extraction: {len(uncached_dates)}')
print(f'Already cached: {len(sorted_dates) - len(uncached_dates)}')

if not uncached_dates:
    print('All dates already cached!')
else:
    print(f'\nStarting extraction (30 min cap)...')
    print(f'First date: {uncached_dates[0]} ({len(date_profiles[uncached_dates[0]])} profiles)')
    
    start_time = time.time()
    dates_processed = 0
    profiles_matched = 0
    
    for date_str in uncached_dates:
        elapsed = time.time() - start_time
        if elapsed > TIMEOUT:
            print(f'\nTIME CAP HIT at {elapsed:.0f}s')
            break
        
        profiles = date_profiles[date_str]
        print(f'\n[{elapsed:.0f}s] Date: {date_str} ({len(profiles)} profiles)...')
        
        try:
            # Fetch GLORYS field for this date
            t0 = time.time()
            result = copernicusmarine.subset(
                dataset_id=DATASET_ID,
                variables=['thetao'],
                minimum_longitude=LON_MIN,
                maximum_longitude=LON_MAX,
                minimum_latitude=LAT_MIN,
                maximum_latitude=LAT_MAX,
                start_datetime=date_str,
                end_datetime=date_str,
                minimum_depth=min(TARGET_DEPTHS),
                maximum_depth=max(TARGET_DEPTHS),
            )
            fetch_time = time.time() - t0
            print(f'  Fetched in {fetch_time:.1f}s')
            
            # Open with xarray
            ds = xr.open_dataset(result.file_path)
            
            # Extract profiles for this date
            matched = 0
            for pidx in profiles:
                row = df.iloc[pidx]
                lat, lon = row['latitude'], row['longitude']
                
                # Find nearest grid points
                try:
                    # Select nearest point
                    subset = ds['thetao'].sel(
                        latitude=lat, longitude=lon,
                        method='nearest'
                    )
                    
                    # Extract at target depths
                    temp_values = []
                    valid = True
                    for depth in TARGET_DEPTHS:
                        try:
                            val = float(subset.sel(depth=depth, method='nearest').values)
                            if np.isnan(val) or np.isinf(val):
                                valid = False
                                break
                            temp_values.append(val)
                        except:
                            valid = False
                            break
                    
                    if valid and len(temp_values) == len(TARGET_DEPTHS):
                        cache[str(pidx)] = {
                            'lat': float(lat),
                            'lon': float(lon),
                            'date': date_str,
                            'temp': {str(d): v for d, v in zip(TARGET_DEPTHS, temp_values)},
                            'source': 'GLORYS12'
                        }
                        matched += 1
                except Exception as e:
                    pass
            
            profiles_matched += matched
            dates_processed += 1
            print(f'  Matched: {matched}/{len(profiles)} profiles')
            
            # Save cache after each date
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f)
            
            ds.close()
            
        except Exception as e:
            print(f'  ERROR: {e}')
            continue
    
    elapsed = time.time() - start_time
    print(f'\n{"="*60}')
    print(f'EXTRACTION COMPLETE')
    print(f'Time: {elapsed:.0f}s')
    print(f'Dates processed: {dates_processed}')
    print(f'Profiles matched: {profiles_matched}')
    print(f'Total cached: {len(cache)}')
    print(f'{"="*60}')
