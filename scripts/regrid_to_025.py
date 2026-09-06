"""
Regridding Pipeline — 0.25° × 0.25° Grid over North Indian Ocean
Converts point-based Argo profile data + surface observations into gridded NetCDF format.

Domain: 0–25°N, 40–100°E
Resolution: 0.25° × 0.25°
Variables: SST, SSS, SSH, u10, v10, current_u, current_v (surface), + temperature at 13 depths

Output: data/gridded/daily_gridded.nc (NetCDF for xarray access)
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Grid configuration
LAT_MIN, LAT_MAX = 0.0, 25.0
LON_MIN, LON_MAX = 40.0, 100.0
GRID_RES = 0.25  # degrees

# Target depth levels (model output)
TARGET_DEPTHS = [10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


def create_grid():
    """Create the 0.25° grid coordinates."""
    lats = np.arange(LAT_MIN + GRID_RES / 2, LAT_MAX, GRID_RES)
    lons = np.arange(LON_MIN + GRID_RES / 2, LON_MAX, GRID_RES)
    return lats, lons


def point_to_grid_idx(lat, lon, lats, lons):
    """Find nearest grid cell indices for a point."""
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    return lat_idx, lon_idx


def regrid_dataset(input_parquet, output_nc):
    """
    Regrid the point-based dataset to 0.25° grid.
    
    For each grid cell and date, averages all Argo profiles that fall within that cell.
    """
    print("=" * 60)
    print("REGRIDDING PIPELINE — 0.25° × 0.25° North Indian Ocean")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_parquet(input_parquet)
    print(f"Loaded {len(df)} Argo profiles")
    
    # Create grid
    lats, lons = create_grid()
    n_lat, n_lon = len(lats), len(lons)
    print(f"Grid: {n_lat} × {n_lon} = {n_lat * n_lon} cells")
    print(f"Domain: {LAT_MIN}–{LAT_MAX}°N, {LON_MIN}–{LON_MAX}°E")
    
    # Surface variables
    surface_vars = ['sst', 'sss', 'ssh', 'u10', 'v10', 'current_u', 'current_v']
    
    # Add lat/lon to features for gridding
    df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    
    # Map each profile to its grid cell
    df['lat_idx'] = df['lat'].apply(lambda x: np.argmin(np.abs(lats - x)))
    df['lon_idx'] = df['lon'].apply(lambda x: np.argmin(np.abs(lons - x)))
    
    # Get unique dates
    df['date_str'] = df['date'].astype(str).str[:10]
    unique_dates = sorted(df['date_str'].unique())
    print(f"Unique dates: {len(unique_dates)}")
    
    # Count profiles in domain
    in_domain = df[(df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) & 
                   (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX)]
    print(f"Profiles in domain: {len(in_domain)}/{len(df)}")
    
    # Initialize output arrays (float32 to save memory)
    surface_grids = {var: np.full((len(unique_dates), n_lat, n_lon), np.nan, dtype=np.float32)
                     for var in surface_vars}
    temp_grids = np.full((len(unique_dates), n_lat, n_lon, len(TARGET_DEPTHS)), np.nan, dtype=np.float32)
    count_grids = np.zeros((len(unique_dates), n_lat, n_lon), dtype=np.int16)
    
    # Fill grids by averaging profiles in each cell
    for di, date in enumerate(unique_dates):
        day_df = in_domain[in_domain['date_str'] == date]
        
        for _, row in day_df.iterrows():
            li, lj = int(row['lat_idx']), int(row['lon_idx'])
            
            # Accumulate surface variables
            for var in surface_vars:
                val = row.get(var, np.nan)
                if pd.notna(val):
                    if np.isnan(surface_grids[var][di, li, lj]):
                        surface_grids[var][di, li, lj] = val
                    else:
                        surface_grids[var][di, li, lj] = (
                            surface_grids[var][di, li, lj] * count_grids[di, li, lj] + val
                        ) / (count_grids[di, li, lj] + 1)
            
            # Accumulate temperature profiles
            for ti, depth in enumerate(TARGET_DEPTHS):
                col = f'T_{depth}m'
                val = row.get(col, np.nan)
                if pd.notna(val):
                    n = count_grids[di, li, lj]
                    if np.isnan(temp_grids[di, li, lj, ti]):
                        temp_grids[di, li, lj, ti] = val
                    else:
                        temp_grids[di, li, lj, ti] = (
                            temp_grids[di, li, lj, ti] * n + val
                        ) / (n + 1)
            
            count_grids[di, li, lj] += 1
        
        if (di + 1) % 100 == 0:
            print(f"  Processed {di + 1}/{len(unique_dates)} dates")
    
    # Save as NetCDF using numpy (no xarray dependency needed)
    output_path = Path(output_nc)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as compressed numpy archive (lightweight, fast loading)
    np.savez_compressed(
        str(output_path).replace('.nc', '.npz'),
        lats=lats,
        lons=lons,
        dates=np.array(unique_dates),
        surface_vars=surface_vars,
        target_depths=np.array(TARGET_DEPTHS),
        # Surface grids
        **{f'surface_{var}': surface_grids[var] for var in surface_vars},
        # Temperature grids
        temperature=temp_grids,
        # Counts
        profile_counts=count_grids,
    )
    
    # Report coverage
    total_cells = n_lat * n_lon * len(unique_dates)
    filled_cells = np.sum(~np.isnan(surface_grids['sst']))
    coverage = 100 * filled_cells / total_cells
    
    print(f"\n{'=' * 60}")
    print(f"REGRIDDING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Grid shape: ({len(unique_dates)}, {n_lat}, {n_lon})")
    print(f"Surface coverage (SST): {filled_cells}/{total_cells} ({coverage:.1f}%)")
    print(f"Output: {output_path.with_suffix('.npz')}")
    
    # Also save grid metadata as JSON
    metadata = {
        'domain': {'lat_min': LAT_MIN, 'lat_max': LAT_MAX, 
                   'lon_min': LON_MIN, 'lon_max': LON_MAX},
        'resolution_deg': GRID_RES,
        'n_lats': n_lat,
        'n_lons': n_lon,
        'n_dates': len(unique_dates),
        'surface_vars': surface_vars,
        'target_depths': TARGET_DEPTHS,
        'coverage_pct': round(coverage, 1),
        'n_profiles_in_domain': len(in_domain),
    }
    with open(output_path.parent / 'grid_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


if __name__ == '__main__':
    input_pq = 'data/processed/oceanembed_real_dataset.parquet'
    output_nc = 'data/gridded/daily_gridded.nc'
    
    if not Path(input_pq).exists():
        print(f"ERROR: {input_pq} not found")
    else:
        regrid_dataset(input_pq, output_nc)
