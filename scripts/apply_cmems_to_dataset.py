"""
Apply cached CMEMS data to the training dataset.

Replaces climatological SSS and interpolated currents with real CMEMS values
where cached. Adds source tracking columns for transparency.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = Path('data/raw/cmems_cache')
PARQUET_PATH = Path('data/processed/oceanembed_real_dataset.parquet')
TARGET_DEPTHS = [10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


def get_cache_key(lat, lon, date_str):
    return f"cmems_{date_str}_{lat:.2f}_{lon:.2f}"


def main():
    print("=" * 60)
    print("APPLY CMEMS DATA TO TRAINING DATASET")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_parquet(PARQUET_PATH)
    df['date_str'] = df['date'].astype(str).str[:10]
    print(f"Dataset: {len(df)} profiles")
    
    # Initialize source tracking columns
    df['sss_source'] = 'climatology_fallback'
    df['currents_source'] = 'interpolated_fallback'
    
    # Load CMEMS cache
    cache_files = list(CACHE_DIR.glob('*.json'))
    print(f"CMEMS cache: {len(cache_files)} profiles")
    
    # Build cache lookup
    cache = {}
    for f in cache_files:
        with open(f) as fh:
            data = json.load(fh)
        cache[f.stem] = data
    
    # Apply cached CMEMS data
    sss_updated = 0
    currents_updated = 0
    temp_updated = 0
    
    for idx, row in df.iterrows():
        key = get_cache_key(row['latitude'], row['longitude'], row['date_str'])
        if key not in cache:
            continue
        
        cmems = cache[key]
        
        # Update SSS
        if cmems.get('sss') is not None and not np.isnan(cmems['sss']):
            df.at[idx, 'sss'] = cmems['sss']
            df.at[idx, 'sss_source'] = 'cmems'
            sss_updated += 1
        
        # Update currents
        if cmems.get('current_u') is not None and not np.isnan(cmems['current_u']):
            df.at[idx, 'current_u'] = cmems['current_u']
            df.at[idx, 'current_v'] = cmems['current_v']
            df.at[idx, 'currents_source'] = 'cmems'
            currents_updated += 1
        
        # Update temperature profile from GLORYS
        temp_profile = cmems.get('temp_profile', {})
        if len(temp_profile) >= 5:
            for depth_str, temp_val in temp_profile.items():
                depth = int(depth_str)
                col = f'T_{depth}m'
                if col in df.columns:
                    df.at[idx, col] = temp_val
            temp_updated += 1
    
    print(f"\nUpdates applied:")
    print(f"  SSS: {sss_updated}/{len(df)} replaced with CMEMS ({100*sss_updated/len(df):.1f}%)")
    print(f"  Currents: {currents_updated}/{len(df)} replaced with CMEMS ({100*currents_updated/len(df):.1f}%)")
    print(f"  Temperature: {temp_updated}/{len(df)} updated from GLORYS ({100*temp_updated/len(df):.1f}%)")
    
    # Report source breakdown
    print(f"\nSSS sources:")
    print(f"  {df['sss_source'].value_counts().to_dict()}")
    print(f"\nCurrents sources:")
    print(f"  {df['currents_source'].value_counts().to_dict()}")
    
    # Verify values are reasonable
    print(f"\nSSS after update: {df['sss'].min():.2f} to {df['sss'].max():.2f} (std={df['sss'].std():.4f})")
    print(f"Current_u after update: {df['current_u'].min():.4f} to {df['current_u'].max():.4f}")
    print(f"Current_v after update: {df['current_v'].min():.4f} to {df['current_v'].max():.4f}")
    
    # Save updated dataset
    df.drop(columns=['date_str'], inplace=True)
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"\nSaved updated dataset to {PARQUET_PATH}")
    
    return sss_updated, currents_updated, temp_updated


if __name__ == '__main__':
    main()
