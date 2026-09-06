"""
Apply cached CMEMS data to the training dataset.
Updates sss_source, currents_source columns and fills real SSS/currents values.
"""
import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path

CACHE_DIR = Path('data/raw/cmems_cache')
DATASET_PATH = Path('data/processed/oceanembed_real_dataset.parquet')

def main():
    df = pd.read_parquet(DATASET_PATH)
    df['date_str'] = df['date'].astype(str).str[:10]

    # Load all cached CMEMS profiles
    cache = {}
    for f in CACHE_DIR.glob('cmems_*.json'):
        with open(f) as fh:
            cache[f.stem] = json.load(fh)

    print(f"Dataset: {len(df)} profiles, Cache: {len(cache)} files")

    # Initialize source columns
    if 'sss_source' not in df.columns:
        df['sss_source'] = 'climatology_fallback'
    if 'currents_source' not in df.columns:
        df['currents_source'] = 'interpolated_fallback'

    cmems_count = 0
    for idx, row in df.iterrows():
        key = f"cmems_{row['date_str']}_{row['latitude']:.2f}_{row['longitude']:.2f}"
        if key in cache:
            c = cache[key]
            if c.get('sss') is not None:
                df.at[idx, 'sss'] = c['sss']
                df.at[idx, 'sss_source'] = 'cmems'
            if c.get('current_u') is not None:
                df.at[idx, 'current_u'] = c['current_u']
            if c.get('current_v') is not None:
                df.at[idx, 'current_v'] = c['current_v']
            if c.get('current_u') is not None or c.get('current_v') is not None:
                df.at[idx, 'currents_source'] = 'cmems'
            cmems_count += 1

    df.to_parquet(DATASET_PATH, index=False)

    print(f"\nApplied {cmems_count} CMEMS profiles to dataset")
    print(f"\nSSS source distribution:")
    print(df['sss_source'].value_counts().to_string())
    print(f"\nCurrents source distribution:")
    print(df['currents_source'].value_counts().to_string())
    print(f"\nDataset saved to {DATASET_PATH}")

if __name__ == '__main__':
    main()
