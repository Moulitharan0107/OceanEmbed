#!/usr/bin/env python3
"""
Robust batch downloader for surface ocean data from NOAA ERDDAP.
Downloads one product at a time with rate-limit-aware delays.
Caches every successful request and can resume if interrupted.

Products:
  - SST: ncdcOisst21Agg (OISST v2.1) - needs zlev=0.0 axis
  - SSH: nesdisSSH1day (NESDIS SSH) - time, lat, lon
  - Wind: ccmp_31 (CCMP v2.1 wind) - uwnd, vwnd - time, lat, lon
  - SSS: Will try coastwatchSMOSv662SSS3day then HYCOM
"""

import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────
BASE_URL = "https://coastwatch.pfeg.noaa.gov/erddap"
RAW_DIR = Path("data/raw/argo")
SURFACE_CACHE = Path("data/raw/surface_cache")
SURFACE_CACHE.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Rate limit: minimum seconds between requests
MIN_DELAY = 4.0
MAX_DELAY = 120.0
MAX_RETRIES = 8

# Products to fetch (dataset_id, variable, axis_info)
PRODUCTS = {
    "sst": {
        "dataset_id": "ncdcOisst21Agg",
        "variable": "sst",
        "has_zlev": True,
        "time_fmt": "%Y-%m-%dT12:00:00Z",
    },
    "ssh": {
        "dataset_id": "nesdisSSH1day",
        "variable": "sla",
        "has_zlev": False,
        "time_fmt": "%Y-%m-%dT12:00:00Z",
    },
    "u10": {
        "dataset_id": "ccmp_31",
        "variable": "uwnd",
        "has_zlev": False,
        "time_fmt": "%Y-%m-%dT12:00:00Z",
    },
    "v10": {
        "dataset_id": "ccmp_31",
        "variable": "vwnd",
        "has_zlev": False,
        "time_fmt": "%Y-%m-%dT12:00:00Z",
    },
}

# Target depth grid for Argo profiles
TARGET_DEPTHS = [
    0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200,
    250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000
]


# ── Helpers ──────────────────────────────────────────────────────────

def fetch_with_backoff(url, timeout=90):
    """Fetch URL with exponential backoff on rate limits."""
    delay = MIN_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(delay)
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.text
            elif r.status_code == 429:
                delay = min(delay * 2, MAX_DELAY)
                print(f"    429 rate-limited, waiting {delay:.0f}s...")
                continue
            elif r.status_code in (502, 503, 504):
                delay = min(delay * 1.5, MAX_DELAY)
                print(f"    {r.status_code} server error, waiting {delay:.0f}s...")
                continue
            else:
                print(f"    HTTP {r.status_code}: {r.text[:120]}")
                return None
        except requests.exceptions.Timeout:
            delay = min(delay * 2, MAX_DELAY)
            print(f"    Timeout, waiting {delay:.0f}s...")
        except Exception as e:
            print(f"    Error: {e}")
            delay = min(delay * 1.5, MAX_DELAY)
    return None


def build_url(dataset_id, variable, lat, lon, date_str, has_zlev=False):
    """Build ERDDAP griddap URL for point extraction (nearest-neighbor)."""
    if has_zlev:
        # Axes: time, zlev, latitude, longitude
        constraint = (
            f"{variable}"
            f"[({date_str}):1:({date_str})]"
            f"[(0.0):1:(0.0)]"
            f"[({lat:.4f}):1:({lat:.4f})]"
            f"[({lon:.4f}):1:({lon:.4f})]"
        )
    else:
        # Axes: time, latitude, longitude
        constraint = (
            f"{variable}"
            f"[({date_str}):1:({date_str})]"
            f"[({lat:.4f}):1:({lat:.4f})]"
            f"[({lon:.4f}):1:({lon:.4f})]"
        )
    return f"{BASE_URL}/griddap/{dataset_id}.csv?{constraint}"


def parse_value(text):
    """Extract numeric value from ERDDAP CSV response."""
    if not text:
        return None
    lines = text.strip().split('\n')
    if len(lines) < 3:  # header + units + at least one data row
        return None
    try:
        val = float(lines[2].split(',')[1])  # skip time index, get value
        if np.isnan(val) or abs(val) > 999:
            return None
        return val
    except (IndexError, ValueError):
        return None


def get_cache_file(product_name):
    return SURFACE_CACHE / f"{product_name}_values.json"


def load_cache(product_name):
    path = get_cache_file(product_name)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(product_name, data):
    path = get_cache_file(product_name)
    with open(path, 'w') as f:
        json.dump(data, f)


# ── Main pipeline ────────────────────────────────────────────────────

def load_argo_profiles():
    """Load QC'd Argo profiles."""
    df = pd.read_parquet(RAW_DIR / "argo_profiles_qc.parquet")
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    print(f"Loaded {len(df)} Argo profiles ({df['float_id'].nunique()} floats)")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Lat: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
    print(f"  Lon: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
    return df


def fetch_product(profiles, product_name, config):
    """Fetch one surface product for all profiles, with caching."""
    print(f"\n{'='*50}")
    print(f"Fetching {product_name.upper()} from {config['dataset_id']}")
    print(f"{'='*50}")

    cache = load_cache(product_name)
    results = {}
    to_fetch = []

    for idx, row in profiles.iterrows():
        key = f"{idx}"
        if key in cache:
            results[idx] = cache[key]
        else:
            to_fetch.append((idx, row))

    cache_hits = len(profiles) - len(to_fetch)
    print(f"  Cache hits: {cache_hits}/{len(profiles)}")
    print(f"  To fetch: {len(to_fetch)}")

    if not to_fetch:
        valid = sum(1 for v in results.values() if v is not None)
        print(f"  All cached. Valid: {valid}/{len(results)}")
        return results

    fetched_count = 0
    valid_count = 0

    for i, (idx, row) in enumerate(to_fetch):
        date_str = row["date"].strftime(config["time_fmt"])
        url = build_url(
            config["dataset_id"], config["variable"],
            row["latitude"], row["longitude"], date_str,
            has_zlev=config["has_zlev"]
        )

        text = fetch_with_backoff(url)
        value = parse_value(text)

        cache[key] = value
        results[idx] = value
        fetched_count += 1
        if value is not None:
            valid_count += 1

        # Progress every 20
        if (i + 1) % 20 == 0:
            pct = 100 * valid_count / max(1, fetched_count)
            print(f"  Progress: {i+1}/{len(to_fetch)} fetched, "
                  f"{valid_count} valid ({pct:.0f}%)")

        # Save cache every 50 requests
        if (i + 1) % 50 == 0:
            save_cache(product_name, cache)

    # Final save
    save_cache(product_name, cache)
    total_valid = sum(1 for v in results.values() if v is not None)
    print(f"  DONE: {total_valid}/{len(profiles)} valid values")
    return results


def fetch_sss(profiles):
    """Try to fetch SSS. Try SMOS first, fall back to noting it's unavailable."""
    print(f"\n{'='*50}")
    print(f"Fetching SSS (Sea Surface Salinity)")
    print(f"{'='*50}")

    # Try SMOS SSS
    print("  Trying SMOS SSS (coastwatchSMOSv662SSS3day)...")
    
    # Test connectivity first
    test_url = build_url(
        "coastwatchSMOSv662SSS3day", "sss",
        profiles.iloc[0]["latitude"], profiles.iloc[0]["longitude"],
        profiles.iloc[0]["date"].strftime("%Y-%m-%dT12:00:00Z"),
        has_zlev=False
    )
    text = fetch_with_backoff(test_url)
    if text:
        val = parse_value(text)
        if val is not None:
            print(f"  SMOS SSS test OK: {val:.2f} PSU")
            # Now fetch all
            config = {
                "dataset_id": "coastwatchSMOSv662SSS3day",
                "variable": "sss",
                "has_zlev": False,
                "time_fmt": "%Y-%m-%dT12:00:00Z",
            }
            return fetch_product(profiles, "sss", config)
        else:
            print("  SMOS SSS returned NaN for test point")
    else:
        print("  SMOS SSS unavailable (403 or connection error)")
    
    # Try HYCOM salinity
    print("  Trying HYCOM salinity (nrlHycomGLBu008e912D)...")
    test_url = build_url(
        "nrlHycomGLBu008e912D", "salinity_0",
        profiles.iloc[0]["latitude"], profiles.iloc[0]["longitude"],
        profiles.iloc[0]["date"].strftime("%Y-%m-%dT12:00:00Z"),
        has_zlev=False
    )
    text = fetch_with_backoff(test_url)
    if text:
        val = parse_value(text)
        if val is not None:
            print(f"  HYCOM salinity test OK: {val:.2f} PSU")
            config = {
                "dataset_id": "nrlHycomGLBu008e912D",
                "variable": "salinity_0",
                "has_zlev": False,
                "time_fmt": "%Y-%m-%dT12:00:00Z",
            }
            return fetch_product(profiles, "sss", config)
        else:
            print("  HYCOM returned NaN")
    else:
        print("  HYCOM unavailable")

    print("  WARNING: SSS data unavailable from all sources.")
    print("  SSS column will be marked as missing in the final dataset.")
    return {idx: None for idx in profiles.index}


def build_final_dataset(profiles, surface_data):
    """Combine Argo profiles with surface data into final ML dataset."""
    print(f"\n{'='*50}")
    print("Building final dataset")
    print(f"{'='*50}")

    depth_cols = [f"T_{d}m" for d in TARGET_DEPTHS if f"T_{d}m" in profiles.columns]

    records = []
    for idx, row in profiles.iterrows():
        record = {
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "date": str(row["date"].date()),
            "float_id": row["float_id"],
            "max_depth": row["max_depth"],
        }
        # Surface features
        for var in ["sst", "ssh", "sss", "u10", "v10"]:
            record[var] = surface_data.get(var, {}).get(idx)

        # Temperature profile
        for col in depth_cols:
            record[col] = row[col]

        records.append(record)

    df = pd.DataFrame(records)

    # Report coverage
    print(f"\nTotal records: {len(df)}")
    for var in ["sst", "ssh", "sss", "u10", "v10"]:
        valid = df[var].notna().sum()
        print(f"  {var}: {valid}/{len(df)} valid ({100*valid/len(df):.1f}%)")

    # Complete cases (all surface vars present)
    complete = df.dropna(subset=["sst", "ssh", "u10", "v10"])
    print(f"\n  Complete (sst+ssh+u10+v10): {len(complete)}/{len(df)}")

    complete_all = df.dropna(subset=["sst", "ssh", "sss", "u10", "v10"])
    print(f"  Complete (all 5 surface): {len(complete_all)}/{len(df)}")

    return df


def save_dataset(df, profiles_with_sss):
    """Save final dataset and metadata."""
    # Save main dataset
    out_path = PROCESSED_DIR / "oceanembed_real_dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved dataset: {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

    # Save metadata
    metadata = {
        "dataset_type": "REAL DATA - not synthetic",
        "n_samples": len(df),
        "date_range": [df["date"].min(), df["date"].max()],
        "lat_range": [df["latitude"].min(), df["latitude"].max()],
        "lon_range": [df["longitude"].min(), df["longitude"].max()],
        "n_floats": df["float_id"].nunique(),
        "surface_features": {
            "sst": {"source": "NOAA OISST v2.1", "variable": "sst", "unit": "degree_C",
                     "erddap_dataset": "ncdcOisst21Agg", "resolution": "0.25 degree daily"},
            "ssh": {"source": "NESDIS Satellite Altimetry", "variable": "sla", "unit": "m",
                     "erddap_dataset": "nesdisSSH1day", "resolution": "~0.25 degree daily"},
            "sss": {"source": "SMOS L3 SSS or HYCOM" if profiles_with_sss else "UNAVAILABLE",
                     "variable": "sss", "unit": "PSU",
                     "erddap_dataset": "coastwatchSMOSv662SSS3day",
                     "resolution": "~0.25 degree 3-day"},
            "u10": {"source": "CCMP v2.1 Wind", "variable": "uwnd", "unit": "m/s",
                     "erddap_dataset": "ccmp_31", "resolution": "0.25 degree 6-hourly"},
            "v10": {"source": "CCMP v2.1 Wind", "variable": "vwnd", "unit": "m/s",
                     "erddap_dataset": "ccmp_31", "resolution": "0.25 degree 6-hourly"},
        },
        "argo_source": "NOAA ERDDAP Argo (via coriolis) - Indian Ocean subset",
        "depth_grid_m": [d for d in TARGET_DEPTHS if f"T_{d}m" in df.columns],
        "interpolation": "nearest-neighbor for surface data, linear interpolation for Argo depth profiles",
        "temporal_matching": "same day (±0 days), using noon (12:00 UTC) for gridded products",
        "spatial_matching": "nearest-neighbor at original grid resolution",
        "quality_control": "Argo QC flags applied, profiles with <15 valid levels or <1000m depth excluded",
        "creation_date": datetime.now().isoformat(),
        "interpolation_method": {
            "surface_to_argo": "nearest-neighbor spatial, same-day temporal",
            "argo_depth": "numpy.interp linear interpolation to target depth grid"
        }
    }

    meta_path = PROCESSED_DIR / "oceanembed_real_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    return metadata


def main():
    print("=" * 60)
    print("OceanEmbed Real Data Ingestion Pipeline")
    print("Surface satellite data -> Argo profile matching")
    print("=" * 60)

    # 1. Load Argo profiles
    profiles = load_argo_profiles()

    # 2. Fetch surface data product by product
    surface_data = {}
    for product_name, config in PRODUCTS.items():
        surface_data[product_name] = fetch_product(profiles, product_name, config)

    # 3. Fetch SSS (special handling)
    surface_data["sss"] = fetch_sss(profiles)
    profiles_with_sss = any(v is not None for v in surface_data["sss"].values())

    # 4. Build final dataset
    df = build_final_dataset(profiles, surface_data)

    # 5. Save
    metadata = save_dataset(df, profiles_with_sss)

    # 6. Final report
    print("\n" + "=" * 60)
    print("INGESTION REPORT")
    print("=" * 60)
    print(f"Dataset type: REAL DATA")
    print(f"Argo profiles: {len(profiles)}")
    print(f"Final samples: {len(df)}")
    print(f"Floats: {df['float_id'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Lat range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
    print(f"Lon range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
    for var in ["sst", "ssh", "sss", "u10", "v10"]:
        valid = df[var].notna().sum()
        print(f"  {var}: {valid}/{len(df)} valid")
    print(f"\nProducts used:")
    print(f"  SST: NOAA OISST v2.1 via ncdcOisst21Agg")
    print(f"  SSH: NESDIS SSH via nesdisSSH1day")
    if profiles_with_sss:
        print(f"  SSS: SMOS/HYCOM (matched)")
    else:
        print(f"  SSS: UNAVAILABLE - not matched")
    print(f"  Wind: CCMP v2.1 via ccmp_31")


if __name__ == "__main__":
    main()
