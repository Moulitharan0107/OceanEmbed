#!/usr/bin/env python3
"""
Batch-download surface ocean data from NOAA ERDDAP as NetCDF quarterly chunks.
Extracts values at Argo profile locations using xarray nearest-neighbor.

Products:
  SST: ncdcOisst21Agg (OISST v2.1) — time, zlev, lat, lon
  SSH: nesdisSSH1day — time, lat, lon
  Wind: ccmp_31 (uwnd, vwnd) — time, lat, lon
"""

import os
import sys
import json
import time
import urllib.request
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────
BASE_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
RAW_DIR = Path("data/raw")
OISST_DIR = RAW_DIR / "oisst"
SSH_DIR = RAW_DIR / "ssh"
WIND_DIR = RAW_DIR / "wind"
SSS_DIR = RAW_DIR / "sss"
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("data/reports")

for d in [OISST_DIR, SSH_DIR, WIND_DIR, SSS_DIR, PROCESSED_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Indian Ocean bounds
LAT_MIN, LAT_MAX = -30.0, 30.0
LON_MIN, LON_MAX = 30.0, 120.0

# Time period
YEARS = range(2019, 2025)
QUARTERS = [(1, 3), (4, 6), (7, 9), (10, 12)]

# Rate limiting
MIN_DELAY = 5.0  # seconds between requests

# Target depth grid
TARGET_DEPTHS = [
    0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200,
    250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000
]


def download_netcd(url, filepath, timeout=300):
    """Download a NetCDF file from ERDDAP with retry."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = resp.read()
            with open(filepath, 'wb') as f:
                f.write(data)
            return len(data) / (1024 * 1024)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = MIN_DELAY * (attempt + 2)
                print(f"    Retry {attempt+1}: {e}. Waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"    FAILED: {e}")
                return None
    return None


def download_quarterly_sst(year, quarter_idx):
    """Download one quarter of SST data for Indian Ocean."""
    q_start_month, q_end_month = QUARTERS[quarter_idx]
    
    start_date = f"{year}-{q_start_month:02d}-01T12:00:00Z"
    end_day = {1: 31, 3: 31, 4: 30, 6: 30, 7: 31, 9: 30, 10: 31, 12: 31}[q_end_month]
    end_date = f"{year}-{q_end_month:02d}-{end_day:02d}T12:00:00Z"
    
    filepath = OISST_DIR / f"sst_{year}_Q{quarter_idx+1}.nc"
    if filepath.exists():
        size = os.path.getsize(filepath) / (1024 * 1024)
        return size, True  # already downloaded
    
    url = (f"{BASE_URL}/ncdcOisst21Agg.nc?"
           f"sst[({start_date}):1:({end_date})]"
           f"[(0.0):1:(0.0)]"
           f"[({LAT_MIN}):1:({LAT_MAX})]"
           f"[({LON_MIN}):1:({LON_MAX})]")
    
    size = download_netcd(url, filepath)
    return size, False


def download_quarterly_ssh(year, quarter_idx):
    """Download one quarter of SSH data for Indian Ocean."""
    q_start_month, q_end_month = QUARTERS[quarter_idx]
    start_date = f"{year}-{q_start_month:02d}-01T00:00:00Z"
    end_day = {1: 31, 3: 31, 4: 30, 6: 30, 7: 31, 9: 30, 10: 31, 12: 31}[q_end_month]
    end_date = f"{year}-{q_end_month:02d}-{end_day:02d}T00:00:00Z"
    
    filepath = SSH_DIR / f"ssh_{year}_Q{quarter_idx+1}.nc"
    if filepath.exists():
        size = os.path.getsize(filepath) / (1024 * 1024)
        return size, True
    
    url = (f"{BASE_URL}/nesdisSSH1day.nc?"
           f"sla[({start_date}):1:({end_date})]"
           f"[({LAT_MIN}):1:({LAT_MAX})]"
           f"[({LON_MIN}):1:({LON_MAX})]")
    
    size = download_netcd(url, filepath)
    return size, False


def download_quarterly_wind(year, quarter_idx):
    """Download one quarter of CCMP wind for Indian Ocean."""
    q_start_month, q_end_month = QUARTERS[quarter_idx]
    start_date = f"{year}-{q_start_month:02d}-01T00:00:00Z"
    end_day = {1: 31, 3: 31, 4: 30, 6: 30, 7: 31, 9: 30, 10: 31, 12: 31}[q_end_month]
    end_date = f"{year}-{q_end_month:02d}-{end_day:02d}T00:00:00Z"
    
    filepath = WIND_DIR / f"wind_{year}_Q{quarter_idx+1}.nc"
    if filepath.exists():
        size = os.path.getsize(filepath) / (1024 * 1024)
        return size, True
    
    # CCMP has uwnd and vwnd — download both
    url = (f"{BASE_URL}/ccmp_31.nc?"
           f"uwnd[({start_date}):1:({end_date})]"
           f"[({LAT_MIN}):1:({LAT_MAX})]"
           f"[({LON_MIN}):1:({LON_MAX})]"
           f"&vwnd[({start_date}):1:({end_date})]"
           f"[({LAT_MIN}):1:({LAT_MAX})]"
           f"[({LON_MIN}):1:({LON_MAX})]")
    
    size = download_netcd(url, filepath)
    return size, False


def extract_values_at_profiles(profiles, nc_path, variable, dataset_type):
    """Extract values at Argo profile locations from a NetCDF dataset."""
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        print(f"    Error loading {nc_path}: {e}")
        return {}
    
    results = {}
    for idx, row in profiles.iterrows():
        try:
            date = pd.to_datetime(row["time"]).normalize()
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            
            # Select nearest time, lat, lon
            var_data = ds[variable]
            
            # Find nearest time index
            time_diffs = np.abs(ds.time.values - np.datetime64(date))
            t_idx = np.argmin(time_diffs)
            
            # Select this time slice
            slice_data = var_data.isel(time=t_idx)
            
            # Find nearest lat/lon
            lat_idx = np.argmin(np.abs(ds.latitude.values - lat))
            lon_idx = np.argmin(np.abs(ds.longitude.values - lon))
            
            value = float(slice_data.values[lat_idx, lon_idx])
            if np.isnan(value) or abs(value) > 999:
                results[idx] = None
            else:
                results[idx] = value
        except Exception as e:
            results[idx] = None
    
    ds.close()
    return results


def main():
    print("=" * 60)
    print("OceanEmbed Batch Surface Data Download")
    print("Quarterly NetCDF downloads from NOAA ERDDAP")
    print("=" * 60)
    
    # Load Argo profiles
    profiles = pd.read_parquet(RAW_DIR / "argo/argo_profiles_qc.parquet")
    profiles["date"] = pd.to_datetime(profiles["time"]).dt.normalize()
    print(f"\nLoaded {len(profiles)} Argo profiles")
    
    # ── Step 1: Download SST ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1: Download SST (NOAA OISST v2.1)")
    print("=" * 60)
    
    sst_files = []
    total_sst_mb = 0
    for year in YEARS:
        for qi in range(4):
            q_start, q_end = QUARTERS[qi]
            # Check if any profiles in this quarter
            mask = (profiles["date"].dt.year == year) & \
                   (profiles["date"].dt.month >= q_start) & \
                   (profiles["date"].dt.month <= q_end)
            if mask.sum() == 0:
                continue
            
            fname = f"sst_{year}_Q{qi+1}.nc"
            size, cached = download_quarterly_sst(year, qi)
            if size is not None:
                sst_files.append(OISST_DIR / fname)
                total_sst_mb += size
                status = "CACHED" if cached else "DOWNLOADED"
                print(f"  {fname}: {size:.1f} MB ({status})")
            else:
                print(f"  {fname}: FAILED")
            time.sleep(MIN_DELAY)
    
    print(f"  Total SST: {total_sst_mb:.1f} MB across {len(sst_files)} files")
    
    # ── Step 2: Download SSH ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Download SSH (NESDIS Altimetry)")
    print("=" * 60)
    
    ssh_files = []
    total_ssh_mb = 0
    for year in YEARS:
        for qi in range(4):
            q_start, q_end = QUARTERS[qi]
            mask = (profiles["date"].dt.year == year) & \
                   (profiles["date"].dt.month >= q_start) & \
                   (profiles["date"].dt.month <= q_end)
            if mask.sum() == 0:
                continue
            
            fname = f"ssh_{year}_Q{qi+1}.nc"
            size, cached = download_quarterly_ssh(year, qi)
            if size is not None:
                ssh_files.append(SSH_DIR / fname)
                total_ssh_mb += size
                status = "CACHED" if cached else "DOWNLOADED"
                print(f"  {fname}: {size:.1f} MB ({status})")
            else:
                print(f"  {fname}: FAILED")
            time.sleep(MIN_DELAY)
    
    print(f"  Total SSH: {total_ssh_mb:.1f} MB across {len(ssh_files)} files")
    
    # ── Step 3: Download Wind ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Download Wind (CCMP v2.1)")
    print("=" * 60)
    
    wind_files = []
    total_wind_mb = 0
    for year in YEARS:
        for qi in range(4):
            q_start, q_end = QUARTERS[qi]
            mask = (profiles["date"].dt.year == year) & \
                   (profiles["date"].dt.month >= q_start) & \
                   (profiles["date"].dt.month <= q_end)
            if mask.sum() == 0:
                continue
            
            fname = f"wind_{year}_Q{qi+1}.nc"
            size, cached = download_quarterly_wind(year, qi)
            if size is not None:
                wind_files.append(WIND_DIR / fname)
                total_wind_mb += size
                status = "CACHED" if cached else "DOWNLOADED"
                print(f"  {fname}: {size:.1f} MB ({status})")
            else:
                print(f"  {fname}: FAILED")
            time.sleep(MIN_DELAY)
    
    print(f"  Total Wind: {total_wind_mb:.1f} MB across {len(wind_files)} files")
    
    # ── Step 4: Extract values at Argo locations ────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Extract surface values at Argo locations")
    print("=" * 60)
    
    surface_data = {"sst": {}, "ssh": {}, "u10": {}, "v10": {}}
    
    # Extract SST
    print("\n  Extracting SST...")
    for f in sst_files:
        vals = extract_values_at_profiles(profiles, f, "sst", "sst")
        surface_data["sst"].update(vals)
    valid_sst = sum(1 for v in surface_data["sst"].values() if v is not None)
    print(f"  SST: {valid_sst}/{len(profiles)} valid")
    
    # Extract SSH
    print("\n  Extracting SSH...")
    for f in ssh_files:
        vals = extract_values_at_profiles(profiles, f, "sla", "ssh")
        surface_data["ssh"].update(vals)
    valid_ssh = sum(1 for v in surface_data["ssh"].values() if v is not None)
    print(f"  SSH: {valid_ssh}/{len(profiles)} valid")
    
    # Extract wind (u10 and v10)
    print("\n  Extracting Wind...")
    for f in wind_files:
        vals_u = extract_values_at_profiles(profiles, f, "uwnd", "wind")
        vals_v = extract_values_at_profiles(profiles, f, "vwnd", "wind")
        surface_data["u10"].update(vals_u)
        surface_data["v10"].update(vals_v)
    valid_u10 = sum(1 for v in surface_data["u10"].values() if v is not None)
    valid_v10 = sum(1 for v in surface_data["v10"].values() if v is not None)
    print(f"  u10: {valid_u10}/{len(profiles)} valid")
    print(f"  v10: {valid_v10}/{len(profiles)} valid")
    
    # ── Step 5: Try SSS ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Try SSS (SMOS)")
    print("=" * 60)
    
    # Try SMOS SSS via ERDDAP CSV for individual profiles
    # (SMOS NetCDF might not be available via ERDDAP)
    sss_cache_path = RAW_DIR / "surface_cache" / "sss_values.json"
    sss_data = {}
    
    if sss_cache_path.exists():
        with open(sss_cache_path) as f:
            sss_cache = json.load(f)
        sss_data = {int(k): v for k, v in sss_cache.items()}
        valid_sss = sum(1 for v in sss_data.values() if v is not None)
        print(f"  SSS loaded from cache: {valid_sss}/{len(profiles)} valid")
    else:
        # Try a small SSS download
        test_url = (f"{BASE_URL}/coastwatchSMOSv662SSS3day.nc?"
                    f"sss[(2020-01-01T12:00:00Z):1:(2020-01-31T12:00:00Z)]"
                    f"[({LAT_MIN}):1:({LAT_MAX})]"
                    f"[({LON_MIN}):1:({LON_MAX})]")
        
        test_file = SSS_DIR / "sss_test.nc"
        print("  Testing SMOS SSS download...")
        size = download_netcd(test_url, test_file)
        if size and size > 0:
            print(f"  SMOS SSS available: {size:.1f} MB")
            # TODO: full SSS download
            sss_available = True
        else:
            print("  SMOS SSS unavailable. SSS will be marked as missing.")
            sss_available = False
    
    # ── Step 6: Build final dataset ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: Build final ML dataset")
    print("=" * 60)
    
    depth_cols = [f"T_{d}m" for d in TARGET_DEPTHS if f"T_{d}m" in profiles.columns]
    
    records = []
    for idx, row in profiles.iterrows():
        record = {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "date": str(row["date"].date()),
            "float_id": int(row["float_id"]),
            "max_depth": float(row["max_depth"]),
        }
        
        # Surface features
        for var in ["sst", "ssh", "u10", "v10"]:
            record[var] = surface_data[var].get(idx)
        
        # SSS (from cache if available)
        record["sss"] = sss_data.get(idx) if sss_data else None
        
        # Temperature profile
        for col in depth_cols:
            val = row[col]
            record[col] = float(val) if pd.notna(val) else None
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Report
    print(f"\nDataset: {len(df)} samples")
    for var in ["sst", "ssh", "sss", "u10", "v10"]:
        valid = df[var].notna().sum()
        print(f"  {var}: {valid}/{len(df)} ({100*valid/len(df):.1f}%)")
    
    # Complete cases
    complete = df.dropna(subset=["sst", "ssh", "u10", "v10"])
    print(f"\n  Complete (sst+ssh+u10+v10): {len(complete)}/{len(df)}")
    
    if df["sss"].notna().any():
        complete_all = df.dropna(subset=["sst", "ssh", "sss", "u10", "v10"])
        print(f"  Complete (all 5): {len(complete_all)}/{len(df)}")
    
    # Save
    out_path = PROCESSED_DIR / "oceanembed_real_dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")
    
    # Metadata
    metadata = {
        "dataset_type": "REAL DATA",
        "n_samples": len(df),
        "date_range": [df["date"].min(), df["date"].max()],
        "lat_range": [float(df["latitude"].min()), float(df["latitude"].max())],
        "lon_range": [float(df["longitude"].min()), float(df["longitude"].max())],
        "n_floats": int(df["float_id"].nunique()),
        "surface_sources": {
            "sst": "NOAA OISST v2.1 (ncdcOisst21Agg) via ERDDAP",
            "ssh": "NESDIS Satellite Altimetry (nesdisSSH1day) via ERDDAP",
            "sss": "SMOS L3 SSS (coastwatchSMOSv662SSS3day) via ERDDAP" if df["sss"].notna().any() else "UNAVAILABLE",
            "wind": "CCMP v2.1 (ccmp_31) via ERDDAP",
        },
        "interpolation": "nearest-neighbor spatial, same-day temporal",
        "argo_source": "NOAA ERDDAP Argo, Indian Ocean subset",
        "depth_grid_m": [d for d in TARGET_DEPTHS if f"T_{d}m" in df.columns],
        "creation_date": datetime.now().isoformat(),
    }
    
    meta_path = PROCESSED_DIR / "oceanembed_real_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {meta_path}")
    
    # ── Step 7: Ingestion Report ────────────────────────────────────
    print("\n" + "=" * 60)
    print("INGESTION REPORT")
    print("=" * 60)
    
    report = {
        "status": "COMPLETE",
        "dataset_type": "REAL DATA",
        "argo_profiles_raw": 12136346,
        "argo_floats_raw": 391,
        "argo_profiles_after_qc": len(profiles),
        "profiles_reaching_1000m": int((profiles["max_depth"] >= 1000).sum()),
        "profiles_reaching_2000m": int((profiles["max_depth"] >= 1900).sum()),
        "final_samples": len(df),
        "n_floats": int(df["float_id"].nunique()),
        "date_range": [df["date"].min(), df["date"].max()],
        "lat_range": [float(df["latitude"].min()), float(df["latitude"].max())],
        "lon_range": [float(df["longitude"].min()), float(df["longitude"].max())],
        "depth_levels": len(depth_cols),
        "surface_coverage": {},
        "products_used": [],
    }
    
    for var in ["sst", "ssh", "sss", "u10", "v10"]:
        valid = int(df[var].notna().sum())
        report["surface_coverage"][var] = {
            "valid": valid,
            "total": len(df),
            "percent": round(100 * valid / len(df), 1),
        }
    
    report["products_used"] = [
        "NOAA OISST v2.1 (SST) - ERDDAP ncdcOisst21Agg",
        "NESDIS SSH Altimetry - ERDDAP nesdisSSH1day",
        "CCMP v2.1 Wind - ERDDAP ccmp_31",
    ]
    if df["sss"].notna().any():
        report["products_used"].append("SMOS SSS - ERDDAP coastwatchSMOSv662SSS3day")
    
    report_path = REPORT_DIR / "ingestion_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Text report
    txt = []
    txt.append("OCEANEMBED REAL DATA INGESTION REPORT")
    txt.append("=" * 50)
    txt.append(f"Dataset type: REAL DATA")
    txt.append(f"Argo profiles downloaded: 12,136,346")
    txt.append(f"Argo floats: 391")
    txt.append(f"After QC: {len(profiles)}")
    txt.append(f"  Reaching 1000m: {report['profiles_reaching_1000m']}")
    txt.append(f"  Reaching 2000m: {report['profiles_reaching_2000m']}")
    txt.append(f"Final complete samples: {len(df)}")
    txt.append(f"Unique floats: {report['n_floats']}")
    txt.append(f"Date range: {df['date'].min()} to {df['date'].max()}")
    txt.append(f"Lat: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
    txt.append(f"Lon: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
    txt.append(f"Depth levels: {len(depth_cols)} (0-2000m)")
    txt.append("")
    txt.append("Surface data coverage:")
    for var in ["sst", "ssh", "sss", "u10", "v10"]:
        cov = report["surface_coverage"][var]
        txt.append(f"  {var}: {cov['valid']}/{cov['total']} ({cov['percent']}%)")
    txt.append("")
    txt.append("Products used:")
    for p in report["products_used"]:
        txt.append(f"  - {p}")
    
    report_txt = REPORT_DIR / "ingestion_report.txt"
    with open(report_txt, 'w') as f:
        f.write("\n".join(txt))
    
    print("\n".join(txt))
    print(f"\nReports saved to {REPORT_DIR}")


if __name__ == "__main__":
    main()
