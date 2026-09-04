#!/usr/bin/env python3
"""
Build OceanEmbed REAL training dataset.
SST/SSH extracted from cached NetCDF. Wind via ERDDAP CSV point queries.
SSS attempted once, dropped if unavailable.
"""
import os, sys, json, time, urllib.request, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
REPORT_DIR = BASE_DIR / "reports"
CACHE_DIR = RAW_DIR / "surface_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"

TARGET_DEPTHS = [0,10,20,30,40,50,75,100,125,150,175,200,250,300,350,400,450,500,
                 600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]


def fetch_csv(dataset_id, variable, lat, lon, date_str, retries=3, delay=4):
    """ERDDAP CSV point query for a single value."""
    lat = max(-30.0, min(30.0, lat))
    lon = max(30.0, min(120.0, lon))
    c = f"{variable}[({date_str}):1:({date_str})][({lat:.4f}):1:({lat:.4f})][({lon:.4f}):1:({lon:.4f})]"
    url = f"{ERDDAP}/{dataset_id}.csv?{c}"
    for attempt in range(retries):
        try:
            time.sleep(delay)
            r = urllib.request.urlopen(urllib.request.Request(url), timeout=60)
            lines = r.read().decode().strip().split("\n")
            if len(lines) >= 3:
                val = float(lines[2].split(",")[1])
                if np.isnan(val) or abs(val) > 999:
                    return None
                return val
            return None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(delay * 3 * (attempt + 1))
            elif e.code == 404:
                return None
            else:
                time.sleep(delay * 2)
        except Exception:
            time.sleep(delay * 2)
    return None


def extract_netcdf(profiles, nc_path, variable, has_zlev=False):
    """Extract values at Argo locations from a cached NetCDF file."""
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        print(f"    Error loading {nc_path.name}: {e}")
        return {}

    results = {}
    var_data = ds[variable]
    time_vals = ds.time.values

    for idx, row in profiles.iterrows():
        # Handle both tz-aware and tz-naive timestamps
        raw_date = row["date"]
        if hasattr(raw_date, 'tzinfo') and raw_date.tzinfo is not None:
            raw_date = raw_date.tz_localize(None)
        date_np = np.datetime64(raw_date)

        if date_np < time_vals[0] or date_np > time_vals[-1]:
            continue

        t_idx = int(np.argmin(np.abs(time_vals - date_np)))
        lat_idx = int(np.argmin(np.abs(ds.latitude.values - row["latitude"])))
        lon_idx = int(np.argmin(np.abs(ds.longitude.values - row["longitude"])))

        try:
            if has_zlev:
                val = float(var_data.isel(time=t_idx, zlev=0).values[lat_idx, lon_idx])
            else:
                val = float(var_data.isel(time=t_idx).values[lat_idx, lon_idx])
            results[idx] = None if np.isnan(val) or abs(val) > 999 else val
        except Exception:
            results[idx] = None

    ds.close()
    return results


def main():
    print("=" * 60)
    print("OceanEmbed REAL Dataset Builder")
    print("=" * 60)

    # Load profiles
    profiles = pd.read_parquet(RAW_DIR / "argo/argo_profiles_qc.parquet")
    profiles["date"] = pd.to_datetime(profiles["time"]).dt.tz_localize(None).dt.normalize()
    print(f"Loaded {len(profiles)} profiles ({profiles['float_id'].nunique()} floats)")
    print(f"  {profiles['date'].min().date()} to {profiles['date'].max().date()}")

    # ── SST from cached NetCDF ──────────────────────────────────────
    print("\n--- SST (OISST v2.1 from cached NetCDF) ---")
    sst = {}
    for f in sorted(RAW_DIR.glob("oisst/sst_*.nc")):
        vals = extract_netcdf(profiles, f, "sst", has_zlev=True)
        sst.update(vals)
        n = sum(1 for v in sst.values() if v is not None)
        print(f"  {f.name}: +{len(vals)} pts, total {n} valid")
    print(f"  SST total: {sum(1 for v in sst.values() if v is not None)}/{len(profiles)}")

    # ── SSH from cached NetCDF ──────────────────────────────────────
    print("\n--- SSH (NESDIS from cached NetCDF) ---")
    ssh = {}
    for f in sorted(RAW_DIR.glob("ssh/ssh_*.nc")):
        vals = extract_netcdf(profiles, f, "sla", has_zlev=False)
        ssh.update(vals)
        n = sum(1 for v in ssh.values() if v is not None)
        print(f"  {f.name}: +{len(vals)} pts, total {n} valid")
    print(f"  SSH total: {sum(1 for v in ssh.values() if v is not None)}/{len(profiles)}")

    # ── Wind via ERDDAP CSV point queries ───────────────────────────
    print("\n--- Wind (CCMP v2.1 via ERDDAP CSV) ---")
    wind_cache = CACHE_DIR / "wind_values.json"
    wind = {"u10": {}, "v10": {}}

    if wind_cache.exists():
        with open(wind_cache) as f:
            cached = json.load(f)
        wind["u10"] = {int(k): v for k, v in cached.get("u10", {}).items()}
        wind["v10"] = {int(k): v for k, v in cached.get("v10", {}).items()}
        print(f"  Loaded from cache: {sum(1 for v in wind['u10'].values() if v is not None)} valid u10")

    need_wind = [i for i in profiles.index if i not in wind["u10"]]
    print(f"  Need to fetch: {len(need_wind)} profiles")

    for i, idx in enumerate(need_wind):
        row = profiles.loc[idx]
        ds = row["date"].strftime("%Y-%m-%dT12:00:00Z")
        wind["u10"][idx] = fetch_csv("ccmp_31", "uwnd", row["latitude"], row["longitude"], ds, delay=3)
        wind["v10"][idx] = fetch_csv("ccmp_31", "vwnd", row["latitude"], row["longitude"], ds, delay=3)

        if (i + 1) % 20 == 0:
            n = sum(1 for v in wind["u10"].values() if v is not None)
            print(f"  Wind progress: {i+1}/{len(need_wind)}, {n} valid u10")
            with open(wind_cache, "w") as f:
                json.dump(wind, f)

    with open(wind_cache, "w") as f:
        json.dump(wind, f)
    print(f"  u10: {sum(1 for v in wind['u10'].values() if v is not None)}/{len(profiles)}")
    print(f"  v10: {sum(1 for v in wind['v10'].values() if v is not None)}/{len(profiles)}")

    # ── SSS (try once, drop if fails) ───────────────────────────────
    print("\n--- SSS (attempt once) ---")
    sss_cache = CACHE_DIR / "sss_values.json"
    sss = {}
    sss_ok = False

    if sss_cache.exists():
        with open(sss_cache) as f:
            sss = {int(k): v for k, v in json.load(f).items()}
        n = sum(1 for v in sss.values() if v is not None)
        sss_ok = n > 0
        print(f"  Cached: {n} valid")
    else:
        # One test query
        row0 = profiles.iloc[0]
        test_date = row0["date"].strftime("%Y-%m-%dT12:00:00Z")
        val = fetch_csv("coastwatchSMOSv662SSS3day", "sss",
                        row0["latitude"], row0["longitude"], test_date, retries=1, delay=6)
        if val is not None:
            print(f"  SMOS SSS test OK: {val:.2f}")
            # Fetch all
            for i, idx in enumerate(profiles.index):
                row = profiles.loc[idx]
                ds_str = row["date"].strftime("%Y-%m-%dT12:00:00Z")
                sss[idx] = fetch_csv("coastwatchSMOSv662SSS3day", "sss",
                                      row["latitude"], row["longitude"], ds_str, delay=4)
                if (i + 1) % 25 == 0:
                    print(f"  SSS: {i+1}/{len(profiles)}")
            with open(sss_cache, "w") as f:
                json.dump({str(k): v for k, v in sss.items()}, f)
            sss_ok = sum(1 for v in sss.values() if v is not None) > 0
        else:
            print("  SMOS SSS unavailable. DROPPED.")

    # ── Build final dataset ──────────────────────────────────────────
    print("\n--- Building dataset ---")
    depth_cols = [f"T_{d}m" for d in TARGET_DEPTHS if f"T_{d}m" in profiles.columns]

    records = []
    for idx, row in profiles.iterrows():
        rec = {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "date": str(row["date"].date()),
            "float_id": int(row["float_id"]),
            "max_depth": float(row["max_depth"]),
            "sst": sst.get(idx),
            "ssh": ssh.get(idx),
            "u10": wind["u10"].get(idx),
            "v10": wind["v10"].get(idx),
            "sss": sss.get(idx) if sss_ok else None,
        }
        for col in depth_cols:
            v = row[col]
            rec[col] = float(v) if pd.notna(v) else None
        records.append(rec)

    df = pd.DataFrame(records)

    print(f"\nDataset: {len(df)} samples")
    for var in ["sst", "ssh", "u10", "v10", "sss"]:
        if var in df.columns:
            n = df[var].notna().sum()
            print(f"  {var}: {n}/{len(df)} ({100*n/len(df):.1f}%)")

    complete = df.dropna(subset=["sst", "ssh", "u10", "v10"])
    print(f"  Complete (sst+ssh+u10+v10): {len(complete)}/{len(df)}")

    # Save
    out = PROCESSED_DIR / "oceanembed_real_dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved: {out} ({os.path.getsize(out)/1024:.1f} KB)")

    # Metadata
    meta = {
        "dataset_type": "REAL DATA",
        "n_samples": len(df),
        "n_complete": len(complete),
        "date_range": [df["date"].min(), df["date"].max()],
        "lat_range": [float(df["latitude"].min()), float(df["latitude"].max())],
        "lon_range": [float(df["longitude"].min()), float(df["longitude"].max())],
        "n_floats": int(df["float_id"].nunique()),
        "features": {
            "sst": "NOAA OISST v2.1 (ncdcOisst21Agg) - cached quarterly NetCDF",
            "ssh": "NESDIS SSH Altimetry (nesdisSSH1day) - cached quarterly NetCDF",
            "u10": "CCMP v2.1 (ccmp_31) - ERDDAP CSV point query",
            "v10": "CCMP v2.1 (ccmp_31) - ERDDAP CSV point query",
            "sss": "SMOS (coastwatchSMOSv662SSS3day)" if sss_ok else "DROPPED - unavailable",
        },
        "depth_grid_m": [d for d in TARGET_DEPTHS if f"T_{d}m" in df.columns],
        "argo_source": "NOAA ERDDAP Argo, Indian Ocean 2019-2024",
        "creation_date": datetime.now().isoformat(),
    }
    with open(PROCESSED_DIR / "oceanembed_real_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Report
    report = (
        "=" * 60 + "\n"
        "OCEANEMBED REAL DATA INGESTION REPORT\n" + "=" * 60 + "\n"
        f"Argo raw records: 12,136,346\n"
        f"Argo raw floats: 391\n"
        f"After QC: {len(profiles)} profiles\n"
        f"  Reaching 1000m: {int((profiles['max_depth'] >= 1000).sum())}\n"
        f"  Reaching 2000m: {int((profiles['max_depth'] >= 1900).sum())}\n"
        f"Final samples: {len(df)}\n"
        f"Complete (sst+ssh+wind): {len(complete)}\n"
        f"Unique floats: {df['float_id'].nunique()}\n"
        f"Date range: {df['date'].min()} to {df['date'].max()}\n"
        f"Lat: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}\n"
        f"Lon: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}\n"
        f"Depth levels: {len(depth_cols)}\n\n"
        "Surface coverage:\n"
    )
    for var in ["sst", "ssh", "u10", "v10", "sss"]:
        if var in df.columns:
            n = int(df[var].notna().sum())
            report += f"  {var}: {n}/{len(df)} ({100*n/len(df):.1f}%)\n"
    report += "\nProducts:\n"
    for k, v in meta["features"].items():
        report += f"  {k}: {v}\n"

    with open(REPORT_DIR / "ingestion_report.txt", "w") as f:
        f.write(report)
    print(f"\n{report}")
    return df


if __name__ == "__main__":
    main()
