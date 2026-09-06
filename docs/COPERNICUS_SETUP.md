# Copernicus Marine Service Setup

## Authentication

The  Python package reads credentials from:
```
~/.copernicusmarine/.copernicusmarine-credentials
```

To set up:
1. Create account at https://marine.copernicus.eu
2. Run: `copernicusmarine login`
3. Enter your username and password when prompted
4. Credentials are saved automatically

## Available Products

| Variable | Dataset ID | Product |
|----------|-----------|---------|
| SSS | `cmems_mod_glo_phy_my_0.083deg_P1D-m` | GLORYS12V1 (salinity) |
| Currents | `cmems_mod_glo_phy_my_0.083deg_P1D-m` | GLORYS12V1 (u/v) |
| Temperature | `cmems_mod_glo_phy_my_0.083deg_P1D-m` | GLORYS12V1 (thetao) |

## Usage

```bash
# Fetch CMEMS data for uncached profiles
python scripts/fetch_cmems_batch.py --limit 100

# Check cache status
python scripts/fetch_cmems_batch.py --status

# Apply cached data to dataset
python scripts/apply_cmems_to_dataset.py
```

