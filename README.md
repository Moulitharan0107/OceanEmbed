# OceanEmbed — CNN Ocean Embedding Framework

**SIH Problem Statement SIH26066** (Ministry of Earth Sciences)
*"Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations"*

---

## Overview

OceanEmbed is a **CNN Ocean Embedding Framework** that predicts the **vertical temperature profile of the ocean** (0–1000 m depth, **15 levels**) at any location in the **North Indian Ocean** (5°–30°N, 45°–105°E), using **7 surface satellite inputs** on a standardized **0.25° × 0.25° grid**.

Built for **INCOIS (Indian National Centre for Ocean Information Services)**, supporting oceanographers and disaster management teams with rapid subsurface temperature reconstruction.

---

## Alignment with Official Problem Statement (SIH26066)

| Official Expected Solution | Status | Notes |
|---------------------------|--------|-------|
| End-to-end preprocessing pipeline (download → regrid → train → predict) | ✅ Built | Modular scripts for each stage; 0.25° regridding pipeline |
| Satellite embedding engine (CNN/ViT/Autoencoder/GNN/Attention) | ✅ CNN-based | 1D point-wise CNN (production, RMSE 1.03°C) + spatial U-Net variant; ViT/GNN not attempted (hackathon scope) |
| Deep learning reconstruction model | ✅ Built & validated | 13 model-predicted depths (10–1000m) + 2 derived (0m, 5m) = 15 total |
| Standardized daily / 0.25° output | ⚠️ Partial | Grid regridding pipeline exists; CMEMS integration covers ~10% of profiles with real reanalysis data |
| Validation framework using independent Argo | ✅ Done | 289 held-out real Argo test profiles; RMSE 1.03°C, R² 0.81 |
| PoC over Bay of Bengal / Arabian Sea | ✅ Demo covers this | Interactive map dashboard scoped to 5°–30°N, 45°–105°E |
| Surface-only inference after training | ✅ Yes | Only SST, SSH, winds (+ optional SSS/currents) needed at inference time |
| GLORYS as training target | 🔄 Partial | 302/2,992 profiles extracted; proof-of-concept model trained but underperforms Argo labels |

**Region note:** The official spec specifies 5°–30°N, 45°–105°E. Our training data actually spans −30° to +26.83°N, 32° to 120°E (full Indian Ocean). The model's real lat coverage tops out at ~27°N, 3° short of the official 30°N boundary. Within the 5–27°N band, coverage is strong. The northern edge (27–30°N) is a known limitation due to the Argo float distribution in the training set.

---

## Key Features

| Feature | Specification |
|---------|---------------|
| **Surface Inputs** | **7 variables:** SST, SSS, SSH, Winds (u10, v10), Currents (u, v) |
| **Grid Resolution** | **0.25° × 0.25°** uniform grid across North Indian Ocean |
| **Model Architecture** | **Spatial CNN** (U-Net encoder-decoder with residual blocks) |
| **Output** | **15-depth temperature profile** (0–1000 m) |
| **Domain** | North Indian Ocean (5°–30°N, 45°–105°E) — Official SIH26066 |

---

## 7 Surface Inputs (PPT Requirement)

| # | Variable | Description | Source (Used) | Official Spec Recommends | Status |
|---|----------|-------------|---------------|--------------------------|--------|
| 1 | **SST** | Sea Surface Temperature (°C) | NOAA OISST v2.1 | OSTIA 0.05° | ✅ Real (98.7%) |
| 2 | **SSS** | Sea Surface Salinity (PSU) | CMEMS GLORYS12V1 + WOA18 fallback | SMAP/SMOS 0.125° | 🔄 Partial (284 CMEMS / 2,708 fallback) |
| 3 | **SSH** | Sea Surface Height Anomaly (m) | NESDIS Satellite Altimetry | DUACS 0.25° | ✅ Real (100%) |
| 4 | **u10** | Zonal Wind Speed at 10m (m/s) | ERA5 Reanalysis | ASCAT/CCMP | ✅ Real (99.7%) |
| 5 | **v10** | Meridional Wind Speed at 10m (m/s) | ERA5 Reanalysis | ASCAT/CCMP | ✅ Real (99.7%) |
| 6 | **current_u** | Ocean Current u-component (m/s) | CMEMS GLORYS12V1 + interpolation | OSCAR L4 0.25° | 🔄 Partial (284 CMEMS / 2,708 fallback) |
| 7 | **current_v** | Ocean Current v-component (m/s) | CMEMS GLORYS12V1 + interpolation | OSCAR L4 0.25° | 🔄 Partial (284 CMEMS / 2,708 fallback) |

**Note:** SSS and currents are sourced from CMEMS GLORYS12V1 where available (284 profiles, ~9.5% coverage, achieved via stratified representative sampling within hackathon time constraints). Remaining profiles use WOA18 climatology (SSS) and nearest-neighbor interpolation (currents) as fallback. Each row is labeled with its source (`sss_source` and `currents_source` columns) for transparency.

---

## Output: 15-Depth Temperature Profile (0–1000 m)

| Depth | Type | Description |
|-------|------|-------------|
| **0 m** | **Derived** | SST (measured) — direct pass-through from input |
| **5 m** | **Derived** | Linear interpolation: (SST + T_10m) / 2 |
| 10 m | Model | Spatial CNN prediction |
| 20 m | Model | Spatial CNN prediction |
| 30 m | Model | Spatial CNN prediction |
| 50 m | Model | Spatial CNN prediction |
| 75 m | Model | Spatial CNN prediction |
| 100 m | Model | Spatial CNN prediction |
| 125 m | Model | Spatial CNN prediction |
| 150 m | Model | Spatial CNN prediction |
| 200 m | Model | Spatial CNN prediction |
| 300 m | Model | Spatial CNN prediction |
| 500 m | Model | Spatial CNN prediction |
| 700 m | Model | Spatial CNN prediction |
| 1000 m | Model | Spatial CNN prediction |

**Note:** 0m and 5m are **derived values**, not model-predicted:
- **T_0m = SST** (the real sea surface temperature used as model input)
- **T_5m = (SST + T_10m) / 2** (linear interpolation between measured SST and model's 10m prediction)

---

## Model Architecture — CNN Ocean Embedding Framework

### Spatial CNN (U-Net Encoder-Decoder)

```
Input: (B, 7, 8, 8) — 7 surface channels on 0.25° grid patch
    ↓
Encoder (3 levels):
  ConvBlock(7→32) → ResBlock(32) → MaxPool2d
  ConvBlock(32→64) → ResBlock(64) → MaxPool2d
  ConvBlock(64→128) → ResBlock(128) → MaxPool2d
    ↓
Decoder (3 levels):
  Upsample + Skip → ConvBlock(192→64) → ResBlock(64)
  Upsample + Skip → ConvBlock(96→32) → ResBlock(32)
  Upsample + Skip → ConvBlock(64→32) → ResBlock(32)
    ↓
Global Average Pooling → Head (32→128→13)
    ↓
Residual projection from center pixel features
    ↓
Output: (B, 13) — 13 depth temperatures (10–1000 m)
```

- **Parameters:** ~1.3M (RTX 3050 compatible)
- **Patch size:** 8×8 cells (2°×2° at 0.25° resolution)
- **Uncertainty:** MC Dropout for prediction confidence
- **Spatial context:** Each prediction uses a local spatial neighborhood, not just a single point

---

## Test Set Evaluation

### 1D Point-wise Model (current production)

| Metric | Value |
|--------|-------|
| **Overall RMSE** | **1.03°C** |
| **Overall MAE** | **0.72°C** |
| **Overall R²** | **0.81** |
| **Test Samples** | 289 |
| **Model Parameters** | 300,808 |

### 7-Channel Spatial CNN (real patches, ~10% CMEMS coverage)

| Metric | Value |
|--------|-------|
| **Overall RMSE** | **1.51°C** |
| **Overall MAE** | **1.04°C** |
| **Overall R²** | **0.62** |
| **Test Samples** | 293 |
| **Model Parameters** | 1,310,421 |
| **Real Data Fraction** | 3.3% of training patches |

**Note:** Trained on real 0.25° grid patches (302/2,992 profiles with CMEMS SSS/currents + satellite SST/SSH/winds). Performance improved from the synthetic-patch baseline but remains worse than the 1D production model. The spatial architecture needs substantially more real gridded data to realize its potential.

### RMSE by Depth (1D model)

| Depth | RMSE (°C) | R² |
|-------|-----------|------|
| 10 m | 0.95 | 0.86 |
| 20 m | 1.33 | 0.75 |
| 30 m | 1.60 | 0.69 |
| 50 m | 1.78 | 0.59 |
| 75 m | 1.62 | 0.58 |
| 100 m | 1.35 | 0.68 |
| 125 m | 1.09 | 0.78 |
| 150 m | 0.81 | 0.80 |
| 200 m | 0.61 | 0.85 |
| 300 m | 0.66 | 0.85 |
| 500 m | 0.54 | 0.87 |
| 700 m | 0.52 | 0.86 |
| 1000 m | 0.42 | 0.93 |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI backend
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Start the Streamlit dashboard (in a separate terminal)
streamlit run streamlit_app.py
```

- **Interactive Map Dashboard:** http://localhost:8000
- **Streamlit Dashboard:** http://localhost:8501
- **API Documentation:** http://localhost:8000/docs

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Predict 15-depth temperature profile |
| `/api/metrics` | GET | Get model evaluation metrics |
| `/api/status` | GET | Get model and data status |
| `/api/features` | GET | Get 7 surface features for a location |
| `/api/depth-levels` | GET | Get 15 depth levels |
| `/` | GET | Interactive INCOIS map dashboard |

---

## Dashboard Features

- **Interactive map** for lat/lon selection (North Indian Ocean)
- **7 surface input controls** (SST, SSS, SSH, u10, v10, current_u, current_v)
- **15-depth temperature profile** visualization with Plotly
- **Uncertainty bands** from MC Dropout
- **Nearest Argo float comparison** for validation
- **Model metrics** from held-out test set
- **Data source labels** for each surface variable

---

## Data Pipeline — 0.25° Regridding

```bash
# Regrid Argo profiles to 0.25° grid
python scripts/regrid_to_025.py

# Fill SSS with climatology
python scripts/fill_sss_climatology.py

# Fill currents with interpolation
python scripts/fill_currents_interpolation.py
```

**Grid configuration:**
- Domain: 5–30°N, 45–105°E (Official SIH26066)
- Resolution: 0.25° × 0.25° (100 × 240 cells)
- Variables: SST, SSS, SSH, u10, v10, current_u, current_v
- Output: `data/gridded/daily_gridded.npz`

---

## Project Structure

```
oceanembed/
├── streamlit_app.py              # Streamlit dashboard
├── config.py                     # Central configuration (7 inputs, 0.25° grid)
├── backend/
│   └── app.py                    # FastAPI backend (7-input API)
├── models/
│   ├── ocean_model.py            # 1D CNN model architecture
│   ├── spatial_model.py          # Spatial CNN (U-Net) architecture
│   ├── spatial_trainer.py        # RTX 3050 training pipeline
│   ├── training.py               # Training pipeline
│   └── checkpoints/              # Saved model weights
│       ├── ocean_embed_real_best.pt      # Main model (1D, RMSE 1.03°C)
│       ├── ocean_embed_spatial_7ch.pt    # Spatial CNN (7-channel)
│       └── BACKUP_ocean_embed_real_best_13level.pt  # Safety backup
├── scripts/
│   ├── surface_lookup.py         # Real-time surface data lookup
│   ├── regrid_to_025.py          # 0.25° regridding pipeline
│   ├── fill_sss_climatology.py   # SSS climatological fallback
│   ├── fill_currents_interpolation.py  # Current interpolation
│   ├── train_spatial_7ch.py      # Spatial CNN training script
│   ├── data_ingestion.py         # Argo data pipeline
│   └── glorys_batch_extract.py   # GLORYS extraction
├── data/
│   ├── gridded/                  # 0.25° gridded NetCDF
│   ├── processed/                # Argo profiles
│   └── reports/                  # Evaluation metrics
├── experimental/                 # Quarantined scripts
├── backups/                      # Safety backups
├── static/                       # Frontend assets
└── PRESENTATION_NOTES.md         # Judge Q&A reference
```

---

## Data Sources

| Variable | Source (Used) | Official Spec Recommends | Coverage | Status |
|----------|---------------|--------------------------|----------|--------|
| **SST** | NOAA OISST v2.1 | OSTIA 0.05° | 98.7% | ✅ Used |
| **SSS** | CMEMS GLORYS12V1 + WOA18 fallback | SMAP/SMOS 0.125° | ~10% CMEMS | 🔄 Partial (284/2,992 profiles) |
| **SSH** | NESDIS Satellite Altimetry | DUACS 0.25° | 100% | ✅ Used |
| **Wind** | ERA5 Reanalysis | ASCAT/CCMP | 99.7% | ✅ Used |
| **Currents** | CMEMS GLORYS12V1 + interpolation | OSCAR L4 0.25° | ~10% CMEMS | 🔄 Partial (284/2,992 profiles) |
| **GLORYS** | CMEMS GLORYS12V1 (training target) | GLORYS12V1 (training target) | ~10% | 🔄 Partial (302/2,992 profiles) |

---

## CMEMS Integration Status

**Copernicus Marine Service (CMEMS)** is the primary real data source for SSS, currents, and GLORYS reanalysis. The `copernicusmarine` Python toolbox is authenticated and working.

**Current progress:** 302/2,992 profiles (~10.1%) fetched from CMEMS via stratified representative sampling. Each query takes ~60 seconds. Full coverage would require ~48 hours of continuous API calls.

**To continue extraction:**
```bash
python scripts/fetch_cmems_batch.py --limit 100    # fetch next 100 uncached profiles
python scripts/fetch_cmems_batch.py --status         # check cache progress
python scripts/apply_cmems_to_dataset.py              # apply cached data to dataset
```

See `docs/COPERNICUS_SETUP.md` for authentication instructions.

---

## Known Limitations

1. **Northern Edge Gap:** The official SIH26066 domain extends to 30°N, but our training data (Argo floats) only reaches ~26.83°N. The model's reliable coverage is therefore 5–27°N, 45–105°E — 3° short of the official northern boundary. This is due to the Argo float distribution in the training set, not a code limitation.
2. **SSS/Currents Coverage:** CMEMS integration covers ~10% of profiles (284/2,992) with real data, achieved via stratified representative sampling. Full extraction requires ~48 hours of continuous API calls at ~60s per profile.
3. **Spatial CNN:** Trained on real 0.25° grid patches (~10% CMEMS coverage). RMSE 1.51°C vs. 1D production's 1.03°C — needs substantially more real gridded data to realize its potential.
4. **Dataset Substitution:** Some datasets differ from the official spec's recommendations (e.g., NOAA OISST instead of OSTIA, ERA5 instead of ASCAT/CCMP). This was due to data access constraints; the architecture supports swapping in any gridded dataset.
5. **Thermocline Accuracy:** Higher RMSE at 75–200m due to natural variability in the thermocline region.

---

## Safety Backup

If the checkpoint is lost:
- **Backup:** `models/checkpoints/BACKUP_ocean_embed_real_best_13level.pt`
- **Restore:** `cp models/checkpoints/BACKUP_ocean_embed_real_best_13level.pt models/checkpoints/ocean_embed_real_best.pt`

---

## Acknowledgments

- **INCOIS (Indian National Centre for Ocean Information Services)** — primary stakeholder
- **Argo Program** for global ocean profiling data
- **NOAA OISST** for sea surface temperature
- **NESDIS** for satellite altimetry
- **ERA5** for wind reanalysis
- **WOA18** for sea surface salinity climatology
- **Smart India Hackathon 2026** for the problem statement
