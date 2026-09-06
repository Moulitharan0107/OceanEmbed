# OceanEmbed — Satellite Embedding-Based Deep Learning Framework

**SIH Problem Statement SIH26066** (Ministry of Earth Sciences)
*"Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations"*

---

## Overview

OceanEmbed predicts the **vertical temperature profile of the ocean** at any location in the Indian Ocean, using only surface satellite observations as input. Built for **INCOIS (Indian National Centre for Ocean Information Services)**, supporting oceanographers and disaster management teams with rapid subsurface temperature reconstruction.

## Primary Scope

**North Indian Ocean** (Bay of Bengal + Arabian Sea, roughly 0°–25°N, 40°E–100°E) as the proof-of-concept region. The trained model covers the wider Indian Ocean (30°S–30°N, 30°E–120°E) — North Indian Ocean is the default view.

## Depth Levels (15 Total)

The system outputs **15 depth levels** for complete 0–1000m coverage:

| Depth | Type | Description |
|-------|------|-------------|
| **0 m** | **Derived** | SST (measured) — direct pass-through from input |
| **5 m** | **Derived** | Linear interpolation between SST (0m) and model's 10m prediction |
| 10 m | Model | Multi-scale CNN prediction |
| 20 m | Model | Multi-scale CNN prediction |
| 30 m | Model | Multi-scale CNN prediction |
| 50 m | Model | Multi-scale CNN prediction |
| 75 m | Model | Multi-scale CNN prediction |
| 100 m | Model | Multi-scale CNN prediction |
| 125 m | Model | Multi-scale CNN prediction |
| 150 m | Model | Multi-scale CNN prediction |
| 200 m | Model | Multi-scale CNN prediction |
| 300 m | Model | Multi-scale CNN prediction |
| 500 m | Model | Multi-scale CNN prediction |
| 700 m | Model | Multi-scale CNN prediction |
| 1000 m | Model | Multi-scale CNN prediction |

**Note:** 0m and 5m are **derived values**, not independently model-predicted:
- **T_0m = SST** (the real sea surface temperature used as model input)
- **T_5m = (SST + T_10m) / 2** (linear interpolation between 0m and 10m)

This approach provides complete 0–1000m display while maintaining transparency about which values are model-predicted vs. derived.

## Real Data Sources

| Variable | Source | Resolution | Status |
|----------|--------|------------|--------|
| **SST** | NOAA OISST v2.1 | 0.25° daily | ✅ Used (2,953/2,992 profiles) |
| **SSH** | NESDIS Satellite Altimetry | ~0.25° daily | ✅ Used (2,992/2,992 profiles) |
| **u10/v10 (Wind)** | ERA5 Reanalysis (via Open-Meteo) | Hourly | ✅ Used (2,982/2,992 profiles) |
| **SSS** | SMOS L3 SSS | 0.25° 3-day | ❌ Blocked (HTTP 403 from ERDDAP) |
| **Currents** | HYCOM GOFS / OSCAR | — | ❌ Blocked (ERDDAP timeout, Open-Meteo only 3% coverage) |
| **Ground Truth** | Argo Float Profiles | In-situ | ✅ Used (2,992 profiles) |
| **GLORYS / Reanalysis** | Copernicus Marine GLORYS12 | — | 🔄 In progress (301/2,992 profiles cached) |

## Dataset

- **2,992 Argo profiles** from 689 floats in the Indian Ocean (30°E–120°E, 30°S–30°N)
- **2,883 complete samples** with all 6 surface features and 13 depth targets
- **Date range:** 2019-01-01 to 2024-12-31
- **13 model-predicted depth levels** from 10 m to 1000 m
- **2 derived depth levels** (0m, 5m) added via SST-anchoring
- **Train/Test split:** 80/20 random (2,306 train / 577 test)

### Data Pipeline
1. Argo profiles downloaded from NOAA ERDDAP (12.1M raw records → 2,992 QC'd profiles)
2. SST extracted from cached NOAA OISST quarterly NetCDF files (nearest-neighbor)
3. SSH extracted from cached NESDIS quarterly NetCDF files (nearest-neighbor)
4. Wind components obtained from Open-Meteo ERA5 reanalysis archive API
5. Quality control: profiles must reach ≥1000 m depth, contain valid temperature observations

## Model Architecture

**Multi-scale 1D CNN with residual connections and self-attention**

- **Input:** 6 features (latitude, longitude, SST, SSH, u10, v10)
- **Output:** 13 temperature values at defined depth levels (10–1000 m)
- **Derived output:** 2 additional levels (0m, 5m) via SST-anchoring
- **Total display:** 15 depth levels (0–1000 m)
- **Parameters:** 300,808
- **Features:**
  - Multi-scale convolutions (kernels 3, 5, 7) for different spatial patterns
  - Self-attention layer for depth-aware processing
  - Residual skip connections for stable training
  - MC Dropout for uncertainty estimation

## Test Set Evaluation (verified, no retraining)

**Model-predicted levels (10–1000 m):**

| Metric | Value |
|--------|-------|
| **Overall RMSE** | **1.0641°C** |
| **Overall MAE** | **0.7143°C** |
| **Overall R²** | **0.9797** |
| **Test Samples** | 577 |

### RMSE by Depth (Model-Predicted)

| Depth | RMSE (°C) | MAE (°C) | R² | N |
|-------|-----------|----------|------|-----|
| 10 m | 0.736 | 0.491 | 0.926 | 577 |
| 20 m | 0.721 | 0.484 | 0.928 | 577 |
| 30 m | 0.946 | 0.644 | 0.875 | 577 |
| 50 m | 1.351 | 0.964 | 0.764 | 577 |
| 75 m | 1.577 | 1.168 | 0.697 | 577 |
| 100 m | 1.559 | 1.131 | 0.670 | 577 |
| 125 m | 1.380 | 1.028 | 0.693 | 577 |
| 150 m | 1.212 | 0.907 | 0.749 | 577 |
| 200 m | 0.985 | 0.743 | 0.823 | 577 |
| 300 m | 0.787 | 0.583 | 0.811 | 577 |
| 500 m | 0.600 | 0.387 | 0.850 | 577 |
| 700 m | 0.619 | 0.402 | 0.861 | 577 |
| 1000 m | 0.562 | 0.355 | 0.870 | 577 |

**Derived levels (0–5 m):**
- 0m = SST (measured, no RMSE — direct pass-through)
- 5m = interpolation between SST and 10m prediction (no independent validation)

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI backend
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Start the Streamlit dashboard (in a separate terminal)
streamlit run streamlit_app.py
```

- Backend: http://localhost:8000
- Streamlit dashboard: http://localhost:8501

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Predict temperature profile (15 levels, 0-1000m) |
| `/api/metrics` | GET | Get model evaluation metrics |
| `/api/status` | GET | Get data pipeline and model status |
| `/api/features` | GET | Get estimated surface features for a location |
| `/api/depth-levels` | GET | Get depth levels used by the model |
| `/` | GET | Interactive INCOIS map dashboard |
| `/metrics` | GET | Metrics dashboard |

## Streamlit Dashboard

A separate Streamlit-based dashboard provides:
- Interactive map for lat/lon selection (default: North Indian Ocean)
- Real-time prediction via the existing `/api/predict` endpoint
- Plotly temperature profile chart with 15 depth levels
- Visual distinction between model-predicted (10-1000m) and derived (0-5m) values
- Nearest Argo float comparison overlay
- Model metrics from the held-out test set

Run with: `streamlit run streamlit_app.py`

## Project Structure

```
oceanembed/
├── streamlit_app.py        # Streamlit dashboard (15 levels)
├── backend/app.py          # FastAPI backend (SST-anchored 0m/5m)
├── config.py               # Central configuration (13 model + 2 derived levels)
├── data/
│   ├── processed/          # Final ML dataset (parquet)
│   ├── raw/                # Raw data (Argo, OISST, SSH, wind caches)
│   └── reports/            # Evaluation reports
├── models/
│   ├── ocean_model.py      # CNN model architecture
│   ├── training.py         # Training pipeline
│   └── checkpoints/        # Saved model weights (real + synthetic)
├── scripts/
│   ├── surface_lookup.py   # Real-time surface data lookup
│   ├── data_ingestion.py   # Argo data pipeline
│   ├── build_real_dataset.py
│   ├── train_real_model.py # Training script
│   └── fetch_wind_batch.py # Wind data fetching
├── experimental/           # Quarantined HYCOM/GLORYS extraction scripts
├── static/
│   ├── index.html          # Main INCOIS map dashboard
│   ├── metrics.html        # Metrics dashboard
│   └── js/app.js           # Frontend JavaScript (15 levels)
└── sample_data/            # Synthetic demo fallback
```

## Known Limitations

- **SSS:** Not included — SMOS data source returned HTTP 403 Forbidden from ERDDAP. Tried 6+ alternate sources; all blocked by auth or format issues.
- **Ocean Currents:** Not included — HYCOM GOFS OPeNDAP accessible but throttled (only 100/2,992 profiles extracted). Open-Meteo provided only 3% coverage. Insufficient for training.
- **GLORYS Reanalysis:** Partially available — Copernicus Marine credentials work, but full extraction would require ~26 sessions (30 min each). Currently at 301/2,992 profiles.
- **0m/5m depths:** Derived values (SST-anchored), not independently model-predicted. Only 31/2,992 Argo profiles have real 0m measurements.
- **Thermocline region (50–200 m):** Higher RMSE (1.2–1.6°C) due to natural variability.
- **Deep ocean (500–1000 m):** Best performance (RMSE 0.56–0.60°C).

## Acknowledgments

- **INCOIS (Indian National Centre for Ocean Information Services)** — primary stakeholder
- **Argo Program** for global ocean profiling data
- **NOAA OISST** for sea surface temperature
- **NESDIS** for satellite altimetry
- **Open-Meteo** for ERA5 reanalysis wind data
- **Smart India Hackathon 2026** for the problem statement
