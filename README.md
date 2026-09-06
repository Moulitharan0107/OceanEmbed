# OceanEmbed — Satellite Embedding-Based Deep Learning Framework

**SIH Problem Statement SIH26066** (Ministry of Earth Sciences)
*"Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations"*

---

## Overview

OceanEmbed predicts the **vertical temperature profile of the ocean** (0–1000 m depth, **15 levels**) at any location in the **North Indian Ocean**, using surface satellite observations as input.

Built for **INCOIS (Indian National Centre for Ocean Information Services)**, supporting oceanographers and disaster management teams with rapid subsurface temperature reconstruction.

---

## Key Features

| Feature | Specification |
|---------|---------------|
| **Surface Inputs** | 6 features: latitude, longitude, SST, SSH, u10, v10 |
| **Output** | **15-depth temperature profile** (0–1000 m) |
| **Domain** | North Indian Ocean (0–25°N, 40–100°E) |
| **Model** | Multi-scale 1D CNN with residual connections |

---

## Input Features

| # | Variable | Description | Source |
|---|----------|-------------|--------|
| 1 | **latitude** | Latitude (°N) | Argo profiles |
| 2 | **longitude** | Longitude (°E) | Argo profiles |
| 3 | **SST** | Sea Surface Temperature (°C) | NOAA OISST v2.1 |
| 4 | **SSH** | Sea Surface Height Anomaly (m) | NESDIS Satellite Altimetry |
| 5 | **u10** | Zonal Wind Speed at 10m (m/s) | ERA5 Reanalysis |
| 6 | **v10** | Meridional Wind Speed at 10m (m/s) | ERA5 Reanalysis |

**Note:** SSS and ocean currents were attempted but blocked by data access issues (see Known Limitations).

---

## Output: 15-Depth Temperature Profile

| Depth | Type | Description |
|-------|------|-------------|
| **0 m** | **Derived** | SST (measured) — direct pass-through from input |
| **5 m** | **Derived** | Linear interpolation between SST and 10m model prediction |
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

**Note:** 0m and 5m are **derived values**, not model-predicted:
- **T_0m = SST** (the real sea surface temperature used as model input)
- **T_5m = (SST + T_10m) / 2** (linear interpolation)

---

## Model Architecture

**Multi-scale 1D CNN with residual connections**

```
Input: (batch, 6) — 6 surface features
    ↓
Feature Embedding (Linear → BN → ReLU)
    ↓
Multi-scale Conv Blocks (kernels 3, 5, 7)
    ↓
Self-Attention Layer
    ↓
Output Head (Linear → BN → ReLU → Linear)
    ↓
Output: (batch, 13) — 13 depth temperatures
```

- **Parameters:** 300,808
- **Uncertainty:** MC Dropout for prediction confidence

---

## Test Set Evaluation

| Metric | Value |
|--------|-------|
| **Overall RMSE** | **1.03°C** |
| **Overall MAE** | **0.72°C** |
| **Overall R²** | **0.81** |
| **Test Samples** | 289 |

### RMSE by Depth

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

- Backend: http://localhost:8000
- Streamlit dashboard: http://localhost:8501

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Predict 15-depth temperature profile |
| `/api/metrics` | GET | Get model evaluation metrics |
| `/api/status` | GET | Get model and data status |
| `/api/features` | GET | Get 6 surface features for a location |
| `/api/depth-levels` | GET | Get 15 depth levels |
| `/` | GET | Interactive INCOIS map dashboard |

---

## Dashboard Features

- **Interactive map** for lat/lon selection (North Indian Ocean)
- **6 surface input controls** (SST, SSH, u10, v10)
- **15-depth temperature profile** visualization with Plotly
- **Uncertainty bands** from MC Dropout
- **Nearest Argo float comparison** for validation
- **Model metrics** from held-out test set

---

## Project Structure

```
oceanembed/
├── streamlit_app.py              # Streamlit dashboard
├── config.py                     # Central configuration
├── backend/
│   └── app.py                    # FastAPI backend
├── models/
│   ├── ocean_model.py            # CNN model architecture
│   ├── training.py               # Training pipeline
│   └── checkpoints/              # Saved model weights
│       ├── ocean_embed_real_best.pt      # Main model
│       └── BACKUP_ocean_embed_real_best_13level.pt  # Backup
├── scripts/
│   ├── surface_lookup.py         # Real-time surface data lookup
│   ├── data_ingestion.py         # Argo data pipeline
│   └── glorys_batch_extract.py   # GLORYS extraction
├── data/
│   ├── processed/                # Argo profiles
│   └── reports/                  # Evaluation metrics
├── experimental/                 # Quarantined scripts
├── backups/                      # Safety backups
├── static/                       # Frontend assets
└── PRESENTATION_NOTES.md         # Judge Q&A reference
```

---

## Data Sources

| Variable | Source | Status |
|----------|--------|--------|
| **SST** | NOAA OISST v2.1 | ✅ Used (2,953/2,992 profiles) |
| **SSH** | NESDIS Satellite Altimetry | ✅ Used (2,992/2,992 profiles) |
| **Wind** | ERA5 Reanalysis | ✅ Used (2,982/2,992 profiles) |
| **SSS** | SMOS L3 SSS | ❌ Blocked (HTTP 403 from ERDDAP) |
| **Currents** | HYCOM GOFS | ❌ Blocked (3.3% coverage) |
| **GLORYS** | Copernicus Marine | 🔄 Partial (301/2,992 profiles) |

---

## Known Limitations

1. **SSS:** Not included — all 6 attempted sources blocked (403, timeout, format issues)
2. **Ocean Currents:** Not included — only 3.3% coverage (100/2,992 profiles)
3. **GLORYS:** Partially available — 301 profiles cached, full extraction requires ~26 sessions
4. **Spatial relationships:** Not modeled (1D point-wise, not spatial CNN)
5. **Thermocline accuracy:** Higher RMSE at 75-200m due to natural variability

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
- **Smart India Hackathon 2026** for the problem statement
