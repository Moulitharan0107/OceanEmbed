# OceanEmbed — Satellite Embedding-Based Deep Learning Framework

**SIH Problem Statement SIH26066** (Ministry of Earth Sciences)
*"Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations"*

---

## Overview

OceanEmbed predicts the **vertical temperature profile of the ocean** (0–2000 m depth) at any location in the Indian Ocean, using only surface satellite observations as input. The core idea: satellite surface data is dense and global, but subsurface measurements (Argo floats) are sparse. This model bridges that gap.

## Real Data Sources

| Variable | Source | Resolution | Status |
|----------|--------|------------|--------|
| **SST** | NOAA OISST v2.1 | 0.25° daily | ✅ Used |
| **SSH** | NESDIS Satellite Altimetry | ~0.25° daily | ✅ Used |
| **u10/v10 (Wind)** | ERA5 Reanalysis (via Open-Meteo) | Hourly | ✅ Used |
| **SSS** | SMOS L3 SSS | 0.25° 3-day | ❌ Unavailable (HTTP 403) |
| **Ground Truth** | Argo Float Profiles | In-situ | ✅ Used |

**Note:** SSS was intentionally dropped because the SMOS data source returned HTTP 403 Forbidden from ERDDAP. The model uses 6 input features: latitude, longitude, SST, SSH, u10, v10.

## Dataset

- **2,992 Argo profiles** from 689 floats in the Indian Ocean (30°E–120°E, 30°S–30°N)
- **2,944 complete samples** with all surface features
- **Date range:** 2019-01-01 to 2024-12-31
- **33 depth levels** from 0 m to 2000 m
- **Train/Val/Test split:** Float-aware (70%/15%/15%) to prevent data leakage

### Data Pipeline
1. Argo profiles downloaded from NOAA ERDDAP (12.1M raw records → 2,992 QC'd profiles)
2. SST extracted from cached NOAA OISST quarterly NetCDF files (nearest-neighbor)
3. SSH extracted from cached NESDIS quarterly NetCDF files (nearest-neighbor)
4. Wind components obtained from Open-Meteo ERA5 reanalysis archive API
5. Quality control: profiles must reach ≥1000 m depth, contain valid temperature observations at all target depths

## Model Architecture

**Multi-scale 1D CNN with residual connections and self-attention**

- **Input:** 6 features (latitude, longitude, SST, SSH, u10, v10)
- **Output:** 33 temperature values at defined depth levels
- **Parameters:** 302,248
- **Features:**
  - Multi-scale convolutions (kernels 3, 5, 7) for different spatial patterns
  - Self-attention layer for depth-aware processing
  - Residual skip connections for stable training
  - MC Dropout for uncertainty estimation

## Test Set Evaluation

| Metric | Value |
|--------|-------|
| **Overall RMSE** | 0.926°C |
| **Overall MAE** | 0.548°C |
| **Overall R²** | 0.9878 |
| **Shallow RMSE (0–200 m)** | 1.429°C |
| **Deep RMSE (500–2000 m)** | 0.342°C |

### RMSE by Depth

| Depth | RMSE (°C) | R² | N |
|-------|-----------|-----|---|
| 10 m | 0.919 | 0.898 | 427 |
| 50 m | 1.408 | 0.778 | 446 |
| 100 m | 1.752 | 0.603 | 446 |
| 200 m | 1.109 | 0.786 | 446 |
| 500 m | 0.438 | 0.864 | 446 |
| 1000 m | 0.350 | 0.910 | 446 |
| 1500 m | 0.242 | 0.861 | 423 |
| 2000 m | 0.262 | -0.243 | 124 |

### Feature Importance (Gradient-Based)

| Feature | Importance |
|---------|------------|
| Latitude | 44.6% |
| SSH | 16.0% |
| Longitude | 14.2% |
| SST | 11.0% |
| Wind (u10+v10) | 14.2% |

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the real-data model (if not already trained)
python scripts/train_real_model.py

# Start the server
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Predict temperature profile at location/date |
| `/api/metrics` | GET | Get model evaluation metrics |
| `/api/status` | GET | Get data pipeline and model status |
| `/api/features` | GET | Get estimated surface features for a location |
| `/api/depth-levels` | GET | Get depth levels used by the model |
| `/` | GET | Interactive map frontend |
| `/metrics` | GET | Metrics dashboard |

## Frontend Features

- **Interactive Map:** Click any point in the Indian Ocean to predict subsurface temperature
- **Temperature Profile Chart:** Plotly visualization of predicted temperature vs depth
- **Uncertainty Estimation:** MC Dropout provides confidence bands
- **Feature Explainability:** Gradient-based feature importance visualization
- **Model Mode Indicator:** Shows "REAL MODEL" or "CACHED SAMPLE MODEL"
- **Metrics Dashboard:** Real evaluation metrics from held-out test set

## Fallback Behavior

The existing 500-sample synthetic demo dataset remains available as a fallback:
- If the real model checkpoint is missing, falls back to the synthetic model
- Status badge clearly shows "CACHED SAMPLE MODEL" in fallback mode
- The UI distinguishes between real and cached modes

## Limitations

- **Region:** Indian Ocean only (30°E–120°E, 30°S–30°N)
- **Period:** 2019–2024 (6 years)
- **Depth range:** 0–2000 m (33 levels)
- **SSS:** Not included (SMOS data unavailable)
- **No real-time satellite access:** Surface features are estimated from climatological models unless explicitly provided
- **Shallow ocean (50–200 m):** Higher RMSE due to thermocline variability
- **Deep ocean (2000 m):** Limited data (only 124 profiles reach this depth)

## Project Structure

```
oceanembed/
├── backend/app.py          # FastAPI backend
├── config.py               # Central configuration
├── data/
│   ├── processed/          # Final ML dataset
│   ├── raw/                # Raw data (Argo, OISST, SSH, wind caches)
│   └── reports/            # Evaluation reports
├── models/
│   ├── ocean_model.py      # CNN model architecture
│   ├── training.py         # Training pipeline
│   └── checkpoints/        # Saved model weights
├── scripts/
│   ├── train_real_model.py # Real-data training script
│   └── fetch_wind_batch.py # Wind data fetching
├── static/
│   ├── index.html          # Main frontend
│   ├── metrics.html        # Metrics dashboard
│   ├── css/styles.css      # Styles
│   └── js/app.js           # Frontend JavaScript
└── sample_data/            # Synthetic demo fallback
```

## Acknowledgments

- **Argo Program** for global ocean profiling data
- **NOAA OISST** for sea surface temperature
- **NESDIS** for satellite altimetry
- **Open-Meteo** for ERA5 reanalysis wind data
- **Smart India Hackathon 2026** for the problem statement
