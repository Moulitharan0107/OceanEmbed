"""
OceanEmbed FastAPI Backend
Serves the prediction API and static frontend files.
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models.ocean_model import OceanEmbedModel
from models.training import OceanEmbedTrainer
from scripts.data_ingestion import get_data_status, generate_synthetic_dataset, _load_static_dataset
from scripts.surface_lookup import lookup_real_features, find_nearest_argo, init_surface_lookup, is_ocean

# ============================================================
#  App Initialization
# ============================================================

app = FastAPI(
    title="OceanEmbed API",
    description="Satellite Embedding-Based Deep Learning Framework for Subsurface Ocean Temperature Reconstruction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
trainer: Optional[OceanEmbedTrainer] = None
data_status: dict = {}
metrics: dict = {}


@app.on_event("startup")
async def startup():
    """Load model and data on startup."""
    global trainer, data_status, metrics
    
    print("[STARTUP] Initializing OceanEmbed...")
    
    # Ensure directories exist
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    
    # Try to load real model first, then fall back to synthetic
    real_model_path = os.path.join(config.CHECKPOINT_DIR, "ocean_embed_real_best.pt")
    real_metrics_path = "data/reports/real_model_metrics.json"
    synth_model_path = os.path.join(config.CHECKPOINT_DIR, "ocean_embed_best.pt")
    synth_metrics_path = os.path.join(config.SAMPLE_DIR, "metrics.json")
    
    # Try real model first
    if os.path.exists(real_model_path):
        try:
            trainer = OceanEmbedTrainer.load(real_model_path)
            if os.path.exists(real_metrics_path):
                with open(real_metrics_path) as f:
                    metrics = json.load(f)
            data_status = {
                "mode": "real",
                "message": "Running on REAL trained model (Argo + satellite data)",
                "n_profiles": 2992,
                "combined_samples": metrics.get("n_test_samples", 0),
                "surface_inputs": ["sst", "sss", "ssh", "u10", "v10", "current_u", "current_v"],
                "date_range": "2019-2024",
                "region": "North Indian Ocean (5-30N, 45-105E) — Official SIH26066 domain",
                "sss_source": "WOA18 climatology (SMOS 403)",
                "currents_source": "nearest-neighbor interpolated (HYCOM limited)",
            }
            print("[STARTUP] Loaded REAL trained model")
        except Exception as e:
            print(f"[STARTUP] Failed to load real model: {e}")
            trainer = None
    else:
        trainer = None
    
    # Fall back to synthetic model
    if trainer is None and os.path.exists(synth_model_path):
        try:
            trainer = OceanEmbedTrainer.load(synth_model_path)
            if os.path.exists(synth_metrics_path):
                with open(synth_metrics_path) as f:
                    metrics = json.load(f)
            data_status = {
                "mode": "cached",
                "message": "Running on cached sample data (synthetic profiles for Indian Ocean)",
                "combined_samples": metrics.get("n_test_samples", 0),
            }
            print("[STARTUP] Loaded cached synthetic model")
        except Exception as e:
            print(f"[STARTUP] Failed to load model: {e}")
            trainer = None
    
    # If no model available, train on synthetic
    if trainer is None:
        print("[STARTUP] No model found. Training on synthetic data...")
        try:
            X, y, data_status = _load_static_dataset()
            trainer = OceanEmbedTrainer()
            train_loader, val_loader, test_loader = trainer.prepare_data(X, y)
            history = trainer.train(train_loader, val_loader, num_epochs=50, patience=10)
            metrics = trainer.evaluate(test_loader)
            trainer.save()
            print("[STARTUP] Synthetic model trained and saved")
        except Exception as e:
            print(f"[STARTUP] Failed to train model: {e}")
            import traceback
            traceback.print_exc()
    
    # Initialize surface data lookup
    print("[STARTUP] Loading surface data lookup...")
    init_surface_lookup()
    print(f"[STARTUP] Data mode: {data_status.get('mode', 'unknown')}")


# ============================================================
#  Pydantic Models
# ============================================================

class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=5, le=30, description="Latitude (Official SIH26066 domain: 5°N to 30°N)")
    longitude: float = Field(..., ge=45, le=105, description="Longitude (Official SIH26066 domain: 45°E to 105°E)")
    date: str = Field("2022-06-15", description="Date (YYYY-MM-DD)")
    sst: Optional[float] = Field(None, description="Sea Surface Temperature (°C). Auto-estimated if not provided.")
    ssh: Optional[float] = Field(None, description="Sea Surface Height Anomaly (m)")
    sss: Optional[float] = Field(None, description="Sea Surface Salinity (PSU)")
    u10: Optional[float] = Field(None, description="Zonal wind speed (m/s)")
    v10: Optional[float] = Field(None, description="Meridional wind speed (m/s)")
    current_u: Optional[float] = Field(None, description="Ocean current u-component (m/s)")
    current_v: Optional[float] = Field(None, description="Ocean current v-component (m/s)")


class PredictionResponse(BaseModel):
    latitude: float
    longitude: float
    date: str
    predicted_profile: List[float]
    uncertainty: List[float]
    depth_levels: List[int]
    features_used: dict
    data_mode: str
    confidence_score: float
    nearest_argo: Optional[dict] = None


class MetricsResponse(BaseModel):
    overall_rmse: float
    overall_r2: float
    n_test_samples: int
    depth_rmse: List[float]
    depth_r2: List[float]
    depth_levels: List[int]
    feature_importance: dict
    timestamp: str
    model_params: int


class StatusResponse(BaseModel):
    data_mode: str
    message: str
    model_loaded: bool
    n_profiles: int
    n_samples: int


class FeaturesResponse(BaseModel):
    latitude: float
    longitude: float
    sst: float
    sss: float
    ssh: float
    u10: float
    v10: float
    current_u: float
    current_v: float
    date: str
    data_mode: str
    sources: dict


# ============================================================
#  Helper Functions
# ============================================================

def get_features(lat: float, lon: float, date_str: str) -> Tuple[np.ndarray, dict]:
    """
    Get surface features using real cached data where available.
    Falls back to climatological estimates only when real data is unavailable.
    
    Returns:
        features: numpy array [lat, lon, sst, ssh, u10, v10]
        metadata: dict with source info and data mode
    """
    result = lookup_real_features(lat, lon, date_str)
    f = result["features"]
    # Build full 7-feature array for API response
    all_features = {
        "latitude": f["latitude"], "longitude": f["longitude"],
        "sst": f["sst"], "sss": f.get("sss", 35.0), "ssh": f["ssh"],
        "u10": f["u10"], "v10": f["v10"],
        "current_u": f.get("current_u", 0.0), "current_v": f.get("current_v", 0.0),
    }
    result["all_features"] = all_features
    # Model-specific feature array (the 1D model was trained on 6 features)
    features = np.array([
        f["latitude"], f["longitude"],
        f["sst"], f["ssh"], f["u10"], f["v10"]
    ], dtype=np.float32)
    return features, result


# ============================================================
#  API Endpoints
# ============================================================

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current data pipeline and model status."""
    return StatusResponse(
        data_mode=data_status.get("mode", "unknown"),
        message=data_status.get("message", "Not initialized"),
        model_loaded=trainer is not None,
        n_profiles=data_status.get("argo_profiles", 0),
        n_samples=data_status.get("combined_samples", 0),
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """
    Predict subsurface temperature profile at a given location and time.
    Uses real cached surface data where available, climatological fallback otherwise.
    """
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait for initialization.")

    # --- Date range validation ---
    from datetime import datetime, timedelta
    TRAINING_END_DATE = datetime(2024, 12, 31)
    MAX_PREDICTION_DATE = TRAINING_END_DATE + timedelta(days=183)  # ~6 months buffer
    try:
        req_date = datetime.strptime(req.date, "%Y-%m-%d")
    except (ValueError, TypeError):
        req_date = datetime.now()
    if req_date > MAX_PREDICTION_DATE:
        raise HTTPException(
            status_code=400,
            detail=f"Uncertainty will be high - this date ({req.date}) is beyond the model's validated time range (2019-2024, plus 6 months buffer). Please select a date before {MAX_PREDICTION_DATE.strftime('%Y-%m-%d')}.",
        )

    # --- Land / Ocean gate (in addition to lat/lon bounds already validated by Pydantic) ---
    if not is_ocean(req.latitude, req.longitude):
        raise HTTPException(
            status_code=400,
            detail="Selected point is on land. Please select an ocean point to get a subsurface temperature prediction.",
        )

    # Get real features with source tracking
    features, feature_meta = get_features(req.latitude, req.longitude, req.date)
    
    # Allow user overrides
    if req.sst is not None:
        features[2] = req.sst
        feature_meta["sources"]["sst"] = "user provided"
    if req.ssh is not None:
        features[3] = req.ssh
        feature_meta["sources"]["ssh"] = "user provided"
    if req.u10 is not None:
        features[4] = req.u10
        feature_meta["sources"]["u10"] = "user provided"
    if req.v10 is not None:
        features[5] = req.v10
        feature_meta["sources"]["v10"] = "user provided"
    if req.current_u is not None:
        features[6] = req.current_u
        feature_meta["sources"]["current_u"] = "user provided"
    if req.current_v is not None:
        features[7] = req.current_v
        feature_meta["sources"]["current_v"] = "user provided"
    
    # Normalize using saved training scalers
    features_norm = trainer.scaler_X.transform(features.reshape(1, -1)).astype(np.float32)
    features_tensor = torch.tensor(features_norm).to(trainer.device)
    
    # Predict with uncertainty (MC Dropout)
    trainer.model.eval()
    mean_pred, std_pred = trainer.model.predict_with_uncertainty(features_tensor, n_forward_passes=20)
    
    # Denormalize predictions
    profile = trainer.scaler_y.inverse_transform(mean_pred.cpu().numpy())[0]
    uncertainty = (std_pred.cpu().numpy() * trainer.scaler_y.scale_)[0]
    
    # Clamp to physically reasonable range
    profile = np.clip(profile, -2, 35)
    uncertainty = np.abs(uncertainty)
    
    # Add derived 0m and 5m values (SST-anchored)
    # T_0m = SST (direct pass-through, not model-predicted)
    # T_5m = linear interpolation between SST (0m) and model's T_10m
    sst_value = float(features[2])
    t_0m = sst_value
    t_10m = float(profile[0])  # First model output is 10m
    t_5m = (t_0m + t_10m) / 2.0  # Linear interpolation at midpoint

    # Build 15-level profile: [0m, 5m] + model's 13 levels [10m-1000m]
    full_profile = [t_0m, t_5m] + profile.tolist()
    full_uncertainty = [0.0, 0.0] + uncertainty.tolist()  # No uncertainty for derived values
    full_depths = [0, 5] + config.MODEL_DEPTH_LEVELS

    # Confidence score
    temp_range = float(np.max(profile) - np.min(profile)) + 1e-6
    mean_uncert = float(np.mean(uncertainty))
    confidence = max(0, min(1, 1 - mean_uncert / (temp_range * 0.5)))

    # Find nearest Argo profile for comparison
    nearest = find_nearest_argo(req.latitude, req.longitude, req.date)

    return PredictionResponse(
        latitude=req.latitude,
        longitude=req.longitude,
        date=req.date,
        predicted_profile=full_profile,
        uncertainty=full_uncertainty,
        depth_levels=full_depths,
        features_used=feature_meta.get("all_features", {
            "sst": round(float(features[2]), 2),
            "ssh": round(float(features[3]), 4),
            "u10": round(float(features[4]), 2),
            "v10": round(float(features[5]), 2),
            "latitude": round(float(features[0]), 2),
            "longitude": round(float(features[1]), 2),
        }),
        data_mode=feature_meta["mode"],
        confidence_score=round(confidence, 3),
        nearest_argo=nearest,
    )


@app.get("/api/metrics")
async def get_metrics():
    """Get model evaluation metrics from the held-out test set."""
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics available. Train the model first.")
    
    return JSONResponse(content=metrics)


@app.get("/api/features", response_model=FeaturesResponse)
async def get_features_endpoint(
    lat: float = Query(..., ge=5, le=30),
    lon: float = Query(..., ge=45, le=105),
    date: str = Query("2022-06-15"),
):
    """Get surface features for a location with real data lookup."""
    result = lookup_real_features(lat, lon, date)
    f = result["features"]
    
    return FeaturesResponse(
        latitude=lat,
        longitude=lon,
        sst=f["sst"],
        sss=f.get("sss", 35.0),
        ssh=f["ssh"],
        u10=f["u10"],
        v10=f["v10"],
        current_u=f.get("current_u", 0.0),
        current_v=f.get("current_v", 0.0),
        date=date,
        data_mode=result["mode"],
        sources=result["sources"],
    )


@app.get("/api/depth-levels")
async def get_depth_levels():
    """Get the depth levels used by the model."""
    return {"depth_levels": config.DEPTH_LEVELS, "n_levels": len(config.DEPTH_LEVELS)}


@app.get("/api/cyclone-sst")
async def get_cyclone_sst(
    phase: str = Query("before", description="'before' or 'during' the cyclone"),
):
    """
    Return SST grid for Cyclone Fani (May 2019) before/during comparison.
    Before: April 25, 2019 (pre-cyclone warm water)
    During: May 1, 2019 (peak approach, SST cooling from wind mixing)
    """
    import xarray as xr
    import numpy as np
    
    oisst_path = os.path.join(config.DATA_DIR, "raw", "oisst", "sst_2019_Q2.nc")
    if not os.path.exists(oisst_path):
        raise HTTPException(status_code=404, detail="OISST Q2 2019 data not found")
    
    ds = xr.open_dataset(oisst_path)
    
    if phase == "before":
        sst_slice = ds.sst.sel(time="2019-04-25", method="nearest").values[0]
        date_label = "April 25, 2019"
    else:
        sst_slice = ds.sst.sel(time="2019-05-01", method="nearest").values[0]
        date_label = "May 1, 2019"
    
    lats = ds.latitude.values
    lons = ds.longitude.values
    ds.close()
    
    # Subsample to North Indian Ocean for lighter transfer
    lat_mask = (lats >= -5) & (lats <= 25)
    lon_mask = (lons >= 60) & (lons <= 100)
    sst_sub = sst_slice[np.ix_(lat_mask, lon_mask)]
    lats_sub = lats[lat_mask]
    lons_sub = lons[lon_mask]
    
    # Convert NaN to null for JSON
    sst_list = np.where(np.isnan(sst_sub), None, np.round(sst_sub.astype(float), 2)).tolist()
    
    return {
        "cyclone": "Fani",
        "phase": phase,
        "date": date_label,
        "description": f"Cyclone Fani {phase} ({date_label}) — SST from NOAA OISST v2.1",
        "lats": [round(float(x), 2) for x in lats_sub[::4]],  # subsample for transfer
        "lons": [round(float(x), 2) for x in lons_sub[::4]],
        "sst": [row[::4] for row in sst_list[::4]],  # 4x subsample
        "colorbar": {"min": 24, "max": 31, "unit": "°C"},
    }


# Serve static frontend files
app.mount("/static", StaticFiles(directory=os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "static"
)), name="static")


@app.get("/")
async def serve_index():
    """Serve the main frontend page."""
    index_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "static", "index.html"
    )
    return FileResponse(index_path)


@app.get("/metrics")
async def serve_metrics_page():
    """Serve the metrics/admin page."""
    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "static", "metrics.html"
    )
    return FileResponse(metrics_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
