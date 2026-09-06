# OceanEmbed — Presentation Notes (SIH26066)

## Q1: Why 7 surface inputs instead of just SST/SSH?

**Answer:** The PPT specifies 7 surface inputs: SST, SSS, SSH, Winds (u10, v10), and Currents (u, v). We implemented all 7:
- SST: Real from NOAA OISST (98.7% coverage)
- SSS: Climatological fallback from WOA18 (SMOS ERDDAP returns 403)
- SSH: Real from NESDIS Satellite Altimetry (100%)
- Winds: Real from ERA5 (99.7%)
- Currents: Interpolated from nearest cached values (100%)

All sources are clearly labeled in API responses and code. Climatological/interpolated values are honest approximations, not fabricated data.

---

## Q2: Is this really a spatial CNN? How is it different from a 1D model?

**Answer:** Yes — the architecture is a U-Net encoder-decoder with residual blocks that operates on 0.25° grid patches:
- Input: 7-channel 8×8 patch (2°×2° at 0.25° resolution)
- Encoder: 3 levels of Conv2d + BatchNorm + GELU + ResBlock
- Decoder: Upsampling with skip connections
- Output: 13-depth temperature profile at center pixel

This is fundamentally different from a 1D point-wise model because it learns spatial relationships between neighboring ocean cells, not just individual points.

---

## Q3: Why is the spatial CNN's RMSE (1.57°C) worse than the 1D model (1.03°C)?

**Answer:** Because we trained on synthetic grid patches (point data with small perturbations), not real 0.25° gridded CMEMS data. The spatial CNN needs real spatial context to learn meaningful patterns. With real gridded data from CMEMS (SST, SSS, SSH, winds, currents), performance would improve significantly.

---

## Q4: How do you handle missing data (SSS, currents)?

**Answer:** We use transparent fallback mechanisms:
- **SSS:** WOA18 monthly climatology (World Ocean Atlas) — a scientifically validated climatological product, not fabricated
- **Currents:** Nearest-neighbor interpolation from cached HYCOM data — physically reasonable approximation

Both are clearly labeled as "climatological" or "interpolated" in API responses. The model learns from these values during training, and the API reports which source each feature came from.

---

## Q5: What about the 0m and 5m depth levels?

**Answer:** The model predicts 13 depth levels (10–1000m). We add 2 derived values for full 0–1000m display:
- **T_0m = SST** (direct pass-through from measured input, not model-predicted)
- **T_5m = (SST + T_10m) / 2** (linear interpolation between two real values)

This is scientifically honest: SST physically represents near-surface temperature (~0m), and interpolation between two trusted anchor points is standard practice.

---

## Q6: Can this run on an RTX 3050?

**Answer:** Yes. The spatial CNN has ~1.3M parameters (well within 4-6GB VRAM). Training uses:
- Gradient checkpointing for VRAM savings
- Mixed precision (AMP) training
- Small batch sizes (16-32)
- Patch-based input (8×8 cells, not full grid)

---

## Q7: What's the real-world accuracy?

**Answer:** On the held-out test set (289 Argo profiles never seen during training):
- **Overall RMSE:** 1.03°C (1D model) / 1.57°C (spatial CNN on synthetic data)
- **Best depths:** 500–1000m (RMSE 0.42–0.54°C)
- **Challenging depths:** 50–100m (thermocline region, RMSE 1.35–1.78°C)

The thermocline is inherently variable, so higher RMSE there is expected even for physics-based models.

---

## Q8: How does this compare to existing methods?

**Answer:** Our approach differs from traditional methods:
- **Traditional:** Optimal interpolation, regression on individual profiles
- **Our approach:** Deep learning on multi-source satellite data with spatial context

The CNN architecture can learn complex non-linear relationships between surface observations and subsurface temperature that traditional methods miss. With real gridded data, we expect significant improvement.

---

## Q9: What about GLORYS reanalysis?

**Answer:** We extracted 301 GLORYS profiles (10.1% of total) via batch-by-date download from Copernicus Marine. A proof-of-concept model trained on GLORYS labels achieved RMSE 1.87°C (R² 0.43), which is worse than the Argo-trained model because:
- Only 301 samples (vs 2,883 for Argo)
- GLORYS is a model reanalysis, not direct observations

Full GLORYS extraction would require ~26 sessions (13 hours) at the current download rate.

---

## Q10: What's the deployment plan?

**Answer:** The system is deployed as:
1. **FastAPI backend** serving predictions via REST API
2. **Interactive map dashboard** (Leaflet.js) for oceanographers
3. **Streamlit dashboard** for quick prototyping

Both dashboards accept the same 7 surface inputs and display the full 15-depth profile with uncertainty bands.
