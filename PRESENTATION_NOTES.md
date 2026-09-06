# OceanEmbed — Presentation Notes (SIH26066)

## Q1: Why 7 surface inputs instead of just SST/SSH?

**Answer:** The PPT specifies 7 surface inputs: SST, SSS, SSH, Winds (u10, v10), and Currents (u, v). We implemented all 7:
- SST: Real from NOAA OISST (98.7% coverage)
- SSS: Real from CMEMS GLORYS12V1 where available, WOA18 climatology fallback
- SSH: Real from NESDIS Satellite Altimetry (100%)
- Winds: Real from ERA5 (99.7%)
- Currents: Real from CMEMS GLORYS12V1 where available, nearest-neighbor interpolation fallback

CMEMS integration is active: 284/2,992 profiles (~9.5%) have real CMEMS SSS and currents (302 total with GLORYS temperature profiles). Achieved via stratified representative sampling within hackathon time constraints. Each profile query takes ~60 seconds via the `copernicusmarine` Python toolbox. All sources are clearly labeled in API responses and code (sss_source, currents_source columns).

---

## Q2: Is this really a spatial CNN? How is it different from a 1D model?

**Answer:** Yes — the architecture is a U-Net encoder-decoder with residual blocks that operates on 0.25° grid patches:
- Input: 7-channel 8×8 patch (2°×2° at 0.25° resolution)
- Encoder: 3 levels of Conv2d + BatchNorm + GELU + ResBlock
- Decoder: Upsampling with skip connections
- Output: 13-depth temperature profile at center pixel

This is fundamentally different from a 1D point-wise model because it learns spatial relationships between neighboring ocean cells, not just individual points.

---

## Q3: Why is the spatial CNN's RMSE (1.51°C) worse than the 1D model (1.03°C)?

**Answer:** We trained the spatial CNN on real 0.25° grid patches, but only ~10% of profiles had real CMEMS SSS/currents data. The remaining ~90% used fallback values (climatological SSS, interpolated currents). The spatial architecture needs substantially more real gridded data to learn meaningful spatial patterns. With higher CMEMS coverage (30%+), we expect the spatial CNN to outperform the 1D model.

---

## Q4: How do you handle missing data (SSS, currents)?

**Answer:** We use a tiered approach with transparent source tracking:
- **SSS:** Real from CMEMS GLORYS12V1 where available (284 profiles, ~9.5%). WOA18 climatology as fallback for remaining profiles.
- **Currents:** Real from CMEMS GLORYS12V1 where available. Nearest-neighbor interpolation as fallback for remaining profiles.

Every row in the dataset has `sss_source` and `currents_source` columns indicating whether the value is "cmems" (real) or "climatology_fallback"/"interpolated_fallback" (approximation). The API reports which source each feature came from. CMEMS extraction was done via stratified representative sampling within hackathon time constraints (~60 seconds per profile).

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
- **Overall RMSE:** 1.09°C (1D production, 6 features) / 1.29°C (7-feature with SSS, 15 epochs) / 1.51°C (spatial CNN, real patches, ~10% CMEMS coverage)
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
