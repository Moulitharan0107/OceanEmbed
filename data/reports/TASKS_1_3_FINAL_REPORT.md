# OceanEmbed Tasks 1-3: Final Report

## Task 1 - Verify 0m and 5m Depth Levels

### Root Cause Found
- **T_0m**: Only **31 of 2,992 Argo profiles (1.0%)** have real measurements at 0m depth
- **T_5m**: Does not even exist as an original Argo column - was interpolated from 0m and 10m
- The previous 15-level model trained on these unreliable targets, producing **RMSE = 17.0C** at 0m and 5m (predictions of 12C when SST is 27C)

### Per-Depth Metrics (15-level model, BEFORE fix)

| Depth | N   | RMSE   | R-squared |
|-------|-----|--------|-----------|
| **0m** | 442 | **17.037** | **-35.90** |
| **5m** | 442 | **17.763** | **-39.11** |
| 10m   | 442 | 1.097  | 0.847     |
| 20m   | 442 | 1.099  | 0.847     |
| 50m   | 442 | 1.402  | 0.756     |
| 100m  | 442 | 1.635  | 0.663     |
| 500m  | 442 | 0.489  | 0.892     |
| 1000m | 442 | 0.418  | 0.917     |

### Decision
0m and 5m are **unreliable for demo purposes**. The model cannot learn a meaningful mapping from SST/SSH/wind to 0m temperature when 99% of training targets at that depth were interpolated from distant measurements.

### Fix Applied
Retrained with **13 reliable depth levels only** (10m-1000m), dropping 0m and 5m.

---

## Task 1 Result (13-level model, AFTER fix)

### Per-Depth Metrics

| Depth | RMSE (C) | MAE (C) | R-squared |
|-------|----------|---------|-----------|
| 10m   | 0.663    | 0.469   | 0.9347    |
| 20m   | 0.746    | 0.491   | 0.9186    |
| 30m   | 0.936    | 0.646   | 0.8760    |
| 50m   | 1.400    | 1.005   | 0.7542    |
| 75m   | 1.619    | 1.232   | 0.6811    |
| 100m  | 1.516    | 1.113   | 0.6802    |
| 125m  | 1.367    | 1.066   | 0.6980    |
| 150m  | 1.266    | 0.974   | 0.7258    |
| 200m  | 1.057    | 0.798   | 0.7943    |
| 300m  | 0.868    | 0.625   | 0.7726    |
| 500m  | 0.731    | 0.426   | 0.7777    |
| 700m  | 0.682    | 0.419   | 0.8184    |
| 1000m | 0.636    | 0.380   | 0.8231    |

### Overall Metrics
- **RMSE: 1.092C**
- **MAE: 0.742C**
- **R-squared: 0.9785**
- **Test samples: 443**
- **Training: 100 epochs, 314s, early stopping at patience 15**

### Predictions at 5 Demo Locations

| Location | SST Input | 10m Pred | Deep (1000m) Pred |
|----------|-----------|----------|-------------------|
| Bay of Bengal (15N, 80E) | 26.8C | 27.6C | 7.3C |
| Arabian Sea (10N, 55E) | 27.4C | 27.6C | 8.7C |
| South IO (-10S, 95E) | 28.6C | 28.8C | 5.6C |
| Central IO (5N, 65E) | 29.7C | 29.7C | 7.3C |
| N. Bay of Bengal (20N, 90E) | 30.1C | 29.4C | 6.6C |

All 10m predictions are within 1C of the SST input - physically consistent.

---

## Task 2 - Ocean Surface Currents

### Sources Tried
1. **OSCAR Ocean Surface Currents via NOAA ERDDAP** - consistently timed out (multiple attempts)
2. **Open-Meteo Marine API** - returned ocean current velocity, but coverage was very sparse

### Coverage Assessment
- **current_u cache**: 828 / 2,992 profiles (27.7%)
- **current_v cache**: 828 / 2,992 profiles (27.7%)

### Decision: BLOCKED
- 27.7% coverage is too low for training. Adding currents as input would reduce the training set from 2,944 to ~800 samples, losing critical data.
- The 27.7% that have data are biased toward coastal/well-instrumented areas, creating a selection bias.
- **Recommendation**: Currents are not used in the current model. Future work could use a full gridded current product with complete spatial coverage.

---

## Task 3 - Retry SSS (Sea Surface Salinity)

### Sources Tried
1. **SMOS via coastwatch ERDDAP** (original attempt) - HTTP 403 Forbidden
2. **PODAAC ERDDAP for SMAP L3 SSS** - connection failed (DNS resolution error)
3. **coastwatch ERDDAP SSS search** - timed out
4. **PODAAC direct OPeNDAP** - not an ERDDAP grid, incompatible format
5. **Open-Meteo Marine API** - no salinity variable available (only ocean currents, waves)

### Decision: BLOCKED
No programmatically accessible SSS source is available in this environment. All ERDDAP servers either return 403, time out, or do not host a gridded SSS product that can be queried via point extraction.

**SSS remains excluded from the model.**

---

## Summary

| Task | Status | Finding |
|------|--------|---------|
| Task 1 (0m/5m) | COMPLETED | Dropped 0m/5m (1% real data). Retrained 13-level model: RMSE=1.092C, R2=0.9785 |
| Task 2 (Currents) | BLOCKED | 27.7% coverage too low. No full-coverage current source available. |
| Task 3 (SSS) | BLOCKED | No working SSS endpoint found after 5 different sources tried. |

### Final Model
- **Architecture**: Multi-scale 1D CNN with residual connections and self-attention (300,808 params)
- **Features**: latitude, longitude, SST, SSH, u10, v10 (6 features)
- **Depth levels**: 13 (10m, 20m, 30m, 50m, 75m, 100m, 125m, 150m, 200m, 300m, 500m, 700m, 1000m)
- **Training data**: 2,944 complete real Argo profiles (Indian Ocean, 2019-2024)
- **Test RMSE**: 1.092C
- **Test R-squared**: 0.9785
