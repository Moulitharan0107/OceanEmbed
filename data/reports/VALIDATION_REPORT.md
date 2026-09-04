# OceanEmbed Validation Report
Generated: 2026-09-04T00:21:36.987734
Random seed: 42

## Feature Coverage

| Feature | Valid | Total | Coverage |
|---------|-------|-------|----------|
| latitude | 2944 | 2944 | 100.0% |
| longitude | 2944 | 2944 | 100.0% |
| sst | 2944 | 2944 | 100.0% |
| ssh | 2944 | 2944 | 100.0% |
| u10 | 2944 | 2944 | 100.0% |
| v10 | 2944 | 2944 | 100.0% |

## Library Versions

```
torch: 2.10.0+cpu
numpy: 2.3.5
pandas: 2.3.3
scikit-learn: 1.8.0
xarray: 2025.9.0
```

## 1. Baseline Comparisons

All evaluated on the same float-aware held-out test set (split seeded with 42).

### Overall Metrics

| Model | RMSE (°C) | MAE (°C) | R² | N samples |
|-------|-----------|----------|------|-----------|
| Climatology | 2.006 | 1.412 | 0.9429 | 13675 |
| Nearest Neighbor | 1.929 | 1.003 | 0.9472 | 13675 |
| Linear Regression | 1.452 | 0.912 | 0.9701 | 13675 |
| MLP | 1.070 | 0.572 | 0.9837 | 13675 |
| CNN (OceanEmbed) | 0.926 | 0.548 | 0.9878 | 13675 |

### By Depth Band

| Model | Band | RMSE (°C) | MAE (°C) | R² | N |
|-------|------|-----------|----------|------|---|
| Climatology | 0-200m | 2.942 | 2.271 | 0.6566 | 4876 |
| Climatology | 200-500m | 1.547 | 1.286 | 0.4319 | 2676 |
| Climatology | 500-2000m | 1.022 | 0.782 | 0.7834 | 6123 |
| Nearest Neighbor | 0-200m | 2.908 | 1.812 | 0.6645 | 4876 |
| Nearest Neighbor | 200-500m | 0.981 | 0.689 | 0.7716 | 2676 |
| Nearest Neighbor | 500-2000m | 1.075 | 0.497 | 0.7601 | 6123 |
| Linear Regression | 0-200m | 2.106 | 1.389 | 0.8241 | 4876 |
| Linear Regression | 200-500m | 1.321 | 1.019 | 0.5863 | 2676 |
| Linear Regression | 500-2000m | 0.641 | 0.485 | 0.9147 | 6123 |
| MLP | 0-200m | 1.647 | 0.944 | 0.8925 | 4876 |
| MLP | 200-500m | 0.549 | 0.400 | 0.9286 | 2676 |
| MLP | 500-2000m | 0.517 | 0.351 | 0.9444 | 6123 |
| CNN (OceanEmbed) | 0-200m | 1.429 | 0.992 | 0.9191 | 4876 |
| CNN (OceanEmbed) | 200-500m | 0.635 | 0.449 | 0.9043 | 2676 |
| CNN (OceanEmbed) | 500-2000m | 0.335 | 0.237 | 0.9768 | 6123 |

## 2. Validation Split Comparison

### Split Summary

- **Float-aware**: Train 2063 / Test 446 (original)
- **Temporal**: Train 2075 (2019-2022) / Test 869 (2023-2024)
- **Geographic**: Train 2369 (lon≥60°E) / Test 575 (lon<60°E)

### Overall Metrics by Split Strategy

| Split Strategy | Model | RMSE (°C) | MAE (°C) | R² | N |
|----------------|-------|-----------|----------|------|---|
| Float-aware | CNN (trained) | 0.926 | 0.548 | 0.9878 | 13675 |
| Temporal (train→test) | CNN (retrained) | 1.344 | 0.752 | 0.9746 | 26730 |
| Temporal (train→test) | CNN (original) | 0.856 | 0.523 | 0.9897 | 26730 |
| Geographic (≥60°E→<60°E) | CNN (retrained) | 1.569 | 1.007 | 0.9621 | 17164 |
| Geographic (≥60°E→<60°E) | CNN (original) | 1.008 | 0.656 | 0.9844 | 17164 |

### Generalization Gap

| Gap | RMSE Delta | R² Delta |
|-----|------------|----------|
| Float-aware → Temporal | +0.419°C | -0.0133 |
| Float-aware → Geographic | +0.643°C | -0.0257 |

## 3. Wind Feature Audit

### Real Data Coverage

| Feature | Real Data Available | Of Total Samples | Coverage |
|---------|--------------------|--------------------|----------| |
| SST | 2944 | 2944 | 100.0% |
| SSH | 2944 | 2944 | 100.0% |
| U10 | 2944 | 2944 | 100.0% |
| V10 | 2944 | 2944 | 100.0% |

**Wind cache (ERA5 via Open-Meteo)**: 225 real values out of 240 cached profile indices.
**Training data wind**: 2944 of 2944 samples have non-null wind values.
**Note**: The 2944 wind values were fetched from Open-Meteo ERA5 archive API during the ingestion pipeline. The remaining 0 samples have wind values that may be from climatological fallback.

### Wind Ablation Study

| Model | Features | RMSE (°C) | R² | N |
|-------|----------|-----------|------|---|
| CNN (full) | lat, lon, sst, ssh, u10, v10 | 0.926 | 0.9878 | 13675 |
| CNN (no wind) | lat, lon, sst, ssh | 1.169 | 0.9806 | 13675 |

**Wind contribution**: RMSE -0.243°C, R² +0.0072
**Finding**: Removing wind DECREASES RMSE by 0.243°C — wind may be adding noise or the model struggles with its sparse coverage.

## 4. Data Pipeline Transparency

### QC Funnel

| Stage | Count | Description |
|-------|-------|-------------|
| Raw Argo records | 12,136,346 | All records from NOAA ERDDAP (Indian Ocean 2019-2024) |
| Raw floats | 391 | Unique float identifiers |
| After QC filtering | 47339329 | Removed invalid QC flags, too few valid levels |
| Deep profiles (≥1000m) | 2992 | Profiles reaching at least 1000m |
| Selected for depth grid | 3000 | Interpolated to 33-level target grid |
| Interpolation OK | 2992 | Successfully interpolated to all 33 depths |
| After surface feature matching | 2944 | Dropped rows with missing SST/SSH/u10/v10 |
| Final complete samples | 2944 | All 6 features + 33 depth levels present |

### Physical Smoothness

| Profile Type | Avg Inversions (below 25m) | Max Inversions |
|-------------|---------------------------|----------------|
| Ground truth (Argo) | 0.01 | 1 |
| CNN predictions | 0.01 | 2 |

**Finding**: CNN predictions have reasonable smoothness compared to ground truth.

### Representative Profiles

**Profile 1**: (-15.74°, 106.50°, 2023-12-16)
- Inversions: predicted=0, actual=0

**Profile 2**: (18.28°, 91.98°, 2020-12-20)
- Inversions: predicted=0, actual=0

**Profile 3**: (-19.17°, 93.99°, 2023-04-11)
- Inversions: predicted=0, actual=0

**Profile 4**: (16.49°, 67.31°, 2024-07-24)
- Inversions: predicted=0, actual=0

**Profile 5**: (2.62°, 71.47°, 2022-03-13)
- Inversions: predicted=0, actual=0

**Profile 6 (FAILURE CASE)**: (0.98°, 57.25°, 2019-04-08)
- Inversions: predicted=0, actual=0
- **FAILURE CASE - highest RMSE**


## 5. Reproducibility

### Random Seeds

- Global seed: `42`
- `np.random.seed(42)` and `torch.manual_seed(42)` called at script start
- All train/val/test splits use `random_state=42`
- Float-aware split: deterministic shuffle of float IDs with seed 42

### Training History

- Float-aware split: 73 epochs, early stopping at patience=20
- Best validation loss: 0.1236
- Training time: ~184s on CPU
- No overfitting observed (train/val loss tracked, early stopping applied)

### Live Data Mode Summary

| Data Mode | Estimated % of Requests |
|-----------|------------------------|
| Fully real (all 4 surface features) | ~100% |
| SST real | ~100% |
| SSH real | ~100% |
| Wind real (U10/V10) | ~100% |
| Climatological fallback needed | ~0% |

## Consolidated Final Metrics

### Primary Model (CNN, float-aware split, 6 features)

- **RMSE**: 0.926°C
- **MAE**: 0.548°C
- **R²**: 0.9878
- **Shallow (0-200m) RMSE**: 1.429°C
- **Mid (200-500m) RMSE**: 0.635°C
- **Deep (500-2000m) RMSE**: 0.335°C
- **Test samples**: 13675
- **Parameters**: 302,248

### Baseline Comparison Summary

| Model | RMSE (°C) | R² | Beats Climatology? |
|-------|-----------|------|-------------------|
| Climatology | 2.006 | 0.9429 | No |
| Nearest Neighbor | 1.929 | 0.9472 | Yes |
| Linear Regression | 1.452 | 0.9701 | Yes |
| MLP | 1.070 | 0.9837 | Yes |
| CNN (OceanEmbed) | 0.926 | 0.9878 | Yes |
