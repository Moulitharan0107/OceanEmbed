#!/usr/bin/env python3
"""
Retrain OceanEmbed with 15 depth levels (0-1000m) for INCOIS project scope.
Re-derives targets from cached Argo data, does NOT re-download anything.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, json, time, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models.ocean_model import OceanEmbedModel, OceanEmbedLoss

# ============================================================
# 1. LOAD EXISTING ARGO DATA AND DERIVE 15-LEVEL TARGETS
# ============================================================
print("=" * 60)
print("Step 1: Derive 15-level targets from cached Argo data")
print("=" * 60)

NEW_DEPTH_LEVELS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
NEW_DEPTH_COLS = [f"T_{d}m" for d in NEW_DEPTH_LEVELS]

# Load the existing 33-level dataset
df_33 = pd.read_parquet("data/processed/oceanembed_real_dataset.parquet")
print(f"Loaded {len(df_33)} profiles (33-level dataset)")

# Map existing 33-level columns to the new 15 levels
# Existing cols: T_0m, T_10m, T_20m, T_30m, T_40m, T_50m, T_75m, T_100m, T_125m, T_150m,
#                T_175m, T_200m, T_250m, T_300m, T_350m, T_400m, T_450m, T_500m, T_600m,
#                T_700m, T_800m, T_900m, T_1000m, T_1100m...T_2000m
# New 15 cols: T_0m, T_5m, T_10m, T_20m, T_30m, T_50m, T_75m, T_100m, T_125m, T_150m,
#              T_200m, T_300m, T_500m, T_700m, T_1000m

# Build new dataframe with surface features + new depth columns
records = []
for idx, row in df_33.iterrows():
    rec = {
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "date": str(row["date"]),
        "float_id": int(row["float_id"]),
        "max_depth": float(row["max_depth"]),
        "sst": row["sst"] if pd.notna(row["sst"]) else None,
        "ssh": row["ssh"] if pd.notna(row["ssh"]) else None,
        "u10": row["u10"] if pd.notna(row["u10"]) else None,
        "v10": row["v10"] if pd.notna(row["v10"]) else None,
    }
    
    # Direct copy for depths that exist in both grids
    for d in NEW_DEPTH_LEVELS:
        col = f"T_{d}m"
        if d == 5:
            # Interpolate T_5m from T_0m and T_10m
            t0 = row.get("T_0m", np.nan)
            t10 = row.get("T_10m", np.nan)
            if pd.notna(t0) and pd.notna(t10):
                rec[col] = round(float(t0 + (t10 - t0) * 5.0 / 10.0), 3)
            else:
                rec[col] = None
        else:
            val = row.get(col, np.nan)
            rec[col] = float(val) if pd.notna(val) else None
    records.append(rec)

df_15 = pd.DataFrame(records)
print(f"Built 15-level dataset: {len(df_15)} rows, {len(NEW_DEPTH_COLS)} depth columns")

# Check coverage
for col in NEW_DEPTH_COLS:
    n_valid = df_15[col].notna().sum()
    print(f"  {col}: {n_valid}/{len(df_15)} valid ({100*n_valid/len(df_15):.1f}%)")

# Drop rows with missing surface features
required = ["latitude", "longitude", "sst", "ssh", "u10", "v10"]
df_complete = df_15.dropna(subset=required).copy()
print(f"\nComplete samples (all features): {len(df_complete)}")

# Save 15-level dataset
os.makedirs("data/processed", exist_ok=True)
out_path = "data/processed/oceanembed_real_dataset_15level.parquet"
df_complete.to_parquet(out_path, index=False)
print(f"Saved: {out_path}")

# ============================================================
# 2. PREPARE TRAINING DATA
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Prepare training data (float-aware split)")
print("=" * 60)

feature_cols = ["latitude", "longitude", "sst", "ssh", "u10", "v10"]
X_raw = df_complete[feature_cols].values.astype(np.float32)
y_raw = df_complete[NEW_DEPTH_COLS].values.astype(np.float32)
float_ids = df_complete["float_id"].values

print(f"X shape: {X_raw.shape}, y shape: {y_raw.shape}")

# Float-aware split (70/15/15)
unique_floats = np.unique(float_ids)
np.random.seed(42)
np.random.shuffle(unique_floats)

n_floats = len(unique_floats)
n_train_f = int(0.70 * n_floats)
n_val_f = int(0.15 * n_floats)

train_floats = set(unique_floats[:n_train_f])
val_floats = set(unique_floats[n_train_f:n_train_f + n_val_f])
test_floats = set(unique_floats[n_train_f + n_val_f:])

train_idx = np.array([i for i, fid in enumerate(float_ids) if fid in train_floats])
val_idx = np.array([i for i, fid in enumerate(float_ids) if fid in val_floats])
test_idx = np.array([i for i, fid in enumerate(float_ids) if fid in test_floats])

X_train, y_train = X_raw[train_idx], y_raw[train_idx]
X_val, y_val = X_raw[val_idx], y_raw[val_idx]
X_test, y_test = X_raw[test_idx], y_raw[test_idx]

print(f"Train: {len(X_train)} samples ({len(train_floats)} floats)")
print(f"Val:   {len(X_val)} samples ({len(val_floats)} floats)")
print(f"Test:  {len(X_test)} samples ({len(test_floats)} floats)")

# Preprocessing (fit on train only)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_norm = scaler_X.fit_transform(X_train).astype(np.float32)
X_val_norm = scaler_X.transform(X_val).astype(np.float32)
X_test_norm = scaler_X.transform(X_test).astype(np.float32)

y_train_filled = np.nan_to_num(y_train, nan=0.0)
y_val_filled = np.nan_to_num(y_val, nan=0.0)
y_test_filled = np.nan_to_num(y_test, nan=0.0)

scaler_y.fit(y_train_filled)
y_train_norm = scaler_y.transform(y_train_filled).astype(np.float32)
y_val_norm = scaler_y.transform(y_val_filled).astype(np.float32)
y_test_norm = scaler_y.transform(y_test_filled).astype(np.float32)

y_train_mask = (~np.isnan(y_train)).astype(np.float32)
y_val_mask = (~np.isnan(y_val)).astype(np.float32)
y_test_mask = (~np.isnan(y_test)).astype(np.float32)

batch_size = 64
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train_norm), torch.tensor(y_train_norm), torch.tensor(y_train_mask)),
    batch_size=batch_size, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val_norm), torch.tensor(y_val_norm), torch.tensor(y_val_mask)),
    batch_size=batch_size, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test_norm), torch.tensor(y_test_norm), torch.tensor(y_test_mask)),
    batch_size=batch_size, shuffle=False, num_workers=0
)

# ============================================================
# 3. TRAIN MODEL (15 output levels)
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Train OceanEmbed (15-level output)")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = OceanEmbedModel(
    input_dim=config.NUM_FEATURES,
    output_dim=len(NEW_DEPTH_LEVELS),
    dropout=0.3,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")
model = model.to(device)

criterion = OceanEmbedLoss(monotonicity_weight=0.05)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

best_val_loss = float("inf")
patience = 20
patience_counter = 0
best_state = None
history = {"train_loss": [], "val_loss": [], "val_rmse": []}

t0 = time.time()
for epoch in range(200):
    # Train
    model.train()
    train_losses = []
    for batch_X, batch_y, mask in train_loader:
        batch_X, batch_y, mask = batch_X.to(device), batch_y.to(device), mask.to(device)
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = ((pred - batch_y) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())
    scheduler.step()
    avg_train = np.mean(train_losses)

    # Validate
    model.eval()
    val_losses = []
    val_preds_all, val_true_all = [], []
    with torch.no_grad():
        for batch_X, batch_y, mask in val_loader:
            batch_X, batch_y, mask = batch_X.to(device), batch_y.to(device), mask.to(device)
            pred = model(batch_X)
            loss = ((pred - batch_y) ** 2 * mask).sum() / (mask.sum() + 1e-8)
            val_losses.append(loss.item())
            val_preds_all.append(pred.cpu().numpy())
            val_true_all.append(batch_y.cpu().numpy())
    avg_val = np.mean(val_losses)

    preds_np = np.concatenate(val_preds_all)
    trues_np = np.concatenate(val_true_all)
    preds_denorm = scaler_y.inverse_transform(preds_np)
    trues_denorm = scaler_y.inverse_transform(trues_np)
    val_mask_combined = np.concatenate([m.numpy() for _, _, m in val_loader])
    valid = val_mask_combined > 0.5
    val_rmse = np.sqrt(np.mean((preds_denorm[valid] - trues_denorm[valid]) ** 2))

    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)
    history["val_rmse"].append(val_rmse)

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | ValRMSE: {val_rmse:.3f}C | LR: {lr:.2e}")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

elapsed = time.time() - t0
print(f"\nTraining done: {epoch+1} epochs in {elapsed:.1f}s, best val loss: {best_val_loss:.4f}")

if best_state:
    model.load_state_dict(best_state)
    model = model.to(device)

# ============================================================
# 4. TEST EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("Step 4: TEST SET EVALUATION (15-level, 0-1000m)")
print("=" * 60)

model.eval()
all_preds, all_trues, all_masks = [], [], []
with torch.no_grad():
    for batch_X, batch_y, mask in test_loader:
        batch_X = batch_X.to(device)
        pred = model(batch_X)
        all_preds.append(pred.cpu().numpy())
        all_trues.append(batch_y.numpy())
        all_masks.append(mask.numpy())

preds_norm = np.concatenate(all_preds)
trues_norm = np.concatenate(all_trues)
masks = np.concatenate(all_masks)

preds = scaler_y.inverse_transform(preds_norm)
trues = scaler_y.inverse_transform(trues_norm)

valid = masks > 0.5

# Overall metrics
rmse = np.sqrt(np.mean((preds[valid] - trues[valid]) ** 2))
mae = np.mean(np.abs(preds[valid] - trues[valid]))
ss_res = np.sum((preds[valid] - trues[valid]) ** 2)
ss_tot = np.sum((trues[valid] - np.mean(trues[valid])) ** 2)
r2 = 1 - ss_res / (ss_tot + 1e-10)

print(f"\nTest Samples: {len(preds)}")
print(f"Overall RMSE: {rmse:.3f} C")
print(f"Overall MAE:  {mae:.3f} C")
print(f"Overall R2:   {r2:.4f}")

# Per-depth metrics
depth_rmse = []
depth_r2 = []
print(f"\n{'Depth':>8} {'RMSE':>10} {'R2':>10} {'N':>6}")
print("-" * 38)
for j, (col, dv) in enumerate(zip(NEW_DEPTH_COLS, NEW_DEPTH_LEVELS)):
    mask_j = masks[:, j] > 0.5
    n_valid = mask_j.sum()
    if n_valid < 10:
        depth_rmse.append(None)
        depth_r2.append(None)
        print(f"{dv:>6}m  {'N/A':>10} {'N/A':>10} {n_valid:>6}")
        continue
    rmse_j = np.sqrt(np.mean((preds[mask_j, j] - trues[mask_j, j]) ** 2))
    ss_res_j = np.sum((preds[mask_j, j] - trues[mask_j, j]) ** 2)
    ss_tot_j = np.sum((trues[mask_j, j] - np.mean(trues[mask_j, j])) ** 2)
    r2_j = 1 - ss_res_j / (ss_tot_j + 1e-10)
    depth_rmse.append(float(rmse_j))
    depth_r2.append(float(r2_j))
    print(f"{dv:>6}m  {rmse_j:>8.3f}C  {r2_j:>8.4f} {n_valid:>6}")

# Band averages
shallow_mask = np.array([dv <= 200 for dv in NEW_DEPTH_LEVELS])
deep_mask = np.array([dv >= 500 for dv in NEW_DEPTH_LEVELS])

shallow_valid = valid[:, shallow_mask].flatten()
deep_valid = valid[:, deep_mask].flatten()
shallow_preds = preds[:, shallow_mask].flatten()
shallow_trues = trues[:, shallow_mask].flatten()
deep_preds = preds[:, deep_mask].flatten()
deep_trues = trues[:, deep_mask].flatten()

shallow_rmse = np.sqrt(np.mean((shallow_preds[shallow_valid] - shallow_trues[shallow_valid]) ** 2))
deep_rmse = np.sqrt(np.mean((deep_preds[deep_valid] - deep_trues[deep_valid]) ** 2))
print(f"\nShallow (0-200m) RMSE: {shallow_rmse:.3f} C")
print(f"Deep (500-1000m) RMSE: {deep_rmse:.3f} C")

# Feature importance
model.eval()
X_test_tensor = torch.tensor(X_test_norm[:200], requires_grad=True).to(device)
pred = model(X_test_tensor)
pred.sum().backward()
grads = X_test_tensor.grad.abs().mean(dim=0).cpu().numpy()
total = grads.sum() + 1e-10
feature_importance = {config.FEATURES[i]: round(float(grads[i] / total), 4) for i in range(len(config.FEATURES))}

print("\nFeature Importance (gradient-based):")
for feat, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
    bar = "#" * int(imp * 50)
    print(f"  {feat:>10s}: {imp:.3f} {bar}")

# ============================================================
# 5. SAVE EVERYTHING
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Save model and metrics")
print("=" * 60)

# Save model checkpoint
ckpt_path = "models/checkpoints/ocean_embed_real_best.pt"
os.makedirs("models/checkpoints", exist_ok=True)
checkpoint = {
    "model_state_dict": model.state_dict(),
    "scaler_X_mean": scaler_X.mean_,
    "scaler_X_scale": scaler_X.scale_,
    "scaler_y_mean": scaler_y.mean_,
    "scaler_y_scale": scaler_y.scale_,
    "features": config.FEATURES,
    "depth_values": NEW_DEPTH_LEVELS,
    "depth_cols": NEW_DEPTH_COLS,
    "n_features": config.NUM_FEATURES,
    "n_depths": len(NEW_DEPTH_LEVELS),
    "input_dim": config.NUM_FEATURES,
    "output_dim": len(NEW_DEPTH_LEVELS),
}
torch.save(checkpoint, ckpt_path)
print(f"Model: {ckpt_path}")

# Save metrics
metrics = {
    "dataset_type": "REAL DATA (15-level, 0-1000m)",
    "overall_rmse": round(float(rmse), 4),
    "overall_mae": round(float(mae), 4),
    "overall_r2": round(float(r2), 4),
    "shallow_rmse": round(float(shallow_rmse), 4),
    "deep_rmse": round(float(deep_rmse), 4),
    "n_test_samples": int(len(preds)),
    "n_train_samples": int(len(X_train)),
    "n_val_samples": int(len(X_val)),
    "n_test_samples_count": int(len(X_test)),
    "n_train_floats": len(train_floats),
    "n_val_floats": len(val_floats),
    "n_test_floats": len(test_floats),
    "depth_values": NEW_DEPTH_LEVELS,
    "depth_levels": NEW_DEPTH_LEVELS,
    "depth_rmse": depth_rmse,
    "depth_r2": depth_r2,
    "feature_importance": feature_importance,
    "features_used": config.FEATURES,
    "n_params": n_params,
    "training_epochs": epoch + 1,
    "training_time_s": round(elapsed, 1),
    "timestamp": datetime.now().isoformat(),
    "data_sources": {
        "argo": "NOAA ERDDAP Argo (coriolis), Indian Ocean 2019-2024",
        "sst": "NOAA OISST v2.1",
        "ssh": "NESDIS Satellite Altimetry",
        "wind": "ERA5 reanalysis (Open-Meteo)",
        "sss": "DROPPED - SMOS 403 Forbidden",
    },
    "model_params": n_params,
}

metrics_path = "data/reports/real_model_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2, default=str)
print(f"Metrics: {metrics_path}")

# Save training history
with open("data/reports/real_training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"History: data/reports/real_training_history.json")

print("\n" + "=" * 60)
print("TRAINING COMPLETE - 15 LEVELS, 0-1000m")
print("=" * 60)
print(f"Test RMSE: {rmse:.3f} C")
print(f"Test MAE:  {mae:.3f} C")
print(f"Test R2:   {r2:.4f}")
print(f"Checkpoint: {ckpt_path}")
