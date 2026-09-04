#!/usr/bin/env python3
"""
Train OceanEmbed on REAL Argo + satellite data.
Float-aware split, proper preprocessing, full evaluation.
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
# 1. LOAD AND PREPARE DATA
# ============================================================
print("=" * 60)
print("Loading REAL dataset")
print("=" * 60)

df = pd.read_parquet("data/processed/oceanembed_real_dataset.parquet")
print(f"Raw: {len(df)} rows")

# Drop rows with missing surface features
required_features = ["latitude", "longitude", "sst", "ssh", "u10", "v10"]
df_complete = df.dropna(subset=required_features).copy()
print(f"After dropping NaN features: {len(df_complete)} rows")

# Get depth column names and values
depth_cols = [c for c in df.columns if c.startswith("T_")]
depth_values = [int(c.replace("T_", "").replace("m", "")) for c in depth_cols]
print(f"Depth levels: {len(depth_cols)} ({depth_values[0]}m - {depth_values[-1]}m)")

# For each row, only use depth columns where the value is not NaN
# (don't fill NaNs with 0 or interpolate - just keep what's real)
y_raw = df_complete[depth_cols].values.astype(np.float32)
X_raw = df_complete[required_features].values.astype(np.float32)
float_ids = df_complete["float_id"].values

print(f"X shape: {X_raw.shape}, y shape: {y_raw.shape}")

# ============================================================
# 2. FLOAT-AWARE TRAIN/VAL/TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("Float-aware train/val/test split")
print("=" * 60)

# Group by float_id to prevent data leakage
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

# ============================================================
# 3. PREPROCESSING (fit on train only)
# ============================================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_norm = scaler_X.fit_transform(X_train).astype(np.float32)
X_val_norm = scaler_X.transform(X_val).astype(np.float32)
X_test_norm = scaler_X.transform(X_test).astype(np.float32)

# For y: handle NaN targets by fitting on non-NaN values only
# Strategy: replace NaN with 0 for scaling, but track NaN mask for loss
y_train_filled = np.nan_to_num(y_train, nan=0.0)
y_val_filled = np.nan_to_num(y_val, nan=0.0)
y_test_filled = np.nan_to_num(y_test, nan=0.0)

scaler_y.fit(y_train_filled)
y_train_norm = scaler_y.transform(y_train_filled).astype(np.float32)
y_val_norm = scaler_y.transform(y_val_filled).astype(np.float32)
y_test_norm = scaler_y.transform(y_test_filled).astype(np.float32)

# NaN masks for loss computation
y_train_mask = (~np.isnan(y_train)).astype(np.float32)
y_val_mask = (~np.isnan(y_val)).astype(np.float32)
y_test_mask = (~np.isnan(y_test)).astype(np.float32)

print(f"\nFeature scaler: mean={scaler_X.mean_[:3].round(2)}, scale={scaler_X.scale_[:3].round(2)}")
print(f"Target scaler: mean range=[{scaler_y.mean_.min():.1f}, {scaler_y.mean_.max():.1f}]")

# ============================================================
# 4. CREATE DATALOADERS
# ============================================================
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
# 5. CREATE AND TRAIN MODEL
# ============================================================
print("\n" + "=" * 60)
print("Training OceanEmbed Model on REAL DATA")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = OceanEmbedModel(
    input_dim=config.NUM_FEATURES,
    output_dim=len(depth_cols),
    dropout=0.3,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")
model = model.to(device)

criterion = OceanEmbedLoss(monotonicity_weight=0.05)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Training loop
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
        # Masked MSE loss
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

    # Compute RMSE in original units
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

    # Early stopping
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

# Restore best
if best_state:
    model.load_state_dict(best_state)
    model = model.to(device)

# ============================================================
# 6. TEST EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("TEST SET EVALUATION (Held-Out)")
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
for j, (col, dv) in enumerate(zip(depth_cols, depth_values)):
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
    print(f"{dv:>6}m  {rmse_j:>8.3f}C  {r2_j:>8.4f}  {n_valid:>6}")

# Band averages
shallow_mask = np.array([dv <= 200 for dv in depth_values])
deep_mask = np.array([dv >= 500 for dv in depth_values])

shallow_valid = valid[:, shallow_mask].flatten()
deep_valid = valid[:, deep_mask].flatten()
shallow_preds = preds[:, shallow_mask].flatten()
shallow_trues = trues[:, shallow_mask].flatten()
deep_preds = preds[:, deep_mask].flatten()
deep_trues = trues[:, deep_mask].flatten()

shallow_rmse = np.sqrt(np.mean((shallow_preds[shallow_valid] - shallow_trues[shallow_valid]) ** 2))
deep_rmse = np.sqrt(np.mean((deep_preds[deep_valid] - deep_trues[deep_valid]) ** 2))
print(f"\nShallow (0-200m) RMSE: {shallow_rmse:.3f} C")
print(f"Deep (500-2000m) RMSE: {deep_rmse:.3f} C")

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
# 7. SAVE EVERYTHING
# ============================================================
print("\n" + "=" * 60)
print("Saving")
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
    "depth_values": depth_values,
    "depth_cols": depth_cols,
    "n_features": config.NUM_FEATURES,
    "n_depths": len(depth_cols),
    "input_dim": config.NUM_FEATURES,
    "output_dim": len(depth_cols),
}
torch.save(checkpoint, ckpt_path)
print(f"Model: {ckpt_path}")

# Save metrics
metrics = {
    "dataset_type": "REAL DATA",
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
    "depth_values": depth_values,
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
}

metrics_path = "data/reports/real_model_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2, default=str)
print(f"Metrics: {metrics_path}")

# Save training history
with open("data/reports/real_training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"History: data/reports/real_training_history.json")

# Save example predictions for visualization
examples = []
np.random.seed(123)
example_idx = np.random.choice(len(preds), min(10, len(preds)), replace=False)
for i in example_idx:
    examples.append({
        "latitude": float(df_complete.iloc[test_idx[i]]["latitude"]),
        "longitude": float(df_complete.iloc[test_idx[i]]["longitude"]),
        "date": str(df_complete.iloc[test_idx[i]]["date"]),
        "predicted": preds[i].tolist(),
        "actual": trues[i].tolist(),
        "depth_values": depth_values,
    })
with open("data/reports/real_test_examples.json", "w") as f:
    json.dump(examples, f, indent=2)
print(f"Examples: data/reports/real_test_examples.json")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Test RMSE: {rmse:.3f} C")
print(f"Test MAE:  {mae:.3f} C")
print(f"Test R2:   {r2:.4f}")
print(f"Checkpoint: {ckpt_path}")
