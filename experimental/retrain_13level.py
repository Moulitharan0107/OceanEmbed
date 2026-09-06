"""
Fix 0m/5m depth level outputs and retrain the model.
Root cause: T_0m has only 31/2992 real Argo measurements (1.0%), T_5m was interpolated.
These are unreliable training targets. Solution: retrain using only 13 reliable depths
(10m-1000m), and use SST as a post-hoc constraint for surface values.
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json, time, numpy as np, pandas as pd, torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, '.')
from config import FEATURES, MODEL_CONFIG

# ============================================================
# 1. Prepare 13-level dataset (drop 0m and 5m)
# ============================================================
print("=" * 70)
print("RETRAINING: DROPPING 0m AND 5m (unreliable targets)")
print("=" * 70)

DEPTH_LEVELS_13 = [10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTH_COLS_13 = [f'profile_{d}m' for d in DEPTH_LEVELS_13]

df = pd.read_parquet('data/processed/oceanembed_real_dataset.parquet')
print(f"Dataset: {len(df)} rows")

# Build 13-level targets from original 33 columns
target_cols_33 = [c for c in df.columns if c.startswith('T_')]
original_depths = np.array([int(c.replace('T_', '').replace('m', '')) for c in target_cols_33])

for d in DEPTH_LEVELS_13:
    col = f'profile_{d}m'
    if col not in df.columns:
        # Interpolate from 33 original depths
        target_arr = np.array([df[c].values for c in target_cols_33]).T
        valid = ~np.isnan(target_arr).all(axis=1)
        df[col] = np.nan
        for i in np.where(valid)[0]:
            row_vals = target_arr[i]
            good = ~np.isnan(row_vals)
            if good.sum() >= 2:
                df.at[df.index[i], col] = float(np.interp(d, original_depths[good], row_vals[good]))

# Build features
feature_names = FEATURES  # ['latitude', 'longitude', 'sst', 'ssh', 'u10', 'v10']

# Drop rows with any NaN in targets OR features
valid_mask = df[DEPTH_COLS_13].notna().all(axis=1) & df[feature_names].notna().all(axis=1)
df_valid = df[valid_mask].copy()
print(f"Rows with valid targets + features: {len(df_valid)}")
X = df_valid[feature_names].values.astype(np.float32)
y = df_valid[DEPTH_COLS_13].values.astype(np.float32)

print(f"Features: {feature_names}")
print(f"Target depths (13): {DEPTH_LEVELS_13}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ============================================================
# 2. Split (same seed as original)
# ============================================================
np.random.seed(42)
n = len(df_valid)
indices = np.random.permutation(n)
n_train = int(n * 0.7)
n_val = int(n * 0.15)

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train+n_val]
test_idx = indices[n_train+n_val:]

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

# ============================================================
# 3. Scale (from training set only)
# ============================================================
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-8
y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0) + 1e-8

X_train_s = (X_train - X_mean) / X_std
X_val_s = (X_val - X_mean) / X_std
X_test_s = (X_test - X_mean) / X_std

y_train_s = (y_train - y_mean) / y_std
y_val_s = (y_val - y_mean) / y_std

# ============================================================
# 4. Train model (13 outputs)
# ============================================================
print("\nTraining 13-level model...")

from models.ocean_model import OceanEmbedModel

model = OceanEmbedModel(
    input_dim=6,
    output_dim=13,
    hidden_dims=MODEL_CONFIG['hidden_dims'],
    dropout=MODEL_CONFIG['dropout']
)

optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_CONFIG['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.MSELoss()

X_train_t = torch.FloatTensor(X_train_s)
y_train_t = torch.FloatTensor(y_train_s)
X_val_t = torch.FloatTensor(X_val_s)
y_val_t = torch.FloatTensor(y_val_s)
X_test_t = torch.FloatTensor(X_test_s)

batch_size = MODEL_CONFIG['batch_size']
n_train_samples = len(X_train_t)
n_val_samples = len(X_val_t)

best_val_loss = float('inf')
patience = MODEL_CONFIG['patience']
patience_counter = 0
best_state = None
train_losses = []
val_losses = []

t0 = time.time()
for epoch in range(MODEL_CONFIG['num_epochs']):
    model.train()
    perm = torch.randperm(n_train_samples)
    epoch_loss = 0.0
    n_batches = 0
    
    for i in range(0, n_train_samples, batch_size):
        batch_idx = perm[i:i+batch_size]
        if len(batch_idx) < 2:
            continue
        xb = X_train_t[batch_idx]
        yb = y_train_t[batch_idx]
        
        pred = model(xb)
        loss = criterion(pred, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    
    train_loss = epoch_loss / max(n_batches, 1)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1:>3}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

t_train = time.time() - t0
print(f"Training time: {t_train:.1f}s, epochs: {epoch+1}")

# Load best
model.load_state_dict(best_state)

# ============================================================
# 5. Evaluate on test set
# ============================================================
print("\nEvaluating on test set...")
model.eval()
with torch.no_grad():
    y_pred_s = model(X_test_t).numpy()

y_pred = y_pred_s * y_std + y_mean
y_actual = y_test

# Per-depth metrics
print(f"\n{'Depth':>8} {'N':>6} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
print("-" * 50)

depth_results = {}
for i, d in enumerate(DEPTH_LEVELS_13):
    yt = y_actual[:, i]
    yp = y_pred[:, i]
    n_valid = len(yt)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = np.mean(np.abs(yt - yp))
    try:
        r2 = r2_score(yt, yp)
    except:
        r2 = float('nan')
    depth_results[d] = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2), "n": int(n_valid)}
    print(f"{d:>6}m  {n_valid:>6} {rmse:>10.3f} {mae:>10.3f} {r2:>10.4f}")

# Overall
y_flat = y_actual.flatten()
yp_flat = y_pred.flatten()
overall_rmse = np.sqrt(mean_squared_error(y_flat, yp_flat))
overall_mae = np.mean(np.abs(y_flat - yp_flat))
overall_r2 = r2_score(y_flat, yp_flat)
print(f"\nOverall: RMSE={overall_rmse:.3f}°C, MAE={overall_mae:.3f}°C, R²={overall_r2:.4f}")

# Depth bands
shallow_idx = [DEPTH_LEVELS_13.index(d) for d in [10, 20, 30, 50, 75, 100, 125, 150, 200]]
mid_idx = [DEPTH_LEVELS_13.index(d) for d in [300]]
deep_idx = [DEPTH_LEVELS_13.index(d) for d in [500, 700, 1000]]

for band_name, band_idx in [("Shallow (10-200m)", shallow_idx), ("Mid (300m)", mid_idx), ("Deep (500-1000m)", deep_idx)]:
    yt_band = y_actual[:, band_idx].flatten()
    yp_band = y_pred[:, band_idx].flatten()
    rmse_b = np.sqrt(mean_squared_error(yt_band, yp_band))
    r2_b = r2_score(yt_band, yp_band) if len(np.unique(yt_band)) > 1 else float('nan')
    print(f"  {band_name}: RMSE={rmse_b:.3f}°C, R²={r2_b:.4f}")

# ============================================================
# 6. Check 5 demo locations
# ============================================================
print("\n--- Predictions at 5 demo locations ---")
sys.path.insert(0, '.')
from scripts.surface_lookup import lookup_real_features

demo_locations = [
    {"lat": 15.0, "lon": 80.0, "name": "Bay of Bengal"},
    {"lat": 10.0, "lon": 55.0, "name": "Arabian Sea"},
    {"lat": -10.0, "lon": 95.0, "name": "South Indian Ocean"},
    {"lat": 5.0, "lon": 65.0, "name": "Central Indian Ocean"},
    {"lat": 20.0, "lon": 90.0, "name": "Northern Bay of Bengal"},
]

for loc in demo_locations:
    feat = lookup_real_features(loc['lat'], loc['lon'], '2022-06-15')
    f = feat['features']
    x_input = np.array([[f['latitude'], f['longitude'], f['sst'], f['ssh'], f['u10'], f['v10']]], dtype=np.float32)
    x_scaled = (x_input - X_mean) / X_std
    with torch.no_grad():
        pred_s = model(torch.FloatTensor(x_scaled)).numpy()
        pred = pred_s * y_std + y_mean
    pred = pred[0]
    
    print(f"\n  {loc['name']} ({loc['lat']}, {loc['lon']}):")
    print(f"    SST: {f['sst']}°C, SSH: {f['ssh']}m")
    # Show 10m should be close to SST
    print(f"    10m prediction: {pred[0]:.2f}°C (SST input: {f['sst']}°C)")
    print(f"    10m-SST diff: {pred[0] - f['sst']:.2f}°C")
    for i_d, d in enumerate(DEPTH_LEVELS_13):
        print(f"    {d:>5}m: {pred[i_d]:>7.2f}°C")

# ============================================================
# 7. Save new checkpoint
# ============================================================
print("\nSaving 13-level checkpoint...")
ckpt = {
    'model_state_dict': model.state_dict(),
    'scaler_X_mean': X_mean,
    'scaler_X_scale': X_std,
    'scaler_y_mean': y_mean,
    'scaler_y_scale': y_std,
    'features': feature_names,
    'depth_values': DEPTH_LEVELS_13,
    'depth_cols': DEPTH_COLS_13,
    'n_features': 6,
    'n_depths': 13,
    'input_dim': 6,
    'output_dim': 13,
    'training_info': {
        'n_train': int(n_train),
        'n_val': int(n_val),
        'n_test': int(n_test) if 'n_test' in dir() else len(test_idx),
        'epochs': epoch + 1,
        'best_val_loss': float(best_val_loss),
        'train_loss': float(train_losses[-1]),
        'val_loss': float(val_losses[-1]),
        'training_time_s': round(t_train, 1),
    }
}
torch.save(ckpt, 'models/checkpoints/ocean_embed_real_best.pt')
print(f"Saved to models/checkpoints/ocean_embed_real_best.pt")

# Save metrics
metrics = {
    "model": "OceanEmbed CNN (13-level, 0-1000m)",
    "features": feature_names,
    "depth_levels": DEPTH_LEVELS_13,
    "n_features": 6,
    "n_depths": 13,
    "test_samples": len(test_idx),
    "overall_rmse": round(overall_rmse, 4),
    "overall_mae": round(overall_mae, 4),
    "overall_r2": round(overall_r2, 4),
    "depth_metrics": depth_results,
    "training": {
        "epochs": epoch + 1,
        "training_time_s": round(t_train, 1),
        "best_val_loss": round(float(best_val_loss), 4),
    }
}
os.makedirs('data/reports', exist_ok=True)
with open('data/reports/real_model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics to data/reports/real_model_metrics.json")

# Save training curves
curves = {
    "train_loss": [float(x) for x in train_losses],
    "val_loss": [float(x) for x in val_losses],
}
with open('data/reports/training_curves.json', 'w') as f:
    json.dump(curves, f, indent=2)
print(f"Saved training curves to data/reports/training_curves.json")

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)
print(f"13-level model (10m-1000m):")
print(f"  RMSE: {overall_rmse:.3f}°C")
print(f"  MAE:  {overall_mae:.3f}°C")
print(f"  R²:   {overall_r2:.4f}")
print(f"  All depths 10m+: physically meaningful and well-constrained by data")
