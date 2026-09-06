"""
Train Spatial CNN with 7 Input Channels
Uses point data with synthetic grid patches for training.
In production, this would use real 0.25° gridded CMEMS data.

Input: 7 channels (SST, SSS, SSH, u10, v10, current_u, current_v)
Output: 15 depth levels (0-1000m)
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.spatial_model import OceanEmbedSpatialCNN
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import config


def create_spatial_grid_patches(df, feature_cols, target_cols, patch_size=8):
    """
    Create spatial grid patches from point data.
    Each point becomes the center of a patch, with neighbors simulated
    by adding physically-motivated spatial perturbations.
    Grid spacing: 0.25 degrees (the target resolution).
    """
    rng = np.random.RandomState(42)
    n_features = len(feature_cols)
    n_samples = len(df)
    half = patch_size // 2

    # Pre-allocate arrays
    X = np.zeros((n_samples, n_features, patch_size, patch_size), dtype=np.float32)
    y = np.zeros((n_samples, len(target_cols)), dtype=np.float32)

    for idx, (_, row) in enumerate(df.iterrows()):
        center = np.array([row[c] for c in feature_cols])

        for i in range(patch_size):
            for j in range(patch_size):
                di, dj = i - half, j - half
                # Spatial variation: small gradient + noise
                gradient = np.array([
                    di * 0.05 + dj * 0.02,  # SST varies with distance
                    dj * 0.005,              # SSS gradient (small)
                    di * 0.03,               # SSH gradient
                    di * 0.1 + dj * 0.05,    # u10 wind
                    dj * 0.1 + di * 0.05,    # v10 wind
                    di * 0.08,               # current_u
                    dj * 0.08,               # current_v
                ])
                noise = rng.normal(0, 0.005, n_features)
                X[idx, :, i, j] = center + gradient + noise

        y[idx] = [row[c] for c in target_cols]

    return X, y


def train_spatial_model():
    """Train the spatial CNN with 7 input channels."""
    print("="*60)
    print("SPATIAL CNN TRAINING (7 channels -> 15 depths)")
    print("="*60)

    # Load dataset
    df = pd.read_parquet('data/processed/oceanembed_real_dataset.parquet')

    # 7 input channels (use latitude/longitude from the parquet)
    feature_cols = ['latitude', 'longitude', 'sst', 'sss', 'ssh', 'u10', 'v10', 'current_u', 'current_v']
    input_cols = ['sst', 'sss', 'ssh', 'u10', 'v10', 'current_u', 'current_v']  # 7 physical channels for the CNN
    spatial_cols = ['latitude', 'longitude']  # used as context, not CNN channels

    # 13 model-predicted depths (10-1000m); 0m/5m added as SST-anchored in post-processing
    target_depths = [10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    target_cols = [f'T_{d}m' for d in target_depths]
    available_targets = [c for c in target_cols if c in df.columns]
    print(f"Using 13-level targets: {len(available_targets)} available")

    # Filter complete samples (need all 7 input features + lat/lon + all targets)
    valid = df[feature_cols + available_targets].notna().all(axis=1)
    df_valid = df[valid].copy()
    print(f"Complete samples: {len(df_valid)}")

    # Create patches from the 7 physical input channels
    X, y = create_spatial_grid_patches(df_valid, input_cols, available_targets)
    print(f"Patches created: X={X.shape}, y={y.shape}")

    # Split
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    Xtr, Xval, Xte = X[idx[:train_end]], X[idx[train_end:val_end]], X[idx[val_end:]]
    ytr, yval, yte = y[idx[:train_end]], y[idx[train_end:val_end]], y[idx[val_end:]]

    # Standardize features (per channel)
    scaler_X = StandardScaler()
    Xtr_flat = Xtr.reshape(len(Xtr), -1)
    Xval_flat = Xval.reshape(len(Xval), -1)
    Xte_flat = Xte.reshape(len(Xte), -1)

    scaler_X.fit(Xtr_flat)
    Xtr_s = scaler_X.transform(Xtr_flat).reshape(Xtr.shape)
    Xval_s = scaler_X.transform(Xval_flat).reshape(Xval.shape)
    Xte_s = scaler_X.transform(Xte_flat).reshape(Xte.shape)

    # Standardize targets
    scaler_y = StandardScaler().fit(ytr)
    ytr_s = scaler_y.transform(ytr)
    yval_s = scaler_y.transform(yval)

    print(f"Train: {len(Xtr)}, Val: {len(Xval)}, Test: {len(Xte)}")

    # Create model
    n_channels = 7  # SST, SSS, SSH, u10, v10, current_u, current_v
    model = OceanEmbedSpatialCNN(in_channels=n_channels, out_depths=len(available_targets))
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    Xtr_t = torch.FloatTensor(Xtr_s).to(device)
    ytr_t = torch.FloatTensor(ytr_s).to(device)
    Xval_t = torch.FloatTensor(Xval_s).to(device)
    yval_t = torch.FloatTensor(scaler_y.transform(yval)).to(device)
    Xte_t = torch.FloatTensor(Xte_s).to(device)
    yte_t = torch.FloatTensor(scaler_y.transform(yte)).to(device)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    print("\nTraining...")
    start_time = time.time()

    for epoch in range(100):
        model.train()
        pred = model(Xtr_t)
        loss = criterion(pred, ytr_t)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xval_t)
            val_loss = criterion(val_pred, yval_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_X_mean': scaler_X.mean_.tolist(),
                'scaler_X_scale': scaler_X.scale_.tolist(),
                'scaler_y_mean': scaler_y.mean_.tolist(),
                'scaler_y_scale': scaler_y.scale_.tolist(),
                'feature_cols': input_cols,
                'target_depths': target_depths,
                'target_cols': available_targets,
                'n_channels': 7,
                'n_depths': len(available_targets),
            }, 'models/checkpoints/ocean_embed_spatial_7ch.pt')
        else:
            patience_counter += 1

        if (epoch+1) % 25 == 0:
            print(f"  Epoch {epoch+1}: train={loss.item():.4f}, val={val_loss:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_norm = model(Xte_t).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_norm)

    rmse = np.sqrt(np.mean((pred - yte)**2))
    mae = np.mean(np.abs(pred - yte))
    ss_res = np.sum((pred - yte)**2)
    ss_tot = np.sum((yte - np.mean(yte, axis=0))**2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n{'='*60}")
    print(f"FINAL METRICS (7-channel spatial CNN)")
    print(f"{'='*60}")
    print(f"Test samples: {len(Xte)}")
    print(f"RMSE: {rmse:.4f} C")
    print(f"MAE:  {mae:.4f} C")
    print(f"R2:   {r2:.4f}")

    # Save metrics
    metrics = {
        'overall_rmse': float(rmse),
        'overall_mae': float(mae),
        'overall_r2': float(r2),
        'n_test_samples': len(Xte),
        'model_params': sum(p.numel() for p in model.parameters()),
        'n_channels': 7,
        'n_depths': len(available_targets),
        'target_depths': target_depths,
    }
    with open('data/reports/spatial_7ch_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    train_spatial_model()
