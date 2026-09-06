"""Train proof-of-concept model using GLORYS reanalysis as training labels.
Evaluate against independent real Argo observations.

NEVER overwrites the main checkpoint."""
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import Ridge

# Configuration
GLORYS_CACHE = Path('data/raw/surface_cache/glorys/glorys_profiles.json')
ARGO_FILE = Path('data/processed/oceanembed_real_dataset.parquet')
TARGET_DEPTHS = [10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
FEATURE_COLS = ['latitude', 'longitude', 'sst', 'ssh', 'u10', 'v10']

print('='*60)
print('GLORYS PROOF-OF-CONCEPT MODEL')
print('='*60)

# Load data
glorys_cache = json.load(open(GLORYS_CACHE))
argo_df = pd.read_parquet(ARGO_FILE)
print(f'GLORYS profiles: {len(glorys_cache)}')
print(f'Argo profiles: {len(argo_df)}')

# Build training data
X_list = []
y_list = []
profile_indices = []  # Track which Argo profiles we're using

for pidx_str, glorys_data in glorys_cache.items():
    pidx = int(pidx_str)
    if pidx >= len(argo_df):
        continue
    
    row = argo_df.iloc[pidx]
    
    # Get features from Argo dataset
    features = [row[c] for c in FEATURE_COLS]
    if any(pd.isna(f) for f in features):
        continue
    
    # Get GLORYS temperature targets
    glorys_temp = glorys_data.get('temp', {})
    if not glorys_temp:
        continue
    
    # Extract at target depths
    target_temps = []
    valid = True
    for depth in TARGET_DEPTHS:
        val = glorys_temp.get(str(depth))
        if val is None:
            valid = False
            break
        try:
            target_temps.append(float(val))
        except:
            valid = False
            break
    
    if not valid or len(target_temps) != len(TARGET_DEPTHS):
        continue
    
    X_list.append(features)
    y_list.append(target_temps)
    profile_indices.append(pidx)

X = np.array(X_list)
y = np.array(y_list)
print(f'\nValid training samples: {len(X)}')

if len(X) < 50:
    print('ERROR: Too few samples to train')
else:
    # Split 80/20
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    te_profile_indices = [profile_indices[i] for i in te_idx]
    
    print(f'Train: {len(Xtr)}, Test: {len(Xte)}')
    
    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xtr, ytr)
    
    # Evaluate against GLORYS labels
    train_r2 = ridge.score(Xtr, ytr)
    test_r2 = ridge.score(Xte, yte)
    print(f'\nGLORYS-label metrics:')
    print(f'  Train R2: {train_r2:.4f}')
    print(f'  Test R2: {test_r2:.4f}')
    
    # Predictions on test set
    test_preds = ridge.predict(Xte)
    
    # Get REAL Argo targets for the same profiles
    argo_targets = []
    argo_match_count = 0
    
    for pidx in te_profile_indices:
        row = argo_df.iloc[pidx]
        
        # Get Argo temperature at target depths
        argo_temps = []
        valid = True
        for depth in TARGET_DEPTHS:
            col = f'T_{depth}m'
            val = row[col] if col in row.index else np.nan
            if pd.isna(val):
                valid = False
                break
            argo_temps.append(float(val))
        
        if valid:
            argo_targets.append(argo_temps)
            argo_match_count += 1
        else:
            argo_targets.append([np.nan] * len(TARGET_DEPTHS))
    
    argo_targets = np.array(argo_targets)
    print(f'\nArgo comparison: {argo_match_count}/{len(te_idx)} test samples have valid Argo targets')
    
    # Evaluate against Argo
    valid_mask = ~np.any(np.isnan(argo_targets), axis=1)
    if valid_mask.sum() > 0:
        preds = test_preds[valid_mask]
        targets = argo_targets[valid_mask]
        
        rmse = np.sqrt(np.mean((preds - targets)**2))
        mae = np.mean(np.abs(preds - targets))
        ss_res = np.sum((preds - targets)**2)
        ss_tot = np.sum((targets - np.mean(targets, axis=0))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        print(f'\n{"="*60}')
        print(f'GLORYS MODEL vs REAL ARGO')
        print(f'{"="*60}')
        print(f'Samples: {valid_mask.sum()}')
        print(f'RMSE: {rmse:.4f} C')
        print(f'MAE:  {mae:.4f} C')
        print(f'R2:   {r2:.4f}')
        
        # Per-depth breakdown
        rmse_per_depth = np.sqrt(np.mean((preds - targets)**2, axis=0))
        print(f'\nPer-depth RMSE:')
        for i, d in enumerate(TARGET_DEPTHS):
            print(f'  {d:5d}m: {rmse_per_depth[i]:.3f} C')
        
        # Compare with main model
        print(f'\n{"="*60}')
        print(f'COMPARISON WITH MAIN MODEL')
        print(f'{"="*60}')
        print(f'GLORYS-trained ({len(Xtr)} samples): RMSE={rmse:.3f} C, R2={r2:.4f}')
        print(f'Main Argo-trained (2883 samples): RMSE=1.064 C, R2=0.9797')
        
        # Save proof-of-concept model (Ridge coefficients only)
        torch.save({
            'ridge_coef': ridge.coef_.tolist(),
            'ridge_intercept': ridge.intercept_.tolist(),
            'n_samples': len(Xtr),
            'target_depths': TARGET_DEPTHS,
            'feature_cols': FEATURE_COLS,
            'glorys_rmse': float(rmse),
            'glorys_mae': float(mae),
            'glorys_r2': float(r2),
        }, 'models/checkpoints/ocean_embed_glorys_poc.pt')
        print(f'\nSaved: models/checkpoints/ocean_embed_glorys_poc.pt')
    else:
        print('No valid Argo comparison samples')

print(f'\n{"="*60}')
print('DONE - Main checkpoint untouched')
print(f'{"="*60}')
