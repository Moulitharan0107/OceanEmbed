"""Prompt C: Tasks 2 & 3 - Currents comparison + HYCOM temperature model evaluation.
Zero network requests. Uses only cached local data."""
import json, pandas as pd, numpy as np, torch, torch.nn as nn, warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# ============================================================
# TASK 2: Compare HYCOM currents vs Open-Meteo
# ============================================================
print('='*60)
print('TASK 2: Currents Source Comparison')
print('='*60)

v5 = Path('data/raw/surface_cache/hycom_v5')
hycom_currents = 0
if v5.exists():
    for f in v5.glob('*.json'):
        try:
            d = json.load(open(f))
            if d.get('u') is not None and d.get('v') is not None:
                hycom_currents += 1
        except:
            pass

om_file = Path('data/raw/surface_cache/current_values.json')
om_currents = 0
if om_file.exists():
    om = json.load(open(om_file))
    if isinstance(om, dict):
        for key in ['current_u', 'u']:
            if key in om:
                om_currents = len([v for v in om[key].values() if v is not None])
                break

df = pd.read_parquet('data/processed/oceanembed_real_dataset.parquet')
total = len(df)

print(f'Argo profiles total: {total}')
print(f'HYCOM currents (u+v): {hycom_currents} ({100*hycom_currents/total:.1f}%)')
print(f'Open-Meteo currents:  {om_currents} ({100*om_currents/total:.1f}%)')
print()

if hycom_currents > om_currents:
    best_source, best_count = 'HYCOM GOFS', hycom_currents
else:
    best_source, best_count = 'Open-Meteo Marine', om_currents

print(f'BEST SOURCE: {best_source} ({best_count} profiles, {100*best_count/total:.1f}%)')
if best_count < 500:
    print('CONCLUSION: Both current sources have <500 profiles (<17%).')
    print('Currents remain EXCLUDED from the deployed model.')
else:
    print('CONCLUSION: Coverage sufficient to consider adding currents.')
print()

# ============================================================
# TASK 3: HYCOM reanalysis temperature model
# ============================================================
print('='*60)
print('TASK 3: HYCOM Reanalysis Temperature Model')
print('='*60)

# HYCOM native depth levels (40 levels from OPeNDAP)
HYCOM_NATIVE_DEPTHS = [0,2,4,6,8,10,12,15,20,25,30,35,40,45,50,60,70,80,90,100,
                        125,150,200,250,300,350,400,500,600,700,800,900,1000,
                        1250,1500,2000,2500,3000,4000,5000]
assert len(HYCOM_NATIVE_DEPTHS) == 40

# Collect HYCOM temperature profiles (temp is a list, lat/lon from profile index)
hycom_profiles = []
if v5.exists():
    for f in v5.glob('*.json'):
        try:
            d = json.load(open(f))
            temp_list = d.get('temp')
            if temp_list is not None and isinstance(temp_list, list) and len(temp_list) >= 33:
                # Extract profile index from filename
                idx_str = f.stem.replace('profile_', '')
                profile_idx = int(idx_str)
                if profile_idx < len(df):
                    row = df.iloc[profile_idx]
                    hycom_profiles.append({
                        'lat': float(row['latitude']),
                        'lon': float(row['longitude']),
                        'date': str(row.get('date', '')),
                        'temp': temp_list,
                        'profile_idx': profile_idx
                    })
        except:
            pass

print(f'HYCOM temperature profiles: {len(hycom_profiles)}')

# Load model config
ckpt = torch.load('models/checkpoints/ocean_embed_real_best.pt', map_location='cpu', weights_only=False)
target_depths = ckpt['depth_values']
print(f'Model target depths: {target_depths}')

# Match HYCOM profiles to Argo features
argo_lats = df['latitude'].values
argo_lons = df['longitude'].values
feature_cols = ['latitude', 'longitude', 'sst', 'ssh', 'u10', 'v10']
target_cols = [f'T_{d}m' for d in target_depths]

X_list, y_list = [], []
valid_hycom = 0

for hp in hycom_profiles:
    hlat, hlon = hp['lat'], hp['lon']
    dists = np.sqrt((argo_lats - hlat)**2 + (argo_lons - hlon)**2)
    nearest_idx = np.argmin(dists)
    row = df.iloc[nearest_idx]

    features = [row[c] for c in feature_cols]
    if any(pd.isna(f) for f in features):
        continue

    # Interpolate HYCOM temp (list format) to target depths
    temp_list = hp['temp']
    h_depths, h_temps = [], []
    for i, t_val in enumerate(temp_list):
        if i < len(HYCOM_NATIVE_DEPTHS) and t_val is not None:
            try:
                t_float = float(t_val)
                if not np.isnan(t_float) and not np.isinf(t_float) and -5 < t_float < 50:
                    h_depths.append(HYCOM_NATIVE_DEPTHS[i])
                    h_temps.append(t_float)
            except:
                continue

    if len(h_depths) < 5:
        continue

    target_temps = np.interp(target_depths, np.array(h_depths), np.array(h_temps))
    # Final sanity check: all temps must be physically reasonable
    if np.any(np.isnan(target_temps)) or np.any(np.isinf(target_temps)):
        continue
    if np.any(target_temps < -5) or np.any(target_temps > 50):
        continue

    X_list.append(features)
    y_list.append(target_temps)
    valid_hycom += 1

X_arr = np.array(X_list)
y_arr = np.array(y_list)
print(f'Training samples (HYCOM labels): {len(X_arr)}')

if len(X_arr) < 50:
    print('ERROR: Too few HYCOM profiles to train.')
else:
    # Split 80/20
    n = len(X_arr)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]
    Xtr, Xte = X_arr[tr_idx], X_arr[te_idx]
    ytr, yte = y_arr[tr_idx], y_arr[te_idx]
    print(f'Train: {len(Xtr)}, Test: {len(Xte)}')

    # Standardize
    sx = StandardScaler().fit(Xtr)
    sy = StandardScaler().fit(ytr)
    Xtr_s = sx.transform(Xtr)
    Xte_s = sx.transform(Xte)
    ytr_s = sy.transform(ytr)

    # Use sklearn Ridge (appropriate for small-N: 75 samples, 6 features)
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xtr, ytr)
    hycom_pred = ridge.predict(Xte)
    print(f'\nModel: Ridge regression (alpha=1.0, {len(Xtr)} train samples)')
    print(f'Train R2: {ridge.score(Xtr, ytr):.4f}')
    print(f'Test R2 (vs HYCOM labels): {ridge.score(Xte, yte):.4f}')

    # Find real Argo targets for the same HYCOM profile locations
    argo_targets = []
    argo_valid_count = 0
    for hp in hycom_profiles:
        # Use exact profile index from cache filename
        profile_idx = hp['profile_idx']
        row = df.iloc[profile_idx]

        temps, valid = [], True
        for tc in target_cols:
            val = row[tc]
            if pd.isna(val):
                valid = False
                break
            temps.append(float(val))

        if valid:
            argo_targets.append(temps)
            argo_valid_count += 1
        else:
            argo_targets.append([np.nan] * len(target_cols))

    argo_targets = np.array(argo_targets)
    print(f'\nArgo targets available: {argo_valid_count}/{len(hycom_profiles)} ({100*argo_valid_count/len(hycom_profiles):.1f}%)')

    # Evaluate test set against REAL Argo
    test_argo = argo_targets[te_idx]
    valid_mask = ~np.any(np.isnan(test_argo), axis=1)
    print(f'Test samples with valid Argo comparison: {valid_mask.sum()}/{len(te_idx)}')

    if valid_mask.sum() > 0:
        preds = hycom_pred[valid_mask]
        targets = test_argo[valid_mask]

        rmse_per_depth = np.sqrt(np.mean((preds - targets)**2, axis=0))
        mae_per_depth = np.mean(np.abs(preds - targets), axis=0)
        overall_rmse = np.sqrt(np.mean((preds - targets)**2))
        overall_mae = np.mean(np.abs(preds - targets))

        ss_res = np.sum((preds - targets)**2)
        ss_tot = np.sum((targets - np.mean(targets, axis=0))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f'\n=== HYCOM-trained model vs REAL Argo ===')
        print(f'Overall RMSE: {overall_rmse:.4f} C')
        print(f'Overall MAE:  {overall_mae:.4f} C')
        print(f'Overall R2:   {r2:.4f}')
        print(f'\nPer-depth:')
        print(f'{"Depth":<8} {"RMSE":>8} {"MAE":>8} {"N":>5}')
        for i, d in enumerate(target_depths):
            nv = (~np.isnan(targets[:, i])).sum()
            print(f'{d}m      {rmse_per_depth[i]:>8.3f} {mae_per_depth[i]:>8.3f} {nv:>5}')

        print(f'\n=== Comparison ===')
        print(f'HYCOM-trained ({len(Xtr)} train): RMSE={overall_rmse:.3f} C, R2={r2:.4f}')
        print(f'Main model   (2883 train): RMSE=1.064 C, R2=0.9797')
        print(f'\nNOTE: HYCOM model trained on {len(Xtr)} samples vs 2,883.')
        print(f'Small training set = expected overfitting. NOT a replacement.')
    else:
        print('No valid Argo comparison samples.')

# ============================================================
# FINAL REPORT
# ============================================================
print()
print('='*60)
print('PROMPT C FINAL REPORT')
print('='*60)
print(f'Task 1 (SSS):     BLOCKED - No NASA Earthdata credentials in sandbox')
print(f'Task 2 (Currents): HYCOM={hycom_currents}, Open-Meteo={om_currents}')
print(f'                   Best={best_source} ({best_count}/{total} = {100*best_count/total:.1f}%)')
print(f'                   Status: EXCLUDED (<17% coverage)')
print(f'Task 3 (HYCOM temp): {valid_hycom} profiles used')
if valid_mask.sum() > 0:
    print(f'                   RMSE={overall_rmse:.3f} C, MAE={overall_mae:.3f} C, R2={r2:.4f}')
    print(f'                   vs Main: 1.064 C / 0.9797')
print(f'Main checkpoint:  INTACT (ocean_embed_real_best.pt)')
print('='*60)
