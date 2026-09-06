"""Check full OceanEmbed project status — zero network requests."""
import json, torch
from pathlib import Path
import pandas as pd

print("=" * 60)
print("OCEANEMBED PROJECT STATUS")
print("=" * 60)

# 1. HYCOM v5 cache
d = Path("data/raw/surface_cache/hycom_v5")
n = sss = uv = tmp = full = 0
for p in d.glob("profile_*.json"):
    try:
        r = json.load(open(p)); n += 1
        if r.get("sss") is not None: sss += 1
        if r.get("u") is not None and r.get("v") is not None: uv += 1
        if r.get("temp") is not None and len(r["temp"]) > 0: tmp += 1
        if r.get("sss") is not None and r.get("u") is not None and r.get("v") is not None and r.get("temp") is not None:
            full += 1
    except: pass

print("\n1. HYCOM CACHE (local, per-profile JSON):")
print(f"   Total files:     {n}")
print(f"   Valid SSS:       {sss} ({100*sss/2992:.1f}%)")
print(f"   Valid currents:  {uv} ({100*uv/2992:.1f}%)")
print(f"   Valid temp:      {tmp} ({100*tmp/2992:.1f}%)")
print(f"   Full (all 4):    {full} ({100*full/2992:.1f}%)")

# 2. Surface caches
print("\n2. SURFACE DATA CACHES:")
for fn in ["sst_values.json", "ssh_values.json", "wind_values.json", "current_values.json"]:
    p = Path("data/raw/surface_cache") / fn
    if p.exists():
        try:
            data = json.load(open(p))
            print(f"   {fn}: {len(data)} entries")
        except Exception as e:
            print(f"   {fn}: CORRUPT ({e})")
    else:
        print(f"   {fn}: NOT FOUND")

# 3. Trained model
ckpt = torch.load("models/checkpoints/ocean_embed_real_best.pt", map_location="cpu", weights_only=False)
n_params = sum(p.numel() for p in ckpt["model_state_dict"].values())
print(f"\n3. TRAINED MODEL:")
print(f"   Features:  {ckpt['features']}")
print(f"   Depths:    {ckpt['depth_values']}")
print(f"   Params:    {n_params:,}")
print(f"   Val loss:  {ckpt['training_info'].get('best_val_loss', 'N/A')}")

# 4. Dataset
df = pd.read_parquet("data/processed/oceanembed_real_dataset.parquet")
print(f"\n4. TRAINING DATASET:")
print(f"   Rows: {len(df)}")
print(f"   SST available:  {df['sst'].notna().sum()}/{len(df)}")
print(f"   SSH available:  {df['ssh'].notna().sum()}/{len(df)}")
print(f"   u10 available:  {df['u10'].notna().sum()}/{len(df)}")
print(f"   v10 available:  {df['v10'].notna().sum()}/{len(df)}")

# 5. Reports
print(f"\n5. REPORTS:")
for fn in ["prompt_c_report.json", "prompt_c_report.md", "VALIDATION_REPORT.md"]:
    p = Path("data/reports") / fn
    print(f"   {fn}: {'EXISTS' if p.exists() else 'MISSING'}")

# 6. Backend + Frontend
print(f"\n6. APPLICATION:")
for fn in ["backend/app.py", "static/js/app.js", "static/index.html"]:
    print(f"   {fn}: {'OK' if Path(fn).exists() else 'MISSING'}")

# 7. Script files
print(f"\n7. SCRIPTS:")
for fn in ["scripts/prompt_c_local.py", "scripts/hycom_extract_v5.py"]:
    print(f"   {fn}: {'OK' if Path(fn).exists() else 'MISSING'}")
