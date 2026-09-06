"""
OceanEmbed Configuration
Central configuration for the project - CNN Ocean Embedding Framework
Matches SIH26066 PPT: 7 surface inputs, spatial CNN, 15-depth output
"""

# Geographic bounds for North Indian Ocean domain
REGION = {
    "lat_min": 0.0,
    "lat_max": 25.0,
    "lon_min": 40.0,
    "lon_max": 100.0,
}

# Grid configuration (0.25° × 0.25°)
GRID_RESOLUTION = 0.25  # degrees
GRID_PATCH_SIZE = 8      # spatial patch size for CNN input (8×8 cells = 2°×2°)

# Depth levels (meters) - 13 model-predicted + 2 derived = 15 total
# 0m = SST (measured), 5m = interpolated between SST and T_10m
MODEL_DEPTH_LEVELS = [
    10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
]
DERIVED_DEPTH_LEVELS = [0, 5]
DEPTH_LEVELS = DERIVED_DEPTH_LEVELS + MODEL_DEPTH_LEVELS

NUM_MODEL_DEPTHS = len(MODEL_DEPTH_LEVELS)  # 13
NUM_DISPLAY_DEPTHS = len(DEPTH_LEVELS)      # 15
NUM_DEPTH_LEVELS = NUM_MODEL_DEPTHS  # backward compat alias

# 7 Surface input channels (PPT requirement)
SURFACE_INPUTS = [
    "sst",         # Sea Surface Temperature
    "sss",         # Sea Surface Salinity (climatological fallback)
    "ssh",         # Sea Surface Height Anomaly
    "u10",         # Zonal wind speed
    "v10",         # Meridional wind speed
    "current_u",   # Ocean current u-component
    "current_v",   # Ocean current v-component
]

# Legacy feature list (for backward compat with 1D model)
FEATURES = ["latitude", "longitude"] + SURFACE_INPUTS
NUM_FEATURES = len(FEATURES)
NUM_INPUT_CHANNELS = len(SURFACE_INPUTS)  # 7 channels for spatial CNN

# Spatial CNN configuration (RTX 3050 compatible)
SPATIAL_MODEL_CONFIG = {
    "in_channels": NUM_INPUT_CHANNELS,   # 7
    "out_depths": NUM_MODEL_DEPTHS,      # 13
    "base_channels": 32,
    "patch_size": GRID_PATCH_SIZE,        # 8×8
    "use_gradient_checkpointing": True,   # VRAM savings for RTX 3050
}

# Legacy 1D model config (still available as fallback)
MODEL_CONFIG = {
    "input_dim": NUM_FEATURES,
    "output_dim": NUM_MODEL_DEPTHS,
    "hidden_dims": [128, 256, 512, 256, 128],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "num_epochs": 100,
    "patience": 15,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
}

# Data paths
DATA_DIR = "data"
SAMPLE_DIR = "sample_data"
MODEL_DIR = "models"
CHECKPOINT_DIR = "models/checkpoints"
GRIDDED_DIR = "data/gridded"

# Checkpoint paths
BEST_CHECKPOINT = f"{CHECKPOINT_DIR}/ocean_embed_real_best.pt"
SPATIAL_CHECKPOINT = f"{CHECKPOINT_DIR}/ocean_embed_spatial_7ch.pt"
BACKUP_CHECKPOINT = f"{CHECKPOINT_DIR}/BACKUP_ocean_embed_real_best_13level.pt"

# For static/fallback dataset
STATIC_DATA_FILE = "sample_data/ocean_embed_dataset.npz"
METRICS_FILE = "sample_data/metrics.json"

# Data source metadata
DATA_SOURCES = {
    "sst": {"source": "OISST / CMEMS", "coverage": "98.7%", "status": "real"},
    "sss": {"source": "WOA18 Climatology", "coverage": "100%", "status": "climatological_fallback"},
    "ssh": {"source": "DUACS / CMEMS", "coverage": "100%", "status": "real"},
    "u10": {"source": "ERA5 / Open-Meteo", "coverage": "99.7%", "status": "real"},
    "v10": {"source": "ERA5 / Open-Meteo", "coverage": "99.7%", "status": "real"},
    "current_u": {"source": "Interpolated (nearest-neighbor)", "coverage": "100%", "status": "interpolated_fallback"},
    "current_v": {"source": "Interpolated (nearest-neighbor)", "coverage": "100%", "status": "interpolated_fallback"},
}

# Server
HOST = "0.0.0.0"
PORT = 8000
