"""
OceanEmbed Configuration
Central configuration for the project - Indian Ocean domain prototype
"""

# Geographic bounds for Indian Ocean domain
REGION = {
    "lat_min": -30.0,
    "lat_max": 30.0,
    "lon_min": 30.0,
    "lon_max": 120.0,
}

# Depth levels (meters) - 13 levels from 10m to 1000m
# 0m and 5m are SST-anchored DERIVED values for display
MODEL_DEPTH_LEVELS = [
    10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
]

# Display levels: 0m (SST) + 5m (interpolated) + model outputs
DEPTH_LEVELS = [0, 5] + MODEL_DEPTH_LEVELS

NUM_DEPTH_LEVELS = len(MODEL_DEPTH_LEVELS)  # Model outputs 13 levels
NUM_DISPLAY_LEVELS = len(DEPTH_LEVELS)  # Display shows 15 levels

# Surface features (input channels)
# NOTE: SSS was dropped from the real-data model because SMOS 403 Forbidden.
FEATURES = [
    "latitude",
    "longitude",
    "sst",    # Sea Surface Temperature
    "ssh",    # Sea Surface Height Anomaly
    "u10",    # Zonal wind speed
    "v10",    # Meridional wind speed
]

NUM_FEATURES = len(FEATURES)

# Model hyperparameters
MODEL_CONFIG = {
    "input_dim": NUM_FEATURES,
    "output_dim": NUM_DEPTH_LEVELS,
    "hidden_dims": [128, 256, 512, 256, 128],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "num_epochs": 100,
    "patience": 15,  # Early stopping patience
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
}

# Data paths
DATA_DIR = "data"
SAMPLE_DIR = "sample_data"
MODEL_DIR = "models"
CHECKPOINT_DIR = "models/checkpoints"

# For static/fallback dataset
STATIC_DATA_FILE = "sample_data/ocean_embed_dataset.npz"
METRICS_FILE = "sample_data/metrics.json"

# Server
HOST = "0.0.0.0"
PORT = 8000
