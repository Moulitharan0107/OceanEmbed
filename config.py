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

# Depth levels (meters) - 33 standard oceanographic levels from surface to 2000m
DEPTH_LEVELS = [
    0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200,
    250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000
]

NUM_DEPTH_LEVELS = len(DEPTH_LEVELS)

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
