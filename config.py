import os
from pathlib import Path
import torch

class Config:
    # ==========================================================================
    # 1. PROJECT ROOT AND PATHS
    # ==========================================================================
    # Automatically get the project root directory
    PROJECT_ROOT = Path(__file__).parent.absolute()
    
    # Data Directories
    DATA_DIR = PROJECT_ROOT / "data"
    PATCHES_DIR = DATA_DIR / "patches_final"
    UNIFIED_PROB_DIR = DATA_DIR / "unified_probability_matrices"
    
    # Model Directories
    MODELS_DIR = PROJECT_ROOT / "models"
    GATE_DIR = PROJECT_ROOT / "gate_models"
    
    # Specific Model Paths
    AUTOENCODER_PATH = MODELS_DIR / 'best_model_aoi.pth'
    IFOREST_PATH = MODELS_DIR / 'isolation_forest_model_128dim.pkl'
    KMEANS_PATH = MODELS_DIR / 'kmeans_model_128dim.pkl'
    
    # GATE Paths
    GATE_MODEL_PKL = GATE_DIR / 'gate_mlp_model.pkl'
    GATE_SCALER_PATH = GATE_DIR / 'gate_scaler.pkl'
    ARCH_EMBEDDINGS_CSV = GATE_DIR / 'Arch_embedding_only_128dim.csv'

    # Raw Data Sources (For preparedata.py) - These might remain absolute if external
    HYDRO_BASE = Path('/path/to/hydro/data') 
    SENTINEL_BASE = Path('/path/to/satellite/data')
    LIDAR_BASE = Path('/path/to/lidar/data')
    # Inside the Config class, add:

# Model-specific settings
    ENCODER_DIM = 128

    # Update these paths (they're slightly different from what you have):
    UNIFIED_PROB_DIR = DATA_DIR / "unified_probability_matrices_with_gate"

    # ==========================================================================
    # 2. HYPERPARAMETERS
    # ==========================================================================
    PATCH_SIZE = 64
    CHANNELS = 7  # DTM, Slope, Roughness, NDVI, NDWI, FlowAcc, FlowDir
    EMBEDDING_DIM = 128
    BATCH_SIZE = 32
    
    # ==========================================================================
    # 3. COMPUTE SETTINGS
    # ==========================================================================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    
    # ==========================================================================
    # 4. VISUALIZATION SETTINGS
    # ==========================================================================
    THEME_PRIMARY = "blue"
    THEME_SECONDARY = "slate"

config = Config()

# Create directories if they don't exist
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
config.GATE_DIR.mkdir(parents=True, exist_ok=True)