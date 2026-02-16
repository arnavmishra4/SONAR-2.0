"""
batch_generate_matrices_with_gate.py - Enhanced with GATE predictions
========================================================================
Generates 5-channel unified probability matrices:
- Channel 0: Autoencoder anomaly scores
- Channel 1: Isolation Forest probabilities  
- Channel 2: K-Means probabilities
- Channel 3: Archaeological similarity scores
- Channel 4: GATE model final predictions (NEW!)

Note: This version handles PyTorch import errors gracefully
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import sys
from tqdm import tqdm
import joblib
from config import config as app_config

# Try to import PyTorch, but don't fail if it's not available
PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
    print("✓ PyTorch available")
except (ImportError, OSError) as e:
    print(f"⚠️  PyTorch not available: {e}")
    print("   Will use scikit-learn GATE model instead")
    torch = None
    nn = None

# Import from your existing files
from utils import (
    ResUNetAutoencoder,
    ResUNetEncoder,
    load_patches,
    load_model,
    normalize_scores,
    broadcast_patch_scores_to_pixels,
    save_unified_probability_matrix,
    compute_autoencoder_probabilities,
    compute_iforest_probabilities,
)

from arch_similarity_utils import compute_arch_similarity_channel

try:
    from utils import load_kmeans_model, compute_kmeans_probabilities
    KMEANS_AVAILABLE = True
except ImportError:
    KMEANS_AVAILABLE = False
    print("⚠️  K-Means functions not available, will skip Channel 2")
    
class BatchConfig:
    """Wrapper for app config with batch-specific settings"""
    def __init__(self):
        # Import all paths from app config
        self.PATCHES_DIR = app_config.PATCHES_DIR
        self.AUTOENCODER_PATH = app_config.AUTOENCODER_PATH
        self.IFOREST_MODEL_PATH = app_config.IFOREST_PATH
        self.KMEANS_MODEL_PATH = app_config.KMEANS_PATH
        self.ARCH_EMBEDDINGS_CSV = app_config.ARCH_EMBEDDINGS_CSV
        self.GATE_MODEL_PKL = app_config.GATE_MODEL_PKL
        self.GATE_SCALER_PATH = app_config.GATE_SCALER_PATH
        self.GATE_MODEL_PT = app_config.GATE_DIR / 'gate_mlp_model.pt'  # Add this to main config if you have it
        self.UNIFIED_PROB_DIR = app_config.UNIFIED_PROB_DIR
        
        # Model parameters from app config
        self.EMBEDDING_DIM = app_config.EMBEDDING_DIM
        self.BATCH_SIZE = app_config.BATCH_SIZE
        
        # Batch-specific settings (not in main config)
        self.USE_SIGMOID_SCORES = True
        self.PREFER_PYTORCH_GATE = True
        self.CHANNEL_NAMES = ['prob_autoencoder', 'prob_iforest', 'prob_kmeans', 
                              'arch_similarity', 'gate_prediction']

# ========================
# GATE Model Definition (PyTorch - optional)
# ========================
if PYTORCH_AVAILABLE:
    class GateMLP(nn.Module):
        """PyTorch version of the GATE MLP model"""
        def __init__(self):
            super(GateMLP, self).__init__()
            self.fc1 = nn.Linear(4, 16)
            self.fc2 = nn.Linear(16, 8)
            self.fc3 = nn.Linear(8, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x


def load_gate_model_pytorch(model_path: str, device) -> Tuple:
    """Load PyTorch GATE model (only if PyTorch is available)"""
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = GateMLP().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler_mean = checkpoint['scaler_mean']
    scaler_scale = checkpoint['scaler_scale']
    
    return model, scaler_mean, scaler_scale


def load_gate_model_sklearn(model_path: str, scaler_path: str) -> Tuple:
    """Load scikit-learn GATE model"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_
    
    return model, scaler_mean, scaler_scale, scaler


def compute_gate_predictions_sklearn(
    prob_autoencoder: np.ndarray,
    prob_iforest: np.ndarray, 
    prob_kmeans: np.ndarray,
    arch_similarity: np.ndarray,
    gate_model,
    scaler
) -> np.ndarray:
    """
    Compute GATE predictions using scikit-learn model
    
    Args:
        prob_autoencoder: (num_patches, 64, 64) autoencoder scores
        prob_iforest: (num_patches, 64, 64) iforest scores
        prob_kmeans: (num_patches, 64, 64) kmeans scores
        arch_similarity: (num_patches, 64, 64) similarity scores
        gate_model: Trained sklearn GATE model
        scaler: Fitted StandardScaler
    
    Returns:
        gate_predictions: (num_patches, 64, 64) GATE probability predictions
    """
    num_patches = prob_autoencoder.shape[0]
    
    # Average pool each channel to get per-patch features
    feat1 = prob_autoencoder.mean(axis=(1, 2))  # Autoencoder
    feat2 = prob_iforest.mean(axis=(1, 2))      # IForest
    feat3 = prob_kmeans.mean(axis=(1, 2))       # KMeans
    feat4 = arch_similarity.mean(axis=(1, 2))   # Arch similarity
    
    # Stack features: (num_patches, 4)
    features = np.stack([feat1, feat2, feat3, feat4], axis=1)
    
    # Normalize features using the fitted scaler
    features_scaled = scaler.transform(features)
    
    # Sklearn inference - get probability of class 1 (archaeological)
    # In compute_gate_predictions_sklearn():
    predictions = gate_model.predict_proba(features_scaled)[:, 1]
    binary_predictions = (predictions > 0.5).astype(float)  # 0 or 1
    gate_predictions = broadcast_patch_scores_to_pixels(binary_predictions, patch_size=64)  # (num_patches,)
    
    return gate_predictions


def compute_gate_predictions_pytorch(
    prob_autoencoder: np.ndarray,
    prob_iforest: np.ndarray, 
    prob_kmeans: np.ndarray,
    arch_similarity: np.ndarray,
    gate_model,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    device
) -> np.ndarray:
    """
    Compute GATE predictions using PyTorch model
    
    Returns:
        gate_predictions: (num_patches, 64, 64) GATE probability predictions
    """
    num_patches = prob_autoencoder.shape[0]
    
    # Average pool each channel to get per-patch features
    feat1 = prob_autoencoder.mean(axis=(1, 2))
    feat2 = prob_iforest.mean(axis=(1, 2))
    feat3 = prob_kmeans.mean(axis=(1, 2))
    feat4 = arch_similarity.mean(axis=(1, 2))
    
    # Stack features: (num_patches, 4)
    features = np.stack([feat1, feat2, feat3, feat4], axis=1)
    
    # Normalize features
    features_scaled = (features - scaler_mean) / scaler_scale
    
    # PyTorch inference
    features_tensor = torch.FloatTensor(features_scaled).to(device)
    
    with torch.no_grad():
        predictions = gate_model(features_tensor).cpu().numpy().squeeze()  # (num_patches,)
    
    # Broadcast predictions back to pixel level
    gate_predictions = broadcast_patch_scores_to_pixels(predictions, patch_size=64)
    
    return gate_predictions

def discover_aois(patches_dir: Path) -> List[str]:
    """Discover all AOI names from patches directory"""
    all_patch_files = sorted(list(patches_dir.glob("AOI_*_all_patches.npz")))
    aoi_names = [f.stem.replace('_all_patches', '') for f in all_patch_files]
    return aoi_names


def check_existing_matrices(aoi_names: List[str], output_dir: Path) -> tuple:
    """Check which matrices already exist"""
    existing = []
    missing = []
    
    for aoi_name in aoi_names:
        matrix_path = output_dir / f"{aoi_name}_unified_prob_matrix.npz"
        if matrix_path.exists():
            existing.append(aoi_name)
        else:
            missing.append(aoi_name)
    
    return existing, missing


def generate_single_matrix_with_gate(
    aoi_name: str,
    config: BatchConfig,
    device,
    autoencoder,
    encoder,
    iforest,
    scaler_iforest,
    gate_model,
    gate_scaler_or_params,
    use_pytorch_gate: bool,
    kmeans=None,
    scaler_kmeans=None
):
    """Generate unified probability matrix with GATE predictions for a single AOI"""
    
    # Load patches
    all_patches_file = config.PATCHES_DIR / f"{aoi_name}_all_patches.npz"
    if not all_patches_file.exists():
        raise FileNotFoundError(f"Patches not found: {all_patches_file}")
    
    patches, metadata = load_patches(all_patches_file)
    num_patches = len(patches)
    
    # Initialize unified matrix: (num_patches, 64, 64, 5) - NOW 5 CHANNELS!
    unified_matrix = np.zeros((num_patches, 64, 64, 5), dtype=np.float32)
    
    # Channel 0: Autoencoder
    prob_scores_ae, pixel_scores_ae, reconstructed_patches = compute_autoencoder_probabilities(
        model=autoencoder,
        patches=patches,
        metadata=metadata,
        device=device,
        batch_size=config.BATCH_SIZE,
        use_sigmoid_scores=config.USE_SIGMOID_SCORES
    )
    
    pixel_scores_ae_norm = normalize_scores(pixel_scores_ae)
    unified_matrix[:, :, :, 0] = pixel_scores_ae_norm
    
    del reconstructed_patches
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Channel 1: Isolation Forest
    prob_scores_if, embeddings_if, predictions_if = compute_iforest_probabilities(
        encoder, iforest, scaler_iforest, patches, metadata, device, config.BATCH_SIZE
    )
    
    prob_scores_if_pixels = broadcast_patch_scores_to_pixels(prob_scores_if, patch_size=64)
    unified_matrix[:, :, :, 1] = prob_scores_if_pixels
    
    del embeddings_if, predictions_if
    
    # Channel 2: K-Means (if available)
    if KMEANS_AVAILABLE and kmeans is not None:
        try:
            prob_scores_km, similarity_scores_km, cluster_assignments = compute_kmeans_probabilities(
                encoder, kmeans, scaler_kmeans, patches, metadata, device, config.BATCH_SIZE
            )
            prob_scores_km_pixels = broadcast_patch_scores_to_pixels(prob_scores_km, patch_size=64)
            unified_matrix[:, :, :, 2] = prob_scores_km_pixels
        except Exception as e:
            print(f"   ⚠️  K-Means failed: {e}, setting Channel 2 to zeros")
            unified_matrix[:, :, :, 2] = 0.0
    else:
        unified_matrix[:, :, :, 2] = 0.0
    
    # Channel 3: Archaeological Similarity
    similarity_channel, analysis_dict = compute_arch_similarity_channel(
        encoder=encoder,
        patches=patches,
        metadata=metadata,
        arch_csv_path=str(config.ARCH_EMBEDDINGS_CSV),
        device=device,
        batch_size=config.BATCH_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        patch_size=64
    )
    
    unified_matrix[:, :, :, 3] = similarity_channel
    
    # Channel 4: GATE Model Predictions (NEW!)
    print(f"   Computing GATE predictions...")
    
    if use_pytorch_gate:
        # PyTorch version
        gate_predictions = compute_gate_predictions_pytorch(
            prob_autoencoder=unified_matrix[:, :, :, 0],
            prob_iforest=unified_matrix[:, :, :, 1],
            prob_kmeans=unified_matrix[:, :, :, 2],
            arch_similarity=unified_matrix[:, :, :, 3],
            gate_model=gate_model,
            scaler_mean=gate_scaler_or_params[0],
            scaler_scale=gate_scaler_or_params[1],
            device=device
        )
    else:
        # Sklearn version
        gate_predictions = compute_gate_predictions_sklearn(
            prob_autoencoder=unified_matrix[:, :, :, 0],
            prob_iforest=unified_matrix[:, :, :, 1],
            prob_kmeans=unified_matrix[:, :, :, 2],
            arch_similarity=unified_matrix[:, :, :, 3],
            gate_model=gate_model,
            scaler=gate_scaler_or_params  # scaler object
        )
    
    unified_matrix[:, :, :, 4] = gate_predictions
    
    # Save with updated channel names
    save_unified_probability_matrix(
        aoi_name=aoi_name,
        unified_matrix=unified_matrix,
        metadata=metadata,
        save_dir=config.UNIFIED_PROB_DIR,
        channel_names=config.CHANNEL_NAMES
    )
    
    return unified_matrix.shape


def main():
    """Main batch generation pipeline with GATE predictions"""
    
    config = BatchConfig()
    
    # Determine device
    if PYTORCH_AVAILABLE:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'  # String for sklearn compatibility
    
    print(f"\n{'#'*80}")
    print(f"BATCH UNIFIED PROBABILITY MATRIX GENERATION (WITH GATE MODEL)")
    print(f"{'#'*80}")
    print(f"Device: {device}")
    print(f"PyTorch available: {PYTORCH_AVAILABLE}")
    print(f"Embedding dimension: {config.EMBEDDING_DIM}")
    print(f"Patches directory: {config.PATCHES_DIR}")
    print(f"Output directory: {config.UNIFIED_PROB_DIR}")
    print(f"{'#'*80}\n")
    
    # Create output directory
    config.UNIFIED_PROB_DIR.mkdir(exist_ok=True, parents=True)
    
    # Verify critical files
    critical_files = [
        config.AUTOENCODER_PATH,
        config.IFOREST_MODEL_PATH,
        config.ARCH_EMBEDDINGS_CSV,
        config.GATE_MODEL_PKL,
        config.GATE_SCALER_PATH
    ]
    
    for file_path in critical_files:
        if not file_path.exists():
            print(f"❌ CRITICAL FILE MISSING: {file_path}")
            sys.exit(1)
    
    print("✓ All critical files found\n")
    
    # Discover AOIs
    print("📂 Discovering AOIs...")
    all_aoi_names = discover_aois(config.PATCHES_DIR)
    print(f"   Found {len(all_aoi_names)} AOIs")
    
    # Check existing matrices
    existing, missing = check_existing_matrices(all_aoi_names, config.UNIFIED_PROB_DIR)
    print(f"\n📊 Status:")
    print(f"   Already generated: {len(existing)}")
    print(f"   Missing: {len(missing)}")
    
    if len(missing) == 0:
        print("\n✅ All matrices already generated!")
        return
    
    # Ask user
    print(f"\n🚀 Will generate {len(missing)} missing matrices with GATE predictions")
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Load models ONCE
    print(f"\n📥 Loading models...")
    
    # 1. Load Autoencoder
    print("   Loading autoencoder...")
    if PYTORCH_AVAILABLE:
        autoencoder = ResUNetAutoencoder(in_channels=7).to(device)
        autoencoder.load_state_dict(torch.load(config.AUTOENCODER_PATH, map_location=device))
        autoencoder.eval()
        print(f"   ✓ Autoencoder loaded (latent dim: 256)")
    else:
        print("   ❌ Cannot load autoencoder without PyTorch")
        sys.exit(1)
    
    # 2. Load Encoder for IForest and K-Means
    print(f"   Loading encoder (embedding_dim={config.EMBEDDING_DIM})...")
    encoder = ResUNetEncoder(in_channels=7, embedding_dim=config.EMBEDDING_DIM).to(device)
    encoder.load_from_autoencoder(str(config.AUTOENCODER_PATH))
    encoder.eval()
    print(f"   ✓ Encoder loaded and ready")
    
    # 3. Load Isolation Forest
    print("   Loading Isolation Forest...")
    iforest, scaler_iforest = load_model(str(config.IFOREST_MODEL_PATH))
    
    # 4. Load K-Means (optional)
    kmeans, scaler_kmeans = None, None
    if KMEANS_AVAILABLE and config.KMEANS_MODEL_PATH.exists():
        print("   Loading K-Means...")
        try:
            kmeans, scaler_kmeans = load_kmeans_model(str(config.KMEANS_MODEL_PATH))
        except Exception as e:
            print(f"   ⚠️  Could not load K-Means: {e}")
    
    # 5. Load GATE Model (NEW!)
    print("   Loading GATE model...")
    use_pytorch_gate = False
    gate_scaler_or_params = None
    
    # Try PyTorch first if available and preferred
    if PYTORCH_AVAILABLE and config.PREFER_PYTORCH_GATE and config.GATE_MODEL_PT.exists():
        try:
            gate_model, scaler_mean, scaler_scale = load_gate_model_pytorch(
                str(config.GATE_MODEL_PT), device
            )
            gate_scaler_or_params = (scaler_mean, scaler_scale)
            use_pytorch_gate = True
            print(f"   ✓ GATE PyTorch model loaded")
        except Exception as e:
            print(f"   ⚠️  Could not load PyTorch GATE model: {e}")
            print(f"   Falling back to scikit-learn model...")
    
    # Fallback to sklearn
    if not use_pytorch_gate:
        gate_model, scaler_mean, scaler_scale, scaler = load_gate_model_sklearn(
            str(config.GATE_MODEL_PKL), str(config.GATE_SCALER_PATH)
        )
        gate_scaler_or_params = scaler  # Pass the whole scaler object
        print(f"   ✓ GATE scikit-learn model loaded")
    
    print("✓ All models loaded\n")
    print(f"Using {'PyTorch' if use_pytorch_gate else 'Scikit-learn'} GATE model\n")
    
    # Process missing AOIs
    print(f"{'='*80}")
    print(f"PROCESSING {len(missing)} AOIs")
    print(f"{'='*80}\n")
    
    success_count = 0
    fail_count = 0
    
    for i, aoi_name in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {aoi_name}")
        
        try:
            # Generate matrix with GATE predictions
            shape = generate_single_matrix_with_gate(
                aoi_name,
                config,
                device,
                autoencoder,
                encoder,
                iforest,
                scaler_iforest,
                gate_model,
                gate_scaler_or_params,
                use_pytorch_gate,
                kmeans,
                scaler_kmeans
            )
            print(f"   ✅ Success! Shape: {shape} (5 channels including GATE)\n")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {e}\n")
            fail_count += 1
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Successful: {success_count}/{len(missing)}")
    print(f"❌ Failed: {fail_count}/{len(missing)}")
    print(f"{'='*80}\n")
    
    if success_count > 0:
        print(f"✅ Generated matrices saved to: {config.UNIFIED_PROB_DIR}")
        print(f"✅ Each matrix now has 5 channels:")
        print(f"   - Channel 0: Autoencoder anomaly scores")
        print(f"   - Channel 1: Isolation Forest probabilities")
        print(f"   - Channel 2: K-Means probabilities")
        print(f"   - Channel 3: Archaeological similarity")
        print(f"   - Channel 4: GATE final predictions ⭐")


if __name__ == "__main__":
    main()