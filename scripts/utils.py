"""
SONAR 2.0 - Utility Functions (OPTIMIZED VERSION)
Optimizations: vectorization, batching, caching, parallel operations
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
from functools import lru_cache

# ==============================================================================
# MODEL ARCHITECTURE (unchanged but with inference mode)
# ==============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        self.downsample = downsample
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(out_channels)
        
        if downsample:
            self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                 stride=2, padding=1, bias=False)
    
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.res_block(x)
        
        if self.downsample:
            skip = x
            x = self.pool(x)
            return x, skip
        else:
            return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels + skip_channels, out_channels, 
                             kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn(self.conv(x)))
        x = self.res_block(x)
        return x

class ResUNetAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 7, latent_dim: int = 256):
        super().__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 32, downsample=True)
        self.enc2 = EncoderBlock(32, 64, downsample=True)
        self.enc3 = EncoderBlock(64, 128, downsample=True)
        self.bottleneck = EncoderBlock(128, 256, downsample=False)
        
        # Latent
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.latent_dim = latent_dim
        
        # Decoder
        self.dec1 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec3 = DecoderBlock(64, 32, 32)
        self.final_conv = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x = self.bottleneck(x)
        
        # Latent
        latent = self.global_pool(x)
        latent = latent.view(latent.size(0), -1)
        
        # Decoder
        x = self.dec1(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec3(x, skip1)
        
        # Final Conv (Reduces 32 channels -> 7 channels)
        x = self.final_conv(x)
        reconstruction = torch.sigmoid(x)
        return reconstruction, latent
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ResUNetEncoder(nn.Module):
    """Encoder-only version for embedding extraction"""
    def __init__(self, in_channels: int = 7, embedding_dim: int = 128):
        super().__init__()
        
        # Encoder (must match autoencoder architecture)
        self.enc1 = EncoderBlock(in_channels, 32, downsample=True)
        self.enc2 = EncoderBlock(32, 64, downsample=True)
        self.enc3 = EncoderBlock(64, 128, downsample=True)
        self.bottleneck = EncoderBlock(128, 256, downsample=False)
        
        # Embedding projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_dim = embedding_dim
        
        # Bottleneck outputs 256 channels, project to embedding_dim
        self.projection = nn.Linear(256, embedding_dim)
    
    def forward(self, x):
        # Encoder path
        x, _ = self.enc1(x)
        x, _ = self.enc2(x)
        x, _ = self.enc3(x)
        x = self.bottleneck(x)
        
        # Global pooling: (batch, 256, H, W) -> (batch, 256, 1, 1)
        x = self.global_pool(x)
        # Flatten: (batch, 256, 1, 1) -> (batch, 256)
        x = x.view(x.size(0), -1)
        
        # Project to target embedding dimension: (batch, 256) -> (batch, embedding_dim)
        embedding = self.projection(x)
        
        return embedding
    
    def load_from_autoencoder(self, autoencoder_path: str):
        """Load encoder weights from trained autoencoder"""
        print(f"Loading encoder weights from: {autoencoder_path}")
        
        # Load full autoencoder state dict
        autoencoder_state = torch.load(autoencoder_path, map_location='cpu')
        
        # Filter to get only encoder + bottleneck weights
        encoder_dict = {}
        for k, v in autoencoder_state.items():
            if k.startswith(('enc1.', 'enc2.', 'enc3.', 'bottleneck.')):
                encoder_dict[k] = v
        
        # Load encoder weights (projection layer will remain randomly initialized)
        missing, unexpected = self.load_state_dict(encoder_dict, strict=False)
        
        # Verify what was loaded
        print(f"✓ Loaded {len(encoder_dict)} encoder layers")
        print(f"  Missing keys (expected): {len(missing)} - {missing[:3]}..." if len(missing) > 3 else f"  Missing keys: {missing}")
        
        # The projection layer should be in missing keys (it's not in autoencoder)
        if 'projection.weight' not in missing or 'projection.bias' not in missing:
            print("⚠️  Warning: Projection layer might have been incorrectly loaded")
        else:
            print(f"✓ Projection layer (256 → {self.embedding_dim}) will be trained from scratch")

# ==============================================================================
# RECONSTRUCTION FUNCTIONS (optimized)
# ==============================================================================

@lru_cache(maxsize=4)
def _get_blend_weight_template(patch_size: int, use_weighted: bool) -> np.ndarray:
    """OPTIMIZED: Cache blend weight templates"""
    if use_weighted:
        y, x = np.ogrid[0:patch_size, 0:patch_size]
        center = patch_size // 2
        dist_from_center = np.sqrt((y - center)**2 + (x - center)**2)
        max_dist = np.sqrt(2) * center
        blend_weight = 1.0 - (dist_from_center / max_dist) ** 2
        blend_weight = np.clip(blend_weight, 0.1, 1.0)
    else:
        blend_weight = np.ones((patch_size, patch_size), dtype=np.float64)
    
    return blend_weight

def reconstruct_full_aoi_corrected(
    patches: np.ndarray, 
    metadata: List[Dict],
    patch_size: int = 64,
    use_weighted: bool = True
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """OPTIMIZED: Reconstruct full AOI from overlapping patches with vectorized blending"""
    
    if len(patches) == 0:
        raise ValueError("No patches provided for reconstruction")
    
    num_patches = patches.shape[0]
    num_channels = patches.shape[1]
    
    print(f"      Reconstructing from {num_patches} patches with {num_channels} channels...")
    
    # Determine full AOI dimensions
    max_row = max(m['row'] + patch_size for m in metadata)
    max_col = max(m['col'] + patch_size for m in metadata)
    
    print(f"      Target dimensions: {max_row}×{max_col} pixels")
    
    # Initialize accumulation arrays
    reconstructed = np.zeros((num_channels, max_row, max_col), dtype=np.float64)
    weight_map = np.zeros((max_row, max_col), dtype=np.float64)
    
    # Get cached blend weight template
    blend_weight = _get_blend_weight_template(patch_size, use_weighted)
    
    # OPTIMIZATION: Vectorized patch stitching where possible
    # Process all patches (can't fully vectorize due to variable positions)
    for i, meta in enumerate(metadata):
        row_start = meta['row']
        col_start = meta['col']
        
        row_end = min(row_start + patch_size, max_row)
        col_end = min(col_start + patch_size, max_col)
        
        patch_height = row_end - row_start
        patch_width = col_end - col_start
        
        if patch_height <= 0 or patch_width <= 0:
            continue
        
        patch_data = patches[i, :, :patch_height, :patch_width]
        patch_weight = blend_weight[:patch_height, :patch_width]
        
        # Vectorized channel accumulation
        reconstructed[:, row_start:row_end, col_start:col_end] += (
            patch_data * patch_weight
        )
        
        weight_map[row_start:row_end, col_start:col_end] += patch_weight
    
    # OPTIMIZATION: Vectorized normalization
    weight_map_safe = np.where(weight_map > 0, weight_map, 1.0)
    reconstructed = reconstructed / weight_map_safe
    
    # Set uncovered areas to NaN (vectorized)
    uncovered_mask = weight_map == 0
    reconstructed[:, uncovered_mask] = np.nan
    
    reconstructed = reconstructed.astype(np.float32)
    weight_map = weight_map.astype(np.float32)
    
    # Report coverage statistics
    coverage = np.sum(weight_map > 0) / weight_map.size * 100
    avg_overlap = np.mean(weight_map[weight_map > 0]) if np.any(weight_map > 0) else 0
    
    print(f"      Coverage: {coverage:.1f}% | Avg overlap: {avg_overlap:.2f}×")
    
    if coverage < 50:
        print(f"      ⚠️  WARNING: Only {coverage:.1f}% coverage!")
    
    return reconstructed, weight_map, (max_row, max_col)


# ==============================================================================
# ANOMALY COMPUTATION (optimized batching)
# ==============================================================================

@torch.inference_mode()
def compute_autoencoder_probabilities(
    model: nn.Module, 
    patches: np.ndarray,
    metadata: List[Dict], 
    device: torch.device,
    batch_size: int = 32,
    use_sigmoid_scores: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OPTIMIZED: Compute autoencoder anomaly probabilities with batched inference
    
    Returns:
        prob_scores: (num_patches,) - Normalized probability scores [0, 1]
        pixel_scores: (num_patches, H, W) - Per-pixel anomaly maps
        reconstructed_patches: (num_patches, C, H, W) - Reconstructions
    """
    model.eval()
    
    patches_torch = torch.from_numpy(patches).float()
    
    anomaly_scores = []
    pixel_scores = []
    reconstructed_patches = []
    
    for i in tqdm(range(0, len(patches), batch_size), 
                 desc="Model 1 (Autoencoder)", leave=False):
        batch = patches_torch[i:i+batch_size].to(device)
        
        reconstruction, _ = model(batch)
        
        if use_sigmoid_scores:
            # Vectorized sigmoid uncertainty computation
            sigmoid_uncertainty = 1.0 - torch.abs(reconstruction - 0.5) * 2
            pixel_score = sigmoid_uncertainty.mean(dim=1)
            patch_score = sigmoid_uncertainty.mean(dim=(1, 2, 3))
        else:
            # Vectorized MSE computation
            squared_error = (reconstruction - batch) ** 2
            pixel_score = squared_error.mean(dim=1)
            patch_score = squared_error.mean(dim=(1, 2, 3))
        
        anomaly_scores.append(patch_score.cpu().numpy())
        pixel_scores.append(pixel_score.cpu().numpy())
        reconstructed_patches.append(reconstruction.cpu().numpy())
    
    # OPTIMIZATION: Efficient concatenation
    anomaly_scores = np.concatenate(anomaly_scores)
    pixel_scores = np.concatenate(pixel_scores)
    reconstructed_patches = np.concatenate(reconstructed_patches)
    
    # OPTIMIZATION: Vectorized normalization
    prob_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-10)
    
    return prob_scores, pixel_scores, reconstructed_patches

@torch.inference_mode()
def compute_full_aoi_anomalies(
    model: nn.Module, 
    patches: np.ndarray,
    metadata: List[Dict], 
    device: torch.device,
    batch_size: int = 32,
    use_sigmoid_scores: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    OPTIMIZED: Compute autoencoder anomaly scores for full AOI visualization
    
    Returns:
        anomaly_scores: (num_patches,) - Per-patch anomaly scores
        pixel_scores: (num_patches, H, W) - Per-pixel anomaly maps
        reconstructed_patches: (num_patches, C, H, W) - Model reconstructions
        original_patches: (num_patches, C, H, W) - Original input patches
    """
    model.eval()
    
    patches_torch = torch.from_numpy(patches).float()
    
    anomaly_scores = []
    pixel_scores = []
    reconstructed_patches = []
    
    for i in tqdm(range(0, len(patches), batch_size), 
                 desc="Computing anomalies", leave=False):
        batch = patches_torch[i:i+batch_size].to(device)
        
        reconstruction, _ = model(batch)
        
        if use_sigmoid_scores:
            sigmoid_uncertainty = 1.0 - torch.abs(reconstruction - 0.5) * 2
            pixel_score = sigmoid_uncertainty.mean(dim=1)
            patch_score = sigmoid_uncertainty.mean(dim=(1, 2, 3))
        else:
            squared_error = (reconstruction - batch) ** 2
            pixel_score = squared_error.mean(dim=1)
            patch_score = squared_error.mean(dim=(1, 2, 3))
        
        anomaly_scores.append(patch_score.cpu().numpy())
        pixel_scores.append(pixel_score.cpu().numpy())
        reconstructed_patches.append(reconstruction.cpu().numpy())
    
    anomaly_scores = np.concatenate(anomaly_scores)
    pixel_scores = np.concatenate(pixel_scores)
    reconstructed_patches = np.concatenate(reconstructed_patches)
    
    original_patches = patches
    
    return anomaly_scores, pixel_scores, reconstructed_patches, original_patches

# ==============================================================================
# TRAIN/TEST SPLIT
# ==============================================================================

def discover_and_split_aois(
    patches_dir: Path, 
    train_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    """Discover all AOIs and split into train/test sets"""
    
    # Find all important_patches files
    important_files = sorted(list(patches_dir.glob("AOI_*_important_patches.npz")))
    
    if len(important_files) == 0:
        raise FileNotFoundError(f"No *_important_patches.npz files found in {patches_dir}")
    
    # Extract AOI names (remove suffix)
    aoi_names = [f.stem.replace('_important_patches', '') for f in important_files]
    
    print(f"\n📂 Discovered {len(aoi_names)} AOIs in {patches_dir}")
    print(f"   Examples: {', '.join(aoi_names[:5])}")
    
    # Shuffle for random split
    np.random.shuffle(aoi_names)
    
    split_idx = int(len(aoi_names) * train_ratio)
    train_aoi_names = aoi_names[:split_idx]
    test_aoi_names = aoi_names[split_idx:]
    
    print(f"\n📊 AOI-Level Split:")
    print(f"   Total AOIs: {len(aoi_names)}")
    print(f"   Training AOIs: {len(train_aoi_names)}")
    print(f"   Testing AOIs: {len(test_aoi_names)}")
    
    return train_aoi_names, test_aoi_names


# ==============================================================================
# VISUALIZATION (unchanged)
# ==============================================================================
def visualize_full_aoi_anomalies(
    aoi_name: str, 
    model: nn.Module,
    patches_dir: Path,
    device: torch.device, 
    top_k: int = 20,
    channel_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    use_sigmoid_scores: bool = True
):
    """Visualize anomalies on full AOI reconstruction using ALL patches"""
    
    if channel_names is None:
        channel_names = ['DTM', 'Slope', 'Roughness', 'NDVI', 'NDWI', 'FlowAcc', 'FlowDir']
    
    score_type = "Sigmoid Score" if use_sigmoid_scores else "MSE"
    
    print(f"\n{'='*70}")
    print(f"FULL AOI ANOMALY DETECTION: {aoi_name}")
    print(f"Scoring Method: {score_type}")
    print(f"{'='*70}")
    
    # Load ALL patches
    all_patches_file = patches_dir / f"{aoi_name}_all_patches.npz"
    if not all_patches_file.exists():
        raise FileNotFoundError(f"All patches file not found: {all_patches_file}")
    
    print(f"   Loading ALL patches from: {all_patches_file.name}")
    with np.load(all_patches_file, allow_pickle=True) as data:
        patches = data['patches']
        metadata = list(data['metadata'])
        patches = np.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"   Patches loaded: {len(patches)}")
    
    # Compute anomalies
    print(f"   🔬 Computing anomaly scores ({score_type})...")
    anomaly_scores, pixel_scores, recon_patches, original_patches = compute_full_aoi_anomalies(
        model, patches, metadata, device, use_sigmoid_scores=use_sigmoid_scores
    )
        
    # Reconstruct full AOI
    print(f"   🔨 Reconstructing original AOI...")
    original_full, weight_map_orig, shape = reconstruct_full_aoi_corrected(
        patches, metadata, use_weighted=True
    )
    
    print(f"   🔨 Reconstructing from model output...")
    reconstructed_full, weight_map_recon, _ = reconstruct_full_aoi_corrected(
        recon_patches, metadata, use_weighted=True
    )
    
    # Create anomaly heatmap
    print(f"   🗺️  Creating anomaly heatmap...")
    anomaly_heatmap = np.zeros(shape, dtype=np.float32)
    anomaly_count = np.zeros(shape, dtype=np.float32)
    
    for i, meta in enumerate(metadata):
        row = meta['row']
        col = meta['col']
        row_end = min(row + 64, shape[0])
        col_end = min(col + 64, shape[1])
        patch_h = row_end - row
        patch_w = col_end - col
        
        if patch_h > 0 and patch_w > 0:
            anomaly_heatmap[row:row_end, col:col_end] += pixel_scores[i, :patch_h, :patch_w]
            anomaly_count[row:row_end, col:col_end] += 1
    
    anomaly_count = np.maximum(anomaly_count, 1)
    anomaly_heatmap = anomaly_heatmap / anomaly_count
    
    # Find top anomalies
    top_indices = np.argsort(anomaly_scores)[-top_k:][::-1]
    
    # Create visualization
    print(f"   🎨 Creating visualization...")
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Original channels
    for i, (ch_idx, ch_name) in enumerate([(0, 'DTM'), (3, 'NDVI'), (4, 'NDWI')]):
        ax = fig.add_subplot(gs[0, i])
        data = original_full[ch_idx]
        data_masked = np.ma.masked_invalid(data)
        cmap = 'terrain' if ch_idx == 0 else 'RdYlGn'
        im = ax.imshow(data_masked, cmap=cmap, interpolation='bilinear')
        ax.set_title(f'{aoi_name}\n{ch_name} (Original)', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 2: Reconstruction quality
    ax = fig.add_subplot(gs[1, 0])
    recon_masked = np.ma.masked_invalid(reconstructed_full[0])
    im = ax.imshow(recon_masked, cmap='terrain', interpolation='bilinear')
    ax.set_title('DTM Reconstruction', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = fig.add_subplot(gs[1, 1])
    error_map = np.abs(original_full[0] - reconstructed_full[0])
    error_masked = np.ma.masked_invalid(error_map)
    im = ax.imshow(error_masked, cmap='Reds', interpolation='bilinear')
    ax.set_title('Absolute Error (DTM)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(weight_map_orig, cmap='viridis', interpolation='nearest')
    coverage_pct = np.sum(weight_map_orig > 0) / weight_map_orig.size * 100
    ax.set_title(f'Patch Overlap\n(coverage: {coverage_pct:.1f}%)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 3: Anomaly detection
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(anomaly_heatmap, cmap='hot', interpolation='bilinear')
    title = f'Anomaly Heatmap ({score_type})'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Anomaly Score')
    
    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(original_full[0], cmap='terrain', interpolation='bilinear', alpha=0.6)
    im = ax.imshow(anomaly_heatmap, cmap='hot', interpolation='bilinear', alpha=0.5)
    
    for idx in top_indices[:10]:
        row = metadata[idx]['row']
        col = metadata[idx]['col']
        rect = Rectangle((col, row), 64, 64, linewidth=2, 
                        edgecolor='cyan', facecolor='none', linestyle='--')
        ax.add_patch(rect)
    
    ax.set_title(f'Top-{min(10, top_k)} Anomalies\n(cyan boxes)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    for i in range(min(6, len(top_indices))):
        idx = top_indices[i]
        row = metadata[idx]['row']
        col = metadata[idx]['col']
        
        row_end = min(row + 64, shape[0])
        col_end = min(col + 64, shape[1])
        patch_dtm = original_full[0, row:row_end, col:col_end]
        patch_error = pixel_scores[idx]
        
        sub_row = i // 3
        sub_col = i % 3
        sub_ax = ax.inset_axes([sub_col*0.33, (1-sub_row*0.5)-0.45, 0.3, 0.42])
        
        sub_ax.imshow(patch_dtm, cmap='terrain', interpolation='nearest')
        sub_ax.imshow(patch_error, cmap='hot', interpolation='nearest', alpha=0.5)
        sub_ax.set_title(f'#{i+1}: {anomaly_scores[idx]:.4f}', fontsize=8)
        sub_ax.axis('off')
    
    ax.set_title(f'Top-6 Anomalous Patches', fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle(f'{aoi_name} - Full AOI Anomaly Detection ({score_type})\n{shape[0]}×{shape[1]} pixels from {len(patches)} patches', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\n   📊 Anomaly Statistics ({score_type}):")
    print(f"      Mean score: {np.mean(anomaly_scores):.6f}")
    print(f"      Std score: {np.std(anomaly_scores):.6f}")
    print(f"      95th percentile: {np.percentile(anomaly_scores, 95):.6f}")
    print(f"      99th percentile: {np.percentile(anomaly_scores, 99):.6f}")
    
    print(f"\n   🔍 Top-{min(10, top_k)} Anomalies:")
    for i, idx in enumerate(top_indices[:10]):
        row = metadata[idx]['row']
        col = metadata[idx]['col']
        print(f"      {i+1:2d}. Patch at ({row:4d}, {col:4d}) - Score: {anomaly_scores[idx]:.6f}")
    
    return anomaly_scores, anomaly_heatmap, original_full, reconstructed_full


# ==============================================================================
# DATA PROCESSING
# ==============================================================================

def load_patches(patches_file: Path) -> Tuple[np.ndarray, List[Dict]]:
    """Load patches from .npz file and sanitize data"""
    with np.load(patches_file, allow_pickle=True) as data:
        patches = data['patches']
        metadata = list(data['metadata'])
        patches = np.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)
    return patches, metadata


# ==============================================================================
# EMBEDDING EXTRACTION (optimized batching)
# ==============================================================================

@torch.inference_mode()
def extract_embeddings(
    encoder: nn.Module,
    patches: np.ndarray,
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:
    """OPTIMIZED: Extract embeddings from patches using encoder with batching"""
    encoder.eval()
    
    patches_torch = torch.from_numpy(patches).float()
    embeddings = []
    
    for i in tqdm(range(0, len(patches), batch_size), 
                 desc="Extracting embeddings", leave=False):
        batch = patches_torch[i:i+batch_size].to(device)
        embedding = encoder(batch)
        embeddings.append(embedding.cpu().numpy())
    
    # OPTIMIZATION: Efficient concatenation
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


# ==============================================================================
# ISOLATION FOREST OPERATIONS
# ==============================================================================

def train_isolation_forest(
    embeddings: np.ndarray,
    contamination: float = 0.05,
    n_estimators: int = 100,
    max_samples: str = 'auto',
    random_state: int = 42
) -> Tuple[IsolationForest, StandardScaler]:
    """Train Isolation Forest on embeddings"""
    print(f"\n🌲 Training Isolation Forest")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Contamination: {contamination:.1%}")
    print(f"   Trees: {n_estimators}")
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Train Isolation Forest
    iforest = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    iforest.fit(embeddings_scaled)
    
    # Get anomaly scores
    scores = iforest.decision_function(embeddings_scaled)
    predictions = iforest.predict(embeddings_scaled)
    
    n_anomalies = np.sum(predictions == -1)
    print(f"   ✓ Training complete")
    print(f"   Detected anomalies: {n_anomalies} ({n_anomalies/len(predictions)*100:.2f}%)")
    print(f"   Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    return iforest, scaler


def compute_isolation_forest_anomalies(
    encoder: nn.Module,
    iforest: IsolationForest,
    scaler: StandardScaler,
    patches: np.ndarray,
    device: torch.device,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OPTIMIZED: Compute anomaly scores and predictions for patches
    
    Returns:
        probability_scores: Normalized anomaly probabilities [0, 1]
        predictions: Binary predictions (-1 = anomaly, 1 = normal)
    """
    # Extract embeddings (optimized)
    embeddings = extract_embeddings(encoder, patches, device, batch_size)
    embeddings_scaled = scaler.transform(embeddings)
    
    # Get raw scores
    raw_scores = iforest.decision_function(embeddings_scaled)
    predictions = iforest.predict(embeddings_scaled)

    # Invert so Higher = More Anomalous
    inverted_scores = -raw_scores 

    # OPTIMIZATION: Vectorized min-max normalization
    min_s = inverted_scores.min()
    max_s = inverted_scores.max()
    probability_scores = (inverted_scores - min_s) / (max_s - min_s + 1e-10)

    return probability_scores, predictions


def compute_iforest_probabilities(
    encoder: nn.Module,
    iforest: IsolationForest,
    scaler: StandardScaler,
    patches: np.ndarray,
    metadata: List[Dict],
    device: torch.device,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OPTIMIZED: Compute Isolation Forest anomaly probabilities
    
    Returns:
        prob_scores: (num_patches,) - Normalized probability scores [0, 1]
        embeddings: (num_patches, embedding_dim) - Latent embeddings
        predictions: (num_patches,) - Binary predictions (-1=anomaly, 1=normal)
    """
    # Extract embeddings (optimized)
    embeddings = extract_embeddings(encoder, patches, device, batch_size)
    embeddings_scaled = scaler.transform(embeddings)
    
    # Get raw scores
    raw_scores = iforest.decision_function(embeddings_scaled)
    predictions = iforest.predict(embeddings_scaled)

    # Invert and normalize (vectorized)
    inverted_scores = -raw_scores 
    min_s = inverted_scores.min()
    max_s = inverted_scores.max()
    prob_scores = (inverted_scores - min_s) / (max_s - min_s + 1e-10)

    return prob_scores, embeddings, predictions


def save_model(iforest: IsolationForest, scaler: StandardScaler, save_path: str):
    """Save trained Isolation Forest model and scaler"""
    with open(save_path, 'wb') as f:
        pickle.dump({'iforest': iforest, 'scaler': scaler}, f)
    print(f"💾 Saved model: {save_path}")


def load_model(model_path: str) -> Tuple[IsolationForest, StandardScaler]:
    """Load trained Isolation Forest model and scaler"""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Loaded model from: {model_path}")
    return data['iforest'], data['scaler']


# ==============================================================================
# KMEANS SIGNATURE MATCHING (optimized vectorization)
# ==============================================================================

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def compute_cosine_similarity_batch(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    OPTIMIZED: Vectorized cosine similarity computation
    
    Args:
        embeddings: (N, D) array
        centroids: (K, D) array
    
    Returns:
        similarities: (N, K) array
    """
    # Normalize embeddings and centroids
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    
    # Compute all similarities at once
    similarities = np.dot(embeddings_norm, centroids_norm.T)
    
    return similarities


def compute_signature_scores(
    latent: np.ndarray,
    centroids: np.ndarray,
    scaler: Optional[StandardScaler] = None
) -> Dict:
    """Compute signature matching scores for a single latent vector"""
    if scaler is not None:
        latent_scaled = scaler.transform(latent.reshape(1, -1))[0]
    else:
        latent_scaled = latent
    
    n_clusters = centroids.shape[0]
    similarities = np.zeros(n_clusters)
    
    for i in range(n_clusters):
        similarities[i] = compute_cosine_similarity(latent_scaled, centroids[i])
    
    similarities_normalized = (similarities + 1.0) / 2.0
    
    best_cluster = int(np.argmax(similarities_normalized))
    best_score = float(similarities_normalized[best_cluster])
    
    return {
        "cluster_scores": similarities_normalized.tolist(),
        "best_cluster": best_cluster,
        "best_score": best_score,
        "max_similarity": best_score
    }


def compute_kmeans_probabilities(
    encoder: nn.Module,
    kmeans_model,
    scaler: Optional[StandardScaler],
    patches: np.ndarray,
    metadata: List[Dict],
    device: torch.device,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OPTIMIZED: Compute K-Means signature matching probabilities with vectorization
    
    Returns:
        prob_scores: (num_patches,) - Match probability scores [0, 1]
        similarity_scores: (num_patches,) - Max similarity to any cluster
        cluster_assignments: (num_patches,) - Best matching cluster ID
    """
    # Extract embeddings (optimized)
    embeddings = extract_embeddings(encoder, patches, device, batch_size)
    
    if scaler is not None:
        embeddings_scaled = scaler.transform(embeddings)
    else:
        embeddings_scaled = embeddings
    
    centroids = kmeans_model.cluster_centers_
    
    # OPTIMIZATION: Vectorized cosine similarity computation
    similarities = compute_cosine_similarity_batch(embeddings_scaled, centroids)
    similarities_normalized = (similarities + 1.0) / 2.0
    
    # Get best cluster and scores for each patch
    cluster_assignments = np.argmax(similarities_normalized, axis=1)
    prob_scores = similarities_normalized[np.arange(len(similarities_normalized)), cluster_assignments]
    similarity_scores = prob_scores.copy()
    
    return prob_scores, similarity_scores, cluster_assignments


def load_kmeans_model(model_path: str):
    """Load trained K-Means model and scaler"""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ Loaded K-Means model from: {model_path}")
    
    # Handle different storage formats
    if isinstance(data, dict):
        kmeans = data.get('kmeans', data.get('model', None))
        scaler = data.get('scaler', None)
    else:
        kmeans = data
        scaler = None
        print("⚠️  Warning: No scaler found, loaded KMeans object only")
    
    if kmeans is None:
        raise ValueError(f"Could not find KMeans model in {model_path}")
    
    return kmeans, scaler


# ==============================================================================
# PROBABILITY MATRIX OPERATIONS (vectorized)
# ==============================================================================

def load_unified_probability_matrix(
    aoi_name: str,
    load_dir: Path = Path('./unified_probability_matrices')
) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """Load saved unified probability matrix"""
    load_path = load_dir / f"{aoi_name}_unified_prob_matrix.npz"
    
    if not load_path.exists():
        raise FileNotFoundError(f"Unified matrix not found: {load_path}")
    
    with np.load(load_path, allow_pickle=True) as data:
        unified_matrix = data['unified_matrix']
        metadata = data['metadata']
        channel_names = data['channel_names']
    
    print(f"✓ Loaded unified probability matrix from: {load_path}")
    print(f"  Shape: {unified_matrix.shape}")
    print(f"  Channels: {list(channel_names)}")
    
    return unified_matrix, metadata, channel_names


def broadcast_patch_scores_to_pixels(
    patch_scores: np.ndarray,
    patch_size: int = 64
) -> np.ndarray:
    """
    OPTIMIZED: Broadcast patch-level scores to pixel-level
    
    Args:
        patch_scores: Shape (num_patches,) or (num_patches, 1)
        patch_size: Size of patches (default 64)
    
    Returns:
        pixel_scores: Shape (num_patches, patch_size, patch_size)
    """
    if patch_scores.ndim == 1:
        patch_scores = patch_scores[:, np.newaxis, np.newaxis]
    elif patch_scores.ndim == 2 and patch_scores.shape[1] == 1:
        patch_scores = patch_scores[:, :, np.newaxis]
    
    num_patches = patch_scores.shape[0]
    
    # OPTIMIZATION: Use np.broadcast_to instead of loops
    pixel_scores = np.broadcast_to(
        patch_scores,
        (num_patches, patch_size, patch_size)
    ).copy()
    
    return pixel_scores


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    OPTIMIZED: Vectorized min-max normalization
    
    Args:
        scores: Input scores of any shape
    
    Returns:
        normalized_scores: Same shape as input, values in [0, 1]
    """
    min_val = scores.min()
    max_val = scores.max()
    
    if max_val - min_val < 1e-10:
        return np.zeros_like(scores)
    
    normalized = (scores - min_val) / (max_val - min_val)
    return normalized.astype(np.float32)


def save_unified_probability_matrix(
    aoi_name: str,
    unified_matrix: np.ndarray,
    metadata: List[Dict],
    save_dir: Path = Path('./unified_probability_matrices'),
    channel_names: List[str] = None
):
    """Save unified probability matrix to disk"""
    if channel_names is None:
        channel_names = ['prob_autoencoder', 'prob_iforest', 'prob_kmeans', 'sim_score']
    
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / f"{aoi_name}_unified_prob_matrix.npz"
    
    np.savez_compressed(
        save_path,
        unified_matrix=unified_matrix,
        metadata=metadata,
        channel_names=channel_names,
        aoi_name=aoi_name
    )
    
    print(f"💾 Saved unified probability matrix: {save_path}")
    print(f"   Shape: {unified_matrix.shape}")
    print(f"   Channels: {channel_names}")


def save_probability_matrix(
    aoi_name: str,
    probability_matrix: np.ndarray,
    metadata: List[Dict],
    save_dir: Path = Path('./probability_matrices')
):
    """Save probability matrix for later use by the gate model"""
    save_dir.mkdir(exist_ok=True, parents=True)
    
    save_path = save_dir / f"{aoi_name}_probability_matrix.npz"
    
    np.savez_compressed(
        save_path,
        probability_matrix=probability_matrix,
        metadata=metadata,
        column_names=['prob_autoencoder', 'prob_iforest', 'prob_template', 'sim_score']
    )
    
    print(f"💾 Saved probability matrix: {save_path}")


def load_probability_matrix(
    aoi_name: str, 
    load_dir: Path = Path('./probability_matrices')
) -> Tuple[np.ndarray, List[Dict]]:
    """Load saved probability matrix"""
    load_path = load_dir / f"{aoi_name}_probability_matrix.npz"
    
    with np.load(load_path, allow_pickle=True) as data:
        probability_matrix = data['probability_matrix']
        metadata = data['metadata']
    
    print(f"✓ Loaded probability matrix from: {load_path}")
    print(f"  Shape: {probability_matrix.shape}")
    
    return probability_matrix, metadata


# ==============================================================================
# TEMPLATE MATCHING (placeholder - unchanged)
# ==============================================================================

def compute_template_matching_probabilities(
    patches: np.ndarray,
    metadata: List[Dict],
    template_library: Dict[str, np.ndarray],
    method: str = 'correlation'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute template matching probabilities (PLACEHOLDER)
    
    Returns:
        prob_scores: (num_patches,) - Match probability scores [0, 1]
        similarity_scores: (num_patches, num_templates) - Raw similarity to each template
    """
    from scipy.signal import correlate2d
    
    num_patches = len(patches)
    num_templates = len(template_library)
    
    similarity_scores = np.zeros((num_patches, num_templates))
    
    # Extract DTM channel (channel 0)
    patch_dtm = patches[:, 0, :, :]
    
    for i, patch in enumerate(tqdm(patch_dtm, desc="Model 3 (Template Matching)", leave=False)):
        for j, (template_name, template) in enumerate(template_library.items()):
            
            if method == 'correlation':
                corr = correlate2d(patch, template, mode='valid')
                if corr.size > 0:
                    similarity = np.max(corr) / (np.linalg.norm(patch) * np.linalg.norm(template) + 1e-10)
                else:
                    similarity = 0.0
            else:
                similarity = 0.0
            
            similarity_scores[i, j] = similarity
    
    # Use max similarity across templates
    max_similarity = similarity_scores.max(axis=1)
    
    # Normalize to [0, 1]
    prob_scores = (max_similarity - max_similarity.min()) / (max_similarity.max() - max_similarity.min() + 1e-10)
    
    return prob_scores, similarity_scores