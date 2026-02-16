import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple, List, Dict
from tqdm import tqdm


def load_archaeological_embeddings(
    arch_csv_path: str,
    embedding_dim: int = 256
) -> np.ndarray:
    """
    Load known archaeological site embeddings from CSV
    
    Args:
        arch_csv_path: Path to Arch_embedding_only.csv
        embedding_dim: Expected embedding dimension (default 256)
    
    Returns:
        arch_embeddings: Shape (num_sites, embedding_dim)
    """
    print(f"\n📚 Loading archaeological site embeddings...")
    print(f"   Path: {arch_csv_path}")
    
    df = pd.read_csv(arch_csv_path)
    
    # Extract embedding columns (emb_000 to emb_255)
    emb_cols = [col for col in df.columns if col.startswith('emb_')]
    emb_cols_sorted = sorted(emb_cols, key=lambda x: int(x.split('_')[1]))
    
    if len(emb_cols_sorted) != embedding_dim:
        print(f"   ⚠️  Warning: Expected {embedding_dim} embedding dims, found {len(emb_cols_sorted)}")
    
    arch_embeddings = df[emb_cols_sorted].values
    
    print(f"   ✓ Loaded {len(arch_embeddings)} archaeological site embeddings")
    print(f"   ✓ Embedding shape: {arch_embeddings.shape}")
    print(f"   ✓ Sites from AOIs: {df['aoi_name'].unique()}")
    
    return arch_embeddings


def compute_cosine_similarity_vectorized(
    patch_embeddings: np.ndarray,
    arch_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between patch embeddings and all arch site embeddings
    
    Args:
        patch_embeddings: Shape (num_patches, embedding_dim)
        arch_embeddings: Shape (num_arch_sites, embedding_dim)
    
    Returns:
        similarities: Shape (num_patches, num_arch_sites)
                     Each row contains similarities to all arch sites
    """
    # Normalize embeddings
    patch_norm = patch_embeddings / (np.linalg.norm(patch_embeddings, axis=1, keepdims=True) + 1e-10)
    arch_norm = arch_embeddings / (np.linalg.norm(arch_embeddings, axis=1, keepdims=True) + 1e-10)
    
    # Compute all pairwise similarities at once
    similarities = np.dot(patch_norm, arch_norm.T)
    
    return similarities


def compute_max_arch_similarity_scores(
    patch_embeddings: np.ndarray,
    arch_embeddings: np.ndarray,
    return_details: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    Compute maximum similarity score for each patch against known arch sites
    
    Args:
        patch_embeddings: Shape (num_patches, embedding_dim)
        arch_embeddings: Shape (num_arch_sites, embedding_dim)
        return_details: If True, also return best matching site index and all similarities
    
    Returns:
        max_similarities: Shape (num_patches,) - Max similarity per patch
        [Optional] best_match_indices: Shape (num_patches,) - Index of best matching arch site
        [Optional] all_similarities: Shape (num_patches, num_arch_sites) - All pairwise similarities
    """
    print(f"\n🔍 Computing arch site similarity scores...")
    print(f"   Patch embeddings: {patch_embeddings.shape}")
    print(f"   Arch site embeddings: {arch_embeddings.shape}")
    
    # Compute all pairwise similarities
    all_similarities = compute_cosine_similarity_vectorized(patch_embeddings, arch_embeddings)
    
    # Find maximum similarity for each patch
    max_similarities = np.max(all_similarities, axis=1)
    best_match_indices = np.argmax(all_similarities, axis=1)
    
    print(f"   ✓ Computed similarities")
    print(f"   ✓ Max similarity range: [{max_similarities.min():.4f}, {max_similarities.max():.4f}]")
    print(f"   ✓ Mean max similarity: {max_similarities.mean():.4f}")
    
    if return_details:
        return max_similarities, best_match_indices, all_similarities
    else:
        return max_similarities


def extract_embeddings_for_similarity(
    encoder: torch.nn.Module,
    patches: np.ndarray,
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:
    """
    Extract embeddings from patches using encoder (wrapper for consistency)
    
    Args:
        encoder: Trained encoder model
        patches: Shape (num_patches, channels, height, width)
        device: torch device
        batch_size: Batch size for inference
    
    Returns:
        embeddings: Shape (num_patches, embedding_dim)
    """
    encoder.eval()
    
    patches_torch = torch.from_numpy(patches).float()
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), 
                     desc="Extracting embeddings for similarity", leave=False):
            batch = patches_torch[i:i+batch_size].to(device)
            embedding = encoder(batch)
            embeddings.append(embedding.cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def compute_arch_similarity_probabilities(
    encoder: torch.nn.Module,
    patches: np.ndarray,
    arch_csv_path: str,
    device: torch.device,
    batch_size: int = 32,
    embedding_dim: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete pipeline: Extract patch embeddings and compute similarity to arch sites
    
    Args:
        encoder: Trained encoder model
        patches: Shape (num_patches, channels, height, width)
        arch_csv_path: Path to Arch_embedding_only.csv
        device: torch device
        batch_size: Batch size for inference
        embedding_dim: Embedding dimension
    
    Returns:
        prob_scores: Shape (num_patches,) - Normalized similarity scores [0, 1]
        max_similarities: Shape (num_patches,) - Raw max similarities [-1, 1]
        best_match_indices: Shape (num_patches,) - Best matching arch site index
    """
    # Load known archaeological site embeddings
    arch_embeddings = load_archaeological_embeddings(arch_csv_path, embedding_dim)
    
    # Extract patch embeddings
    print(f"\n🔬 Extracting patch embeddings...")
    patch_embeddings = extract_embeddings_for_similarity(
        encoder, patches, device, batch_size
    )
    
    # Compute similarities
    max_similarities, best_match_indices, all_similarities = compute_max_arch_similarity_scores(
        patch_embeddings, arch_embeddings, return_details=True
    )
    
    # Normalize to [0, 1] probability range
    # Cosine similarity is in [-1, 1], shift to [0, 1]
    prob_scores = (max_similarities + 1.0) / 2.0
    
    print(f"\n   📊 Probability Score Statistics:")
    print(f"      Range: [{prob_scores.min():.4f}, {prob_scores.max():.4f}]")
    print(f"      Mean: {prob_scores.mean():.4f}")
    print(f"      Std: {prob_scores.std():.4f}")
    print(f"      Median: {np.median(prob_scores):.4f}")
    
    # Distribution of best matches
    unique_matches, counts = np.unique(best_match_indices, return_counts=True)
    print(f"\n   🎯 Best Match Distribution (Top 5 sites):")
    top_5_idx = np.argsort(counts)[-5:][::-1]
    for idx in top_5_idx:
        site_idx = unique_matches[idx]
        count = counts[idx]
        print(f"      Site {site_idx}: {count} patches ({count/len(patches)*100:.1f}%)")
    
    return prob_scores, max_similarities, best_match_indices


def analyze_top_similarity_patches(
    max_similarities: np.ndarray,
    best_match_indices: np.ndarray,
    metadata: List[Dict],
    arch_csv_path: str,
    top_k: int = 10
):
    """
    Analyze and print information about patches with highest similarity to arch sites
    
    Args:
        max_similarities: Shape (num_patches,)
        best_match_indices: Shape (num_patches,)
        metadata: List of patch metadata
        arch_csv_path: Path to CSV to get site names
        top_k: Number of top patches to analyze
    """
    print(f"\n🏆 Top-{top_k} Patches Most Similar to Known Archaeological Sites:")
    print(f"{'='*80}")
    
    # Load arch site info
    df_arch = pd.read_csv(arch_csv_path)
    
    # Get top-k patches
    top_indices = np.argsort(max_similarities)[-top_k:][::-1]
    
    for rank, patch_idx in enumerate(top_indices, 1):
        similarity = max_similarities[patch_idx]
        best_site_idx = best_match_indices[patch_idx]
        
        # Get site info
        site_info = df_arch.iloc[best_site_idx]
        site_aoi = site_info['aoi_name']
        site_patch_id = site_info['patch_id']
        
        # Get patch info
        patch_meta = metadata[patch_idx]
        patch_row = patch_meta['row']
        patch_col = patch_meta['col']
        
        print(f"\n{rank:2d}. Patch {patch_idx} (row={patch_row}, col={patch_col})")
        print(f"    Similarity: {similarity:.6f}")
        print(f"    Best match: {site_aoi} - {site_patch_id}")
    
    print(f"\n{'='*80}")


# ==============================================================================
# INTEGRATION WITH MAIN PIPELINE
# ==============================================================================

def compute_arch_similarity_channel(
    encoder: torch.nn.Module,
    patches: np.ndarray,
    metadata: List[Dict],
    arch_csv_path: str,
    device: torch.device,
    batch_size: int = 32,
    embedding_dim: int = 256,
    patch_size: int = 64
) -> Tuple[np.ndarray, Dict]:
    """
    Compute Channel 3 (Similarity Score) for unified probability matrix
    
    This is the function to call from main.py to replace the K-Means similarity
    
    Args:
        encoder: Trained encoder model
        patches: Shape (num_patches, channels, height, width)
        metadata: List of patch metadata
        arch_csv_path: Path to Arch_embedding_only.csv
        device: torch device
        batch_size: Batch size
        embedding_dim: Embedding dimension
        patch_size: Size of patches (64)
    
    Returns:
        similarity_channel: Shape (num_patches, 64, 64) - Broadcasted similarity scores
        analysis_dict: Dictionary with analysis results
    """
    from utils import broadcast_patch_scores_to_pixels
    
    # Compute similarity scores
    prob_scores, max_similarities, best_match_indices = compute_arch_similarity_probabilities(
        encoder=encoder,
        patches=patches,
        arch_csv_path=arch_csv_path,
        device=device,
        batch_size=batch_size,
        embedding_dim=embedding_dim
    )
    
    # Broadcast to pixel level (each patch gets uniform score)
    similarity_channel = broadcast_patch_scores_to_pixels(prob_scores, patch_size=patch_size)
    
    # Analyze top matches
    analyze_top_similarity_patches(
        max_similarities=max_similarities,
        best_match_indices=best_match_indices,
        metadata=metadata,
        arch_csv_path=arch_csv_path,
        top_k=10
    )
    
    # Create analysis dictionary
    analysis_dict = {
        'prob_scores': prob_scores,
        'max_similarities': max_similarities,
        'best_match_indices': best_match_indices,
        'mean_similarity': float(max_similarities.mean()),
        'std_similarity': float(max_similarities.std()),
        'max_similarity': float(max_similarities.max()),
        'min_similarity': float(max_similarities.min())
    }
    
    return similarity_channel, analysis_dict