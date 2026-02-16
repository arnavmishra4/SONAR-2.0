"""
visualization.py - FastAPI-Compatible Visualization Functions (WITH GATE SUPPORT)
==================================================================================
Pure visualization functions that return image bytes or JSON data
for FastAPI endpoints. No Streamlit dependencies.

NOW INCLUDES: 5-channel support with GATE predictions visualization!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import io
from PIL import Image
from typing import Dict, List, Tuple, Optional
import json
import base64

# ==============================================================================
# 2D PATCH VISUALIZATION
# ==============================================================================

def generate_2d_patch_preview(
    patches: Dict[str, np.ndarray],
    aoi_name: str,
    patch_id: str,
    layer_order: List[str] = None
) -> bytes:
    """
    Generate 2D visualization of patch layers
    
    Args:
        patches: Dictionary of patch data {layer_name: array}
        aoi_name: Name of AOI
        patch_id: Patch identifier
        layer_order: Order of layers to display
    
    Returns:
        PNG image as bytes
    """
    if layer_order is None:
        layer_order = ['dtm', 'slope', 'ndvi', 'ndwi', 'flow_acc']
    
    show_layers = [l for l in layer_order if l in patches][:3]
    
    if not show_layers:
        # Return empty image
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.axis('off')
    else:
        fig, axes = plt.subplots(1, len(show_layers), figsize=(5*len(show_layers), 5))
        if len(show_layers) == 1:
            axes = [axes]
        
        cmaps = {
            'dtm': 'terrain',
            'slope': 'YlOrRd',
            'ndvi': 'RdYlGn',
            'ndwi': 'Blues',
            'flow_acc': 'viridis',
            'roughness': 'hot'
        }
        
        for idx, layer_name in enumerate(show_layers):
            data = patches[layer_name]
            im = axes[idx].imshow(data, cmap=cmaps.get(layer_name, 'viridis'), 
                                 interpolation='bilinear')
            
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                axes[idx].set_title(
                    f'{layer_name.upper()}\nμ={np.mean(valid):.2f}, σ={np.std(valid):.2f}',
                    fontweight='bold', fontsize=11
                )
            else:
                axes[idx].set_title(f'{layer_name.upper()}', fontweight='bold', fontsize=11)
            
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046)
        
        plt.suptitle(f'{aoi_name} - {patch_id}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
    
    # Convert to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


# ==============================================================================
# 3D TERRAIN VISUALIZATION
# ==============================================================================

def generate_3d_terrain_json(
    patches: Dict[str, np.ndarray],
    aoi_name: str,
    patch_id: str,
    lat: float,
    lon: float
) -> Dict:
    """
    Generate 3D terrain visualization data for Plotly
    
    Returns:
        Dictionary containing Plotly figure JSON
    """
    if 'dtm' not in patches:
        return {
            'error': 'No DTM data available',
            'figure': None
        }
    
    dtm = patches['dtm']
    
    # Handle NaN values
    dtm_clean = dtm.copy()
    valid_mask = ~np.isnan(dtm)
    
    if not valid_mask.any():
        return {
            'error': 'No valid DTM data',
            'figure': None
        }
    
    # Fill NaNs with median
    dtm_clean[~valid_mask] = np.nanmedian(dtm)
    
    # Create coordinate grids
    rows, cols = dtm.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # Color by slope if available
    if 'slope' in patches:
        slope = patches['slope']
        slope_clean = slope.copy()
        slope_clean[np.isnan(slope)] = np.nanmedian(slope)
        colorscale = 'YlOrRd'
        surfacecolor = slope_clean
        colorbar_title = 'Slope (°)'
    else:
        colorscale = 'earth'
        surfacecolor = dtm_clean
        colorbar_title = 'Elevation (m)'
    
    # Create 3D surface
    fig = go.Figure(data=[go.Surface(
        z=dtm_clean,
        x=X,
        y=Y,
        surfacecolor=surfacecolor,
        colorscale=colorscale,
        colorbar=dict(title=colorbar_title),
        lighting=dict(
            ambient=0.4,
            diffuse=0.8,
            fresnel=0.2,
            specular=0.3,
            roughness=0.5
        ),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project=dict(z=True)
            )
        )
    )])
    
    # Update layout
    fig.update_layout(
        title=f'3D Terrain View - {patch_id}<br>Lat: {lat:.6f}, Lon: {lon:.6f}',
        scene=dict(
            xaxis_title='X (pixels)',
            yaxis_title='Y (pixels)',
            zaxis_title='Elevation (m)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Statistics
    stats = {
        'min_elevation': float(np.nanmin(dtm)),
        'max_elevation': float(np.nanmax(dtm)),
        'relief': float(np.nanmax(dtm) - np.nanmin(dtm)),
        'mean_elevation': float(np.nanmean(dtm)),
        'std_elevation': float(np.nanstd(dtm))
    }
    
    return {
        'figure': fig.to_json(),
        'stats': stats
    }


# ==============================================================================
# AI MODEL ANALYSIS VISUALIZATION
# ==============================================================================

def generate_ai_analysis_visualization(
    patches: Dict[str, np.ndarray],
    results: Dict,
    patch_id: str
) -> bytes:
    """
    Generate AI model analysis visualization
    
    Args:
        patches: Patch data
        results: Results from analyze_patch_multi_model()
        patch_id: Patch identifier
    
    Returns:
        PNG image as bytes
    """
    # Determine number of plots
    n_plots = 3 if results['cluster_id'] is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 2:
        axes = list(axes)
    
    # Plot 1: Reconstruction Error
    ax = axes[0]
    dtm_orig = patches['dtm']
    dtm_recon = results['reconstruction'][0]
    diff = np.abs(dtm_orig - dtm_recon)
    
    im = ax.imshow(diff, cmap='Reds', interpolation='bilinear')
    ax.set_title('Reconstruction Error\n(Model 1: Autoencoder)', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Plot 2: IForest decision
    ax = axes[1]
    status = "ANOMALY" if results['iforest_is_anomaly'] else "NORMAL"
    color = 'red' if results['iforest_is_anomaly'] else 'green'
    ax.text(0.5, 0.5, f"{status}\nScore: {results['iforest_score']:.4f}", 
            ha='center', va='center', fontsize=20, fontweight='bold', color=color)
    ax.set_title('Isolation Forest\n(Model 2)', fontweight='bold')
    ax.axis('off')
    
    # Plot 3: Cluster similarities (if available)
    if results['cluster_id'] is not None:
        ax = axes[2]
        clusters = list(range(len(results['all_cluster_similarities'])))
        sims = results['all_cluster_similarities']
        colors = ['red' if i == results['cluster_id'] else 'gray' for i in clusters]
        ax.bar(clusters, sims, color=colors)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Similarity')
        ax.set_title('Cluster Similarities\n(Model 3: K-Means)', fontweight='bold')
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Convert to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


def generate_ai_analysis_json(results: Dict) -> Dict:
    """
    Generate JSON-serializable AI analysis results
    
    Args:
        results: Results from analyze_patch_multi_model()
    
    Returns:
        Dictionary with analysis data
    """
    return {
        'reconstruction_error': float(results['reconstruction_error']),
        'iforest_score': float(results['iforest_score']),
        'iforest_is_anomaly': bool(results['iforest_is_anomaly']),
        'cluster_id': int(results['cluster_id']) if results['cluster_id'] is not None else None,
        'cluster_similarity': float(results['cluster_similarity']) if results['cluster_similarity'] is not None else None,
        'combined_anomaly_score': float(results['combined_anomaly_score']),
        'verdict': get_anomaly_verdict(results['combined_anomaly_score'])
    }


def get_anomaly_verdict(combined_score: float) -> str:
    """Get human-readable verdict"""
    if combined_score > 0.7:
        return "HIGH ANOMALY LIKELIHOOD - Investigate!"
    elif combined_score > 0.5:
        return "MODERATE ANOMALY - Worth checking"
    else:
        return "LIKELY NORMAL TERRAIN"


# ==============================================================================
# UNIFIED PROBABILITY MATRIX VISUALIZATION (NOW WITH 5 CHANNELS!)
# ==============================================================================

def generate_probability_matrix_visualization(
    unified_matrix: np.ndarray,
    patch_idx: int,
    channel_names: List[str]
) -> bytes:
    """
    Visualize all probability channels for a single patch (supports 4 or 5 channels)
    
    Args:
        unified_matrix: Shape (num_patches, 64, 64, N) where N=4 or 5
        patch_idx: Index of patch to visualize
        channel_names: Names of the channels
    
    Returns:
        PNG image as bytes
    """
    patch_data = unified_matrix[patch_idx]  # Shape: (64, 64, N)
    n_channels = patch_data.shape[-1]
    
    # Determine grid layout
    if n_channels == 4:
        nrows, ncols = 2, 2
    elif n_channels == 5:
        nrows, ncols = 2, 3
    else:
        nrows = (n_channels + 2) // 3
        ncols = 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
    axes = axes.flatten() if n_channels > 1 else [axes]
    
    for i in range(n_channels):
        channel_data = patch_data[:, :, i]
        im = axes[i].imshow(channel_data, cmap='hot', vmin=0, vmax=1, interpolation='bilinear')
        axes[i].set_title(f'{channel_names[i]}\n(Range: {channel_data.min():.3f} - {channel_data.max():.3f})', 
                         fontweight='bold')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    # Hide unused subplots
    for i in range(n_channels, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Probability Matrix - Patch {patch_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


def generate_full_aoi_heatmap(
    unified_matrix: np.ndarray,
    metadata: List[Dict],
    aoi_shape: Tuple[int, int],
    channel_idx: int = 0,
    patch_size: int = 64
) -> bytes:
    """
    Generate full AOI heatmap for a specific probability channel
    
    Args:
        unified_matrix: Shape (num_patches, 64, 64, N) where N=4 or 5
        metadata: List of patch metadata with 'row' and 'col'
        aoi_shape: (height, width) of full AOI
        channel_idx: Which channel to visualize (0 to N-1)
        patch_size: Size of patches
    
    Returns:
        PNG image as bytes
    """
    # Reconstruct full heatmap
    heatmap = np.zeros(aoi_shape, dtype=np.float32)
    count_map = np.zeros(aoi_shape, dtype=np.float32)
    
    for i, meta in enumerate(metadata):
        row = meta['row']
        col = meta['col']
        row_end = min(row + patch_size, aoi_shape[0])
        col_end = min(col + patch_size, aoi_shape[1])
        patch_h = row_end - row
        patch_w = col_end - col
        
        if patch_h > 0 and patch_w > 0:
            patch_prob = unified_matrix[i, :patch_h, :patch_w, channel_idx]
            heatmap[row:row_end, col:col_end] += patch_prob
            count_map[row:row_end, col:col_end] += 1
    
    # Average overlapping areas
    count_map = np.maximum(count_map, 1)
    heatmap = heatmap / count_map
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
    ax.set_title(f'Full AOI Probability Heatmap\n{aoi_shape[0]}×{aoi_shape[1]} pixels', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, label='Probability')
    plt.tight_layout()
    
    # Convert to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


# ==============================================================================
# 🆕 GATE PREDICTIONS VISUALIZATION (NEW!)
# ==============================================================================

def generate_gate_prediction_heatmap(
    unified_matrix: np.ndarray,
    metadata: List[Dict],
    aoi_shape: Tuple[int, int],
    aoi_name: str,
    threshold: float = 0.5,
    patch_size: int = 64
) -> bytes:
    """
    Generate GATE prediction heatmap showing archaeological site candidates
    
    Args:
        unified_matrix: Shape (num_patches, 64, 64, 5) - Channel 4 is GATE predictions
        metadata: List of patch metadata
        aoi_shape: (height, width) of full AOI
        aoi_name: Name of AOI for title
        threshold: Classification threshold (default 0.5)
        patch_size: Patch size in pixels
    
    Returns:
        PNG image as bytes
    """
    # Extract GATE channel (channel 4)
    gate_channel_idx = 4
    
    # Reconstruct full heatmap
    heatmap = np.zeros(aoi_shape, dtype=np.float32)
    count_map = np.zeros(aoi_shape, dtype=np.float32)
    
    for i, meta in enumerate(metadata):
        row = meta['row']
        col = meta['col']
        row_end = min(row + patch_size, aoi_shape[0])
        col_end = min(col + patch_size, aoi_shape[1])
        patch_h = row_end - row
        patch_w = col_end - col
        
        if patch_h > 0 and patch_w > 0:
            gate_prob = unified_matrix[i, :patch_h, :patch_w, gate_channel_idx]
            heatmap[row:row_end, col:col_end] += gate_prob
            count_map[row:row_end, col:col_end] += 1
    
    # Average overlapping areas
    count_map = np.maximum(count_map, 1)
    heatmap = heatmap / count_map
    
    # Create binary mask for positive predictions
    positive_mask = heatmap >= threshold
    num_positive_pixels = positive_mask.sum()
    percent_positive = (num_positive_pixels / heatmap.size) * 100
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Show heatmap
    im = ax.imshow(heatmap, cmap='RdYlGn', vmin=0, vmax=1, interpolation='bilinear')
    
    # Overlay contours for high-probability areas
    contour_levels = [threshold, 0.7, 0.9]
    contours = ax.contour(heatmap, levels=contour_levels, colors=['yellow', 'orange', 'red'], 
                          linewidths=2, alpha=0.7)
    ax.clabel(contours, inline=True, fontsize=10)
    
    # Add title with statistics
    ax.set_title(
        f'GATE Archaeological Site Predictions - {aoi_name}\n'
        f'Positive Area: {percent_positive:.2f}% (threshold={threshold})',
        fontsize=16, fontweight='bold', pad=20
    )
    ax.axis('off')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('GATE Prediction Probability', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Convert to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


def generate_gate_positive_patches_visualization(
    unified_matrix: np.ndarray,
    metadata: List[Dict],
    patches_data: Dict[str, np.ndarray],
    threshold: float = 0.5,
    top_n: int = 16
) -> bytes:
    """
    Visualize top N patches with highest GATE predictions (archaeological candidates)
    
    Args:
        unified_matrix: Shape (num_patches, 64, 64, 5)
        metadata: Patch metadata
        patches_data: Original patch data dict with 'dtm', 'slope', etc.
        threshold: Minimum GATE score to consider
        top_n: Number of top patches to show
    
    Returns:
        PNG image as bytes showing DTM of top patches
    """
    # Extract GATE predictions (channel 4)
    gate_predictions = unified_matrix[:, :, :, 4]  # Shape: (num_patches, 64, 64)
    
    # Get mean GATE score per patch
    mean_scores = gate_predictions.mean(axis=(1, 2))  # Shape: (num_patches,)
    
    # Filter patches above threshold
    positive_mask = mean_scores >= threshold
    positive_indices = np.where(positive_mask)[0]
    
    if len(positive_indices) == 0:
        # No positive predictions
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'No patches above threshold {threshold}', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.axis('off')
    else:
        # Sort by score
        sorted_indices = positive_indices[np.argsort(mean_scores[positive_indices])[::-1]]
        top_indices = sorted_indices[:top_n]
        
        # Create grid
        n_show = len(top_indices)
        ncols = 4
        nrows = (n_show + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        axes = axes.flatten() if n_show > 1 else [axes]
        
        for idx, patch_idx in enumerate(top_indices):
            ax = axes[idx]
            
            # Get DTM for this patch
            dtm_patch = patches_data['dtm'][patch_idx] if 'dtm' in patches_data else None
            
            if dtm_patch is not None:
                im = ax.imshow(dtm_patch, cmap='terrain', interpolation='bilinear')
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                # Show GATE prediction if no DTM
                im = ax.imshow(gate_predictions[patch_idx], cmap='hot', vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, fraction=0.046)
            
            score = mean_scores[patch_idx]
            row = metadata[patch_idx]['row']
            col = metadata[patch_idx]['col']
            
            ax.set_title(f'Patch {patch_idx}\nGATE: {score:.3f}\n(r={row}, c={col})', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_show, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Top {n_show} Archaeological Site Candidates (GATE > {threshold})', 
                    fontsize=16, fontweight='bold', y=1.0)
        plt.tight_layout()
    
    # Convert to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


def generate_gate_statistics_json(
    unified_matrix: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    Generate statistics about GATE predictions for JSON response
    
    Args:
        unified_matrix: Shape (num_patches, 64, 64, 5)
        threshold: Classification threshold
    
    Returns:
        Dictionary with GATE statistics
    """
    gate_predictions = unified_matrix[:, :, :, 4]
    mean_scores = gate_predictions.mean(axis=(1, 2))
    
    positive_patches = (mean_scores >= threshold).sum()
    total_patches = len(mean_scores)
    
    return {
        'total_patches': int(total_patches),
        'positive_patches': int(positive_patches),
        'positive_percentage': float(positive_patches / total_patches * 100),
        'mean_gate_score': float(mean_scores.mean()),
        'max_gate_score': float(mean_scores.max()),
        'min_gate_score': float(mean_scores.min()),
        'std_gate_score': float(mean_scores.std()),
        'threshold': float(threshold)
    }


# ==============================================================================
# HELPER: Convert bytes to base64 (for JSON responses)
# ==============================================================================

def image_bytes_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string for JSON embedding"""
    return base64.b64encode(image_bytes).decode('utf-8')


def base64_to_image_bytes(base64_str: str) -> bytes:
    """Convert base64 string back to image bytes"""
    return base64.b64decode(base64_str)


# ==============================================================================
# 🆕 EXPORT FUNCTIONS FOR FASTAPI (WITH GATE SUPPORT)
# ==============================================================================

def get_patch_visualizations(
    patches: Dict[str, np.ndarray],
    aoi_name: str,
    patch_id: str,
    lat: float,
    lon: float,
    ai_results: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Generate all visualizations for a patch and return as base64
    
    Returns:
        Dictionary with base64-encoded images:
        {
            '2d_preview': 'base64...',
            '3d_terrain': {...},  # Plotly JSON
            'ai_analysis': 'base64...',  # Only if ai_results provided
            'ai_data': {...}  # Only if ai_results provided
        }
    """
    result = {}
    
    # 2D preview
    img_2d = generate_2d_patch_preview(patches, aoi_name, patch_id)
    result['2d_preview'] = image_bytes_to_base64(img_2d)
    
    # 3D terrain
    terrain_3d = generate_3d_terrain_json(patches, aoi_name, patch_id, lat, lon)
    result['3d_terrain'] = terrain_3d
    
    # AI analysis (if provided)
    if ai_results is not None:
        img_ai = generate_ai_analysis_visualization(patches, ai_results, patch_id)
        result['ai_analysis'] = image_bytes_to_base64(img_ai)
        result['ai_data'] = generate_ai_analysis_json(ai_results)
    
    return result


def get_gate_visualizations(
    unified_matrix: np.ndarray,
    metadata: List[Dict],
    aoi_shape: Tuple[int, int],
    aoi_name: str,
    patches_data: Optional[Dict[str, np.ndarray]] = None,
    threshold: float = 0.5
) -> Dict:
    """
    Generate all GATE-related visualizations for FastAPI
    
    Args:
        unified_matrix: Shape (num_patches, 64, 64, 5) with GATE channel
        metadata: Patch metadata
        aoi_shape: AOI dimensions
        aoi_name: AOI identifier
        patches_data: Original patch data (optional, for showing DTM)
        threshold: GATE classification threshold
    
    Returns:
        Dictionary with base64 images and statistics:
        {
            'gate_heatmap': 'base64...',
            'positive_patches': 'base64...',
            'statistics': {...}
        }
    """
    result = {}
    
    # 1. Full AOI GATE heatmap
    img_heatmap = generate_gate_prediction_heatmap(
        unified_matrix, metadata, aoi_shape, aoi_name, threshold
    )
    result['gate_heatmap'] = image_bytes_to_base64(img_heatmap)
    
    # 2. Top positive patches
    if patches_data is not None:
        img_positives = generate_gate_positive_patches_visualization(
            unified_matrix, metadata, patches_data, threshold
        )
        result['positive_patches'] = image_bytes_to_base64(img_positives)
    
    # 3. Statistics
    result['statistics'] = generate_gate_statistics_json(unified_matrix, threshold)
    
    return result