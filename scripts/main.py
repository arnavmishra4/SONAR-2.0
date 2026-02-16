"""
Archaeological Site Detection - Gradio Frontend
================================================
Interactive web interface for SONAR 2.0 archaeological detection system
with GATE model predictions and multi-model analysis.

Run: python gradio_app.py
"""

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from pathlib import Path
import torch
import json
from typing import Dict, List, Tuple, Optional
import io
from PIL import Image
import warnings
from config import config
warnings.filterwarnings('ignore')

# Import your existing utilities
from utils import (
    ResUNetAutoencoder, ResUNetEncoder, load_patches,
    load_model, load_kmeans_model, compute_autoencoder_probabilities,
    compute_iforest_probabilities, compute_kmeans_probabilities,
    load_unified_probability_matrix
)
from arch_similarity_utils import compute_arch_similarity_channel
from visualization import (
    generate_2d_patch_preview, generate_3d_terrain_json,
    generate_gate_prediction_heatmap, generate_gate_positive_patches_visualization,
    generate_probability_matrix_visualization, generate_full_aoi_heatmap,
    generate_gate_statistics_json
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Application configuration"""
    # Model paths
    AUTOENCODER_PATH = Path('models/best_model_aoi.pth')
    ENCODER_DIM = 128
    IFOREST_PATH = Path('models/isolation_forest_model_128dim.pkl')
    KMEANS_PATH = Path('models/kmeans_model_128dim.pkl')
    ARCH_EMBEDDINGS_CSV = Path('GATE/Arch_embedding_only_128dim.csv')
    GATE_MODEL_PKL = Path('GATE/gate_mlp_model.pkl')
    GATE_SCALER_PATH = Path('GATE/gate_scaler.pkl')
    
    # Data paths
    PATCHES_DIR = Path('patches/patches_final')
    UNIFIED_PROB_DIR = Path('src/unified_probability_matrices_with_gate')
    
    # Processing
    BATCH_SIZE = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # UI
    THEME = gr.themes.Soft(
        primary_hue=config.THEME_PRIMARY,
        secondary_hue=config.THEME_SECONDARY,
    )

config = Config()

# ==============================================================================
# MODEL MANAGER
# ==============================================================================

class ModelManager:
    """Manages all trained models"""
    
    def __init__(self):
        self.models_loaded = False
        self.autoencoder = None
        self.encoder = None
        self.iforest = None
        self.scaler_iforest = None
        self.kmeans = None
        self.scaler_kmeans = None
        self.gate_model = None
        self.gate_scaler = None
        
    def load_all_models(self):
        """Load all models"""
        status_messages = []
        
        try:
            # Autoencoder
            self.autoencoder = ResUNetAutoencoder(in_channels=7).to(config.DEVICE)
            self.autoencoder.load_state_dict(
                torch.load(str(config.AUTOENCODER_PATH), map_location=config.DEVICE)
            )
            self.autoencoder.eval()
            status_messages.append("✅ Autoencoder loaded")
            
            # Encoder
            self.encoder = ResUNetEncoder(
                in_channels=7, 
                embedding_dim=config.ENCODER_DIM
            ).to(config.DEVICE)
            self.encoder.load_from_autoencoder(str(config.AUTOENCODER_PATH))
            self.encoder.eval()
            status_messages.append("✅ Encoder loaded")
            
            # Isolation Forest
            self.iforest, self.scaler_iforest = load_model(str(config.IFOREST_PATH))
            status_messages.append("✅ Isolation Forest loaded")
            
            # K-Means
            self.kmeans, self.scaler_kmeans = load_kmeans_model(str(config.KMEANS_PATH))
            status_messages.append("✅ K-Means loaded")
            
            # GATE Model
            import joblib
            self.gate_model = joblib.load(str(config.GATE_MODEL_PKL))
            self.gate_scaler = joblib.load(str(config.GATE_SCALER_PATH))
            status_messages.append("✅ GATE model loaded")
            
            self.models_loaded = True
            status_messages.append("\n🎉 All models loaded successfully!")
            
        except Exception as e:
            status_messages.append(f"❌ Error loading models: {e}")
            import traceback
            status_messages.append(traceback.format_exc())
        
        return "\n".join(status_messages)
    
    def get_models_dict(self):
        """Return models as dictionary"""
        return {
            'autoencoder': self.autoencoder,
            'encoder': self.encoder,
            'iforest': self.iforest,
            'scaler_iforest': self.scaler_iforest,
            'kmeans': self.kmeans,
            'scaler_kmeans': self.scaler_kmeans,
            'gate_model': self.gate_model,
            'gate_scaler': self.gate_scaler
        }

# Global model manager
model_manager = ModelManager()

# ==============================================================================
# DATA MANAGER
# ==============================================================================

class DataManager:
    """Manages AOI data and patches"""
    
    def __init__(self):
        self.aoi_list = self._discover_aois()
        self.current_aoi = None
        self.current_patches = None
        self.current_metadata = None
        self.current_unified_matrix = None
        
    def _discover_aois(self):
        """Discover all available AOIs"""
        if not config.PATCHES_DIR.exists():
            return []
        
        patch_files = list(config.PATCHES_DIR.glob("AOI_*_all_patches.npz"))
        aoi_names = sorted([f.stem.replace('_all_patches', '') for f in patch_files])
        return aoi_names
    
    def load_aoi(self, aoi_name: str):
        """Load patches and unified matrix for an AOI"""
        try:
            # Load patches
            patches_file = config.PATCHES_DIR / f"{aoi_name}_all_patches.npz"
            self.current_patches, self.current_metadata = load_patches(patches_file)
            
            # Load unified probability matrix
            matrix_file = config.UNIFIED_PROB_DIR / f"{aoi_name}_unified_prob_matrix.npz"
            if matrix_file.exists():
                self.current_unified_matrix, _, _ = load_unified_probability_matrix(aoi_name, str(config.UNIFIED_PROB_DIR))
            else:
                self.current_unified_matrix = None
            
            self.current_aoi = aoi_name
            
            return f"✅ Loaded {aoi_name}: {len(self.current_patches)} patches"
        
        except Exception as e:
            return f"❌ Error loading {aoi_name}: {e}"
    
    def get_aoi_info(self):
        """Get current AOI information"""
        if self.current_aoi is None:
            return "No AOI loaded"
        
        info = [
            f"**AOI:** {self.current_aoi}",
            f"**Total Patches:** {len(self.current_patches)}",
            f"**Patch Shape:** {self.current_patches[0].shape}",
        ]
        
        if self.current_unified_matrix is not None:
            info.append(f"**Unified Matrix:** {self.current_unified_matrix.shape}")
            info.append(f"**Channels:** 5 (incl. GATE)")
        
        return "\n".join(info)
    
    def get_patch(self, patch_idx: int):
        """Get specific patch"""
        if self.current_patches is None or patch_idx >= len(self.current_patches):
            return None, None
        
        return self.current_patches[patch_idx], self.current_metadata[patch_idx]

# Global data manager
data_manager = DataManager()

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def create_patch_visualization(patch: np.ndarray, metadata: Dict):
    """Create 2D visualization of patch"""
    channel_names = ['DTM', 'Slope', 'Roughness', 'NDVI', 'NDWI', 'FlowAcc', 'FlowDir']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(7):
        ax = axes[i]
        data = patch[i]
        
        # Choose colormap
        if i == 0:  # DTM
            cmap = 'terrain'
        elif i == 1:  # Slope
            cmap = 'YlOrRd'
        elif i == 3:  # NDVI
            cmap = 'RdYlGn'
        elif i == 4:  # NDWI
            cmap = 'Blues'
        else:
            cmap = 'viridis'
        
        im = ax.imshow(data, cmap=cmap, interpolation='bilinear')
        ax.set_title(f'{channel_names[i]}\nμ={np.nanmean(data):.2f}', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide last subplot
    axes[7].axis('off')
    
    plt.suptitle(f"Patch {metadata['patch_id']} (row={metadata['row']}, col={metadata['col']})", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf)

def create_3d_terrain(patch: np.ndarray, metadata: Dict):
    """Create 3D terrain visualization"""
    dtm = patch[0]  # First channel is DTM
    
    # Create coordinate grids
    rows, cols = dtm.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # Handle NaN
    dtm_clean = np.nan_to_num(dtm, nan=np.nanmedian(dtm))
    
    # Create surface
    fig = go.Figure(data=[go.Surface(
        z=dtm_clean,
        x=X,
        y=Y,
        colorscale='earth',
        colorbar=dict(title='Elevation (m)'),
        lighting=dict(
            ambient=0.4,
            diffuse=0.8,
            fresnel=0.2,
            specular=0.3,
            roughness=0.5
        )
    )])
    
    fig.update_layout(
        title=f"3D Terrain - {metadata['patch_id']}",
        scene=dict(
            xaxis_title='X (pixels)',
            yaxis_title='Y (pixels)',
            zaxis_title='Elevation (m)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        width=800,
        height=600
    )
    
    return fig

def create_probability_channels_viz(unified_matrix: np.ndarray, patch_idx: int):
    """Visualize all 5 probability channels"""
    patch_data = unified_matrix[patch_idx]
    channel_names = ['Autoencoder', 'IForest', 'K-Means', 'Arch Similarity', 'GATE Prediction']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(5):
        ax = axes[i]
        data = patch_data[:, :, i]
        
        im = ax.imshow(data, cmap='hot', vmin=0, vmax=1, interpolation='bilinear')
        ax.set_title(f'{channel_names[i]}\nμ={data.mean():.3f}, max={data.max():.3f}', 
                     fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide last subplot
    axes[5].axis('off')
    
    plt.suptitle(f'Probability Channels - Patch {patch_idx}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf)

def create_gate_heatmap(unified_matrix: np.ndarray, metadata: List[Dict], aoi_name: str):
    """Create GATE prediction heatmap"""
    # Determine AOI shape
    max_row = max(m['row'] + 64 for m in metadata)
    max_col = max(m['col'] + 64 for m in metadata)
    aoi_shape = (max_row, max_col)
    
    # Generate heatmap
    heatmap_bytes = generate_gate_prediction_heatmap(
        unified_matrix, metadata, aoi_shape, aoi_name, threshold=0.5
    )
    
    return Image.open(io.BytesIO(heatmap_bytes))

def create_top_candidates(unified_matrix: np.ndarray, metadata: List[Dict], 
                         patches: np.ndarray, threshold: float = 0.5):
    """Visualize top archaeological candidates"""
    # Get patches as dict
    patches_dict = {'dtm': patches[:, 0, :, :]}
    
    candidates_bytes = generate_gate_positive_patches_visualization(
        unified_matrix, metadata, patches_dict, threshold=threshold, top_n=16
    )
    
    return Image.open(io.BytesIO(candidates_bytes))

# ==============================================================================
# GRADIO INTERFACE FUNCTIONS
# ==============================================================================

def load_models_ui():
    """Load models button handler"""
    status = model_manager.load_all_models()
    
    if model_manager.models_loaded:
        return status, gr.update(interactive=True)  # Enable analysis tab
    else:
        return status, gr.update(interactive=False)

def load_aoi_ui(aoi_name):
    """Load AOI handler"""
    status = data_manager.load_aoi(aoi_name)
    info = data_manager.get_aoi_info()
    
    # Update patch slider range
    if data_manager.current_patches is not None:
        max_patches = len(data_manager.current_patches) - 1
        return status, info, gr.update(maximum=max_patches, value=0)
    
    return status, info, gr.update()

def view_patch_ui(patch_idx):
    """View specific patch"""
    if data_manager.current_patches is None:
        return None, None, None, "⚠️ Load an AOI first"
    
    patch, metadata = data_manager.get_patch(patch_idx)
    
    if patch is None:
        return None, None, None, "❌ Invalid patch index"
    
    # Create visualizations
    img_2d = create_patch_visualization(patch, metadata)
    fig_3d = create_3d_terrain(patch, metadata)
    
    # Probability channels if available
    prob_img = None
    if data_manager.current_unified_matrix is not None:
        prob_img = create_probability_channels_viz(
            data_manager.current_unified_matrix, patch_idx
        )
    
    info = f"**Patch {patch_idx}:** {metadata['patch_id']}\n" \
           f"**Position:** (row={metadata['row']}, col={metadata['col']})"
    
    return img_2d, fig_3d, prob_img, info

def analyze_aoi_gate(threshold):
    """Analyze full AOI with GATE"""
    if data_manager.current_unified_matrix is None:
        return None, None, "⚠️ No unified matrix available for this AOI"
    
    # Generate heatmap
    heatmap = create_gate_heatmap(
        data_manager.current_unified_matrix,
        data_manager.current_metadata,
        data_manager.current_aoi
    )
    
    # Generate top candidates
    candidates = create_top_candidates(
        data_manager.current_unified_matrix,
        data_manager.current_metadata,
        data_manager.current_patches,
        threshold=threshold
    )
    
    # Statistics
    stats = generate_gate_statistics_json(
        data_manager.current_unified_matrix,
        threshold=threshold
    )
    
    stats_text = f"""
    **GATE Analysis Results**
    
    - Total Patches: {stats['total_patches']}
    - Positive Predictions: {stats['positive_patches']} ({stats['positive_percentage']:.1f}%)
    - Mean GATE Score: {stats['mean_gate_score']:.3f}
    - Max GATE Score: {stats['max_gate_score']:.3f}
    - Threshold: {stats['threshold']}
    """
    
    return heatmap, candidates, stats_text

def batch_process_ui(start_aoi, end_aoi, progress=gr.Progress()):
    """Batch process multiple AOIs"""
    if not model_manager.models_loaded:
        return "❌ Load models first"
    
    aoi_list = data_manager.aoi_list[start_aoi:end_aoi+1]
    results = []
    
    for i, aoi_name in enumerate(progress.tqdm(aoi_list, desc="Processing AOIs")):
        try:
            data_manager.load_aoi(aoi_name)
            
            if data_manager.current_unified_matrix is not None:
                stats = generate_gate_statistics_json(
                    data_manager.current_unified_matrix, threshold=0.5
                )
                
                results.append({
                    'AOI': aoi_name,
                    'Patches': stats['total_patches'],
                    'Positives': stats['positive_patches'],
                    'Positive %': f"{stats['positive_percentage']:.1f}%",
                    'Mean Score': f"{stats['mean_gate_score']:.3f}"
                })
        except Exception as e:
            results.append({
                'AOI': aoi_name,
                'Error': str(e)
            })
    
    df = pd.DataFrame(results)
    return df

# ==============================================================================
# BUILD GRADIO INTERFACE
# ==============================================================================

def build_interface():
    """Build the main Gradio interface"""
    
    with gr.Blocks(theme=THEME, title="Archaeological Site Detection") as app:
        
        # Header
        gr.Markdown("""
        # 🏛️ Archaeological Site Detection System
        ### SONAR 2.0 - Multi-Model Analysis with GATE Predictions
        
        Explore archaeological sites using advanced AI detection across multiple AOIs.
        """)
        
        # Model Loading Section
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 🤖 Step 1: Load AI Models")
                gr.Markdown("Load all trained models (Autoencoder, Isolation Forest, K-Means, GATE)")
            
            with gr.Column(scale=1):
                load_models_btn = gr.Button("🔄 Load Models", variant="primary", size="lg")
        
        model_status = gr.Textbox(label="Model Status", lines=8, max_lines=10)
        
        gr.Markdown("---")
        
        # Main Interface
        with gr.Tabs():
            
            # TAB 1: Patch Explorer
            with gr.Tab("🔍 Patch Explorer"):
                gr.Markdown("### Explore individual patches from any AOI")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        aoi_dropdown = gr.Dropdown(
                            choices=data_manager.aoi_list,
                            label="Select AOI",
                            value=data_manager.aoi_list[0] if data_manager.aoi_list else None
                        )
                        
                        load_aoi_btn = gr.Button("📂 Load AOI", variant="secondary")
                        
                        aoi_status = gr.Textbox(label="Load Status", lines=2)
                        aoi_info = gr.Markdown("*No AOI loaded*")
                        
                        gr.Markdown("---")
                        
                        patch_slider = gr.Slider(
                            minimum=0, maximum=100, step=1, value=0,
                            label="Select Patch"
                        )
                        
                        view_patch_btn = gr.Button("👁️ View Patch", variant="primary")
                        
                        patch_info = gr.Markdown("*Select a patch*")
                    
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("📊 2D View"):
                                patch_2d = gr.Image(label="Patch Layers", type="pil")
                            
                            with gr.Tab("🏔️ 3D Terrain"):
                                patch_3d = gr.Plot(label="3D Visualization")
                            
                            with gr.Tab("🎯 Probability Channels"):
                                prob_channels = gr.Image(label="5 Probability Channels", type="pil")
            
            # TAB 2: GATE Analysis
            with gr.Tab("🎯 GATE Analysis", id="gate_tab"):
                gr.Markdown("### Full AOI Analysis with GATE Model")
                
                with gr.Row():
                    threshold_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                        label="GATE Threshold (higher = stricter)"
                    )
                    
                    analyze_btn = gr.Button("🔥 Analyze AOI", variant="primary", size="lg")
                
                gate_stats = gr.Markdown("*Run analysis to see statistics*")
                
                with gr.Row():
                    with gr.Column():
                        gate_heatmap = gr.Image(label="GATE Prediction Heatmap", type="pil")
                    
                    with gr.Column():
                        gate_candidates = gr.Image(label="Top Archaeological Candidates", type="pil")
            
            # TAB 3: Batch Processing
            with gr.Tab("⚡ Batch Processing"):
                gr.Markdown("### Process Multiple AOIs")
                
                with gr.Row():
                    start_idx = gr.Number(label="Start AOI Index", value=0, precision=0)
                    end_idx = gr.Number(label="End AOI Index", value=10, precision=0)
                
                batch_btn = gr.Button("🚀 Process Batch", variant="primary")
                
                batch_results = gr.Dataframe(
                    headers=["AOI", "Patches", "Positives", "Positive %", "Mean Score"],
                    label="Batch Results"
                )
            
            # TAB 4: Documentation
            with gr.Tab("📖 Documentation"):
                gr.Markdown("""
                ## System Overview
                
                This system combines multiple AI models for archaeological site detection:
                
                ### Models
                1. **Autoencoder (ResUNet)** - Detects anomalies via reconstruction error
                2. **Isolation Forest** - Identifies outliers in latent space
                3. **K-Means Clustering** - Matches against known archaeological signatures
                4. **Archaeological Similarity** - Compares against verified sites
                5. **GATE Model** - Meta-learner that combines all models for final prediction
                
                ### Workflow
                1. Load models (one-time operation)
                2. Select an AOI to analyze
                3. Explore individual patches or run full AOI analysis
                4. Review GATE predictions to identify high-probability archaeological sites
                5. Batch process multiple AOIs for comprehensive surveys
                
                ### Channels
                The system uses 7 input channels:
                - **DTM**: Digital Terrain Model (LiDAR)
                - **Slope**: Terrain slope
                - **Roughness**: Surface roughness
                - **NDVI**: Normalized Difference Vegetation Index
                - **NDWI**: Normalized Difference Water Index
                - **Flow Accumulation**: Hydrological flow
                - **Flow Direction**: Water flow direction
                
                ### Output
                - **Probability Maps**: 5 probability channels (4 models + GATE)
                - **Heatmaps**: Full AOI visualization of predictions
                - **Candidate List**: Top archaeological site candidates
                
                ### Thresholding
                - **0.3-0.5**: Low threshold - more candidates, higher false positives
                - **0.5-0.7**: Balanced - recommended for initial surveys
                - **0.7-0.9**: High threshold - fewer candidates, high confidence
                """)
        
        # Event Handlers
        load_models_btn.click(
            fn=load_models_ui,
            outputs=[model_status, gr.Tabs.update(selected="gate_tab")]
        )
        
        load_aoi_btn.click(
            fn=load_aoi_ui,
            inputs=[aoi_dropdown],
            outputs=[aoi_status, aoi_info, patch_slider]
        )
        
        view_patch_btn.click(
            fn=view_patch_ui,
            inputs=[patch_slider],
            outputs=[patch_2d, patch_3d, prob_channels, patch_info]
        )
        
        analyze_btn.click(
            fn=analyze_aoi_gate,
            inputs=[threshold_slider],
            outputs=[gate_heatmap, gate_candidates, gate_stats]
        )
        
        batch_btn.click(
            fn=batch_process_ui,
            inputs=[start_idx, end_idx],
            outputs=[batch_results]
        )
    
    return app

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("🏛️ Archaeological Site Detection System")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Discovered AOIs: {len(data_manager.aoi_list)}")
    print("=" * 60)
    
    app = build_interface()
    
    # Launch with options
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public URL
        show_error=True,
        show_api=True
    )