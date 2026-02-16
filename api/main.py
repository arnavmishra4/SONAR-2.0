from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
from pathlib import Path
import numpy as np
import torch
import shutil
import tempfile
import zipfile
from typing import List, Optional
import json
import asyncio
from dataclasses import dataclass
from contextlib import asynccontextmanager

# NEW IMPORTS FOR ENHANCED ENDPOINTS
from scipy.ndimage import sobel, gaussian_filter
from matplotlib import cm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import base64
import io
import rasterio

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.predictor import SONARPredictor
from inference.preprocess import prepare_patches_from_upload

# ==============================================================================
# BATCHING INFRASTRUCTURE
# ==============================================================================

@dataclass
class InferenceRequest:
    """Single inference request with its response future"""
    patches: np.ndarray
    metadata: list
    threshold: float
    future: asyncio.Future

class BatchInferenceQueue:
    """
    Async batching queue for torch inference.
    Collects requests and processes them in batches for better GPU utilization.
    """
    def __init__(self, predictor: SONARPredictor, max_batch_size: int = 8, max_wait_time: float = 0.05):
        self.predictor = predictor
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = asyncio.Queue()
        self.processing_task = None
        
    async def start(self):
        """Start the background batch processing task"""
        self.processing_task = asyncio.create_task(self._process_batches())
        
    async def stop(self):
        """Stop the background processing task"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
    
    async def predict(self, patches: np.ndarray, metadata: list, threshold: float = 0.5):
        """
        Submit prediction request and wait for result.
        Automatically batches with other concurrent requests.
        """
        future = asyncio.Future()
        request = InferenceRequest(
            patches=patches,
            metadata=metadata,
            threshold=threshold,
            future=future
        )
        await self.queue.put(request)
        return await future
    
    async def _process_batches(self):
        """Background task that processes requests in batches"""
        while True:
            try:
                batch = []
                
                # Get first request (wait indefinitely)
                first_request = await self.queue.get()
                batch.append(first_request)
                
                # Try to collect more requests up to max_batch_size or max_wait_time
                deadline = asyncio.get_event_loop().time() + self.max_wait_time
                
                while len(batch) < self.max_batch_size:
                    timeout = deadline - asyncio.get_event_loop().time()
                    if timeout <= 0:
                        break
                    
                    try:
                        request = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break
                
                # Process batch
                await self._process_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
                # Set exceptions for all requests in batch
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(e)
    
    async def _process_batch(self, batch: List[InferenceRequest]):
        """Process a batch of requests using the predictor"""
        try:
            # For single request, process directly
            if len(batch) == 1:
                request = batch[0]
                result = await asyncio.to_thread(
                    self.predictor.predict_aoi,
                    request.patches,
                    request.metadata,
                    request.threshold
                )
                request.future.set_result(result)
                return
            
            # For multiple requests, batch them together
            # Process each request in parallel using thread pool
            tasks = [
                asyncio.to_thread(
                    self.predictor.predict_aoi,
                    request.patches,
                    request.metadata,
                    request.threshold
                )
                for request in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Set results or exceptions
            for request, result in zip(batch, results):
                if isinstance(result, Exception):
                    request.future.set_exception(result)
                else:
                    request.future.set_result(result)
                    
        except Exception as e:
            # Set exception for all requests
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)


# ==============================================================================
# LIFESPAN MANAGEMENT
# ==============================================================================

# Global instances
predictor = None
batch_queue = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    global predictor, batch_queue
    
    # Startup
    print("🚀 Starting SONAR 2.0 API...")
    
    # Get absolute path to project root
    project_root = Path(__file__).parent.parent
    checkpoints_dir = project_root / 'checkpoints'
    
    # Debug prints
    print(f"🔍 Project root: {project_root}")
    print(f"🔍 Checkpoints dir: {checkpoints_dir}")
    print(f"🔍 Checkpoints exists: {checkpoints_dir.exists()}")
    
    config = {
        'autoencoder_path': str(checkpoints_dir / 'best_model_aoi.pth'),
        'iforest_path': str(checkpoints_dir / 'isolation_forest_model.pkl'),
        'kmeans_path': str(checkpoints_dir / 'kmeans_model.pkl'),
        'gate_path': str(checkpoints_dir / 'GATE_model.pt'),
        'reference_embeddings_path': None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Verify files exist
    for key, path in config.items():
        if path and key != 'device' and key != 'reference_embeddings_path':
            if not Path(path).exists():
                print(f"⚠️  WARNING: {key} not found at {path}")
            else:
                print(f"✅ Found {key}: {path}")
    
    predictor = SONARPredictor(config)
    print("✅ Models loaded!")
    
    # Initialize batch queue
    batch_queue = BatchInferenceQueue(predictor, max_batch_size=8, max_wait_time=0.05)
    await batch_queue.start()
    print("✅ Batch inference queue started!")
    
    print("✅ SONAR 2.0 API Ready!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down SONAR 2.0 API...")
    if batch_queue:
        await batch_queue.stop()
    print("✅ Shutdown complete!")


# ==============================================================================
# APP INITIALIZATION
# ==============================================================================

app = FastAPI(
    title="SONAR 2.0 API",
    description="Archaeological Intelligence System - End-to-End Processing",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# HEALTH & DEMO ENDPOINTS
# ==============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SONAR 2.0",
        "models_loaded": predictor is not None,
        "batch_queue_active": batch_queue is not None
    }

@app.get("/api/demo-data")
async def get_demo_data():
    """
    Serve the demo_data.json file for the frontend demo map
    """
    try:
        # Get project root (parent of api folder)
        project_root = Path(__file__).parent.parent
        demo_data_path = project_root / 'demo_data.json'
        
        print(f"Looking for demo_data.json at: {demo_data_path}")
        
        if not demo_data_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"demo_data.json not found at {demo_data_path}"
            )
        
        # Read and return the JSON data
        with open(demo_data_path, 'r', encoding='utf-8') as f:
            demo_data = json.load(f)
        
        print(f"✅ Successfully loaded demo data with {len(demo_data.get('all_anomalies', []))} anomalies")
        
        return demo_data
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Demo data file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON in demo_data.json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading demo data: {str(e)}")


# ==============================================================================
# NEW: FULL PIPELINE - RAW FILES → PREDICTION (WITH BATCHING)
# ==============================================================================

@app.post("/api/analyze")
async def analyze_raw_data(
    lidar_file: UploadFile = File(..., description="LiDAR DTM .tif file"),
    sentinel_files: List[UploadFile] = File(..., description="Sentinel-2 files (NDVI, NDWI, or bands)"),
    hydro_files: List[UploadFile] = File(..., description="HydroSHED files (flow_acc, flow_dir)")
):
    """
    FULL PIPELINE: Process raw geospatial data and run predictions
    
    User uploads:
    - LiDAR: Single .tif file (DTM)
    - Sentinel: Multiple .tif files (NDVI, NDWI or B03, B04, B08)
    - HydroSHED: Multiple .tif files (flow_acc.tif, flow_dir.tif)
    
    Returns:
    - Anomaly predictions
    - Top candidates with coordinates
    - Analysis summary
    """
    if predictor is None or batch_queue is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Create temporary directory structure
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        print(f"\n{'='*60}")
        print("🚀 SONAR 2.0 - FULL ANALYSIS PIPELINE")
        print(f"{'='*60}")
        
        # Create directory structure
        aoi_dir = temp_dir / "AOI_0000"
        lidar_dir = temp_dir / "lidar"
        sentinel_dir = aoi_dir / "sentinel"
        hydro_dir = aoi_dir / "hydro"
        meta_dir = aoi_dir / "meta"
        output_dir = temp_dir / "output"
        
        for d in [lidar_dir, sentinel_dir, hydro_dir, meta_dir, output_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Step 1/4: Saving uploaded files...")
        
        # Save LiDAR file
        lidar_path = lidar_dir / lidar_file.filename
        with open(lidar_path, 'wb') as f:
            shutil.copyfileobj(lidar_file.file, f)
        print(f"   ✅ LiDAR: {lidar_file.filename}")
        
        # Also copy to meta for processing
        shutil.copy(lidar_path, meta_dir / lidar_file.filename)
        
        # Save Sentinel files
        sentinel_file_map = {}
        for sf in sentinel_files:
            file_path = sentinel_dir / sf.filename
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(sf.file, f)
            sentinel_file_map[sf.filename.lower()] = file_path
            print(f"   ✅ Sentinel: {sf.filename}")
        
        # Save HydroSHED files
        for hf in hydro_files:
            file_path = hydro_dir / hf.filename
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(hf.file, f)
            print(f"   ✅ HydroSHED: {hf.filename}")
        
        # Create coordinates.txt for preprocessing
        coords_file = meta_dir / "coordinates.txt"
        with open(coords_file, 'w') as f:
            f.write(f"AOI_ID: AOI_0000\n")
            f.write(f"LiDAR_File: {lidar_file.filename.replace('.tif', '.laz')}\n")
            # These will be computed from the actual raster, these are placeholders
            f.write(f"min_lon: 0\n")
            f.write(f"min_lat: 0\n")
            f.write(f"max_lon: 0\n")
            f.write(f"max_lat: 0\n")
        
        # Ensure we have NDVI and NDWI
        # If not provided, try to compute from bands
        ndvi_path = sentinel_dir / "ndvi.tif"
        ndwi_path = sentinel_dir / "ndwi.tif"
        
        if not ndvi_path.exists():
            # Try to find in uploaded files
            for name, path in sentinel_file_map.items():
                if 'ndvi' in name:
                    shutil.copy(path, ndvi_path)
                    break
        
        if not ndwi_path.exists():
            for name, path in sentinel_file_map.items():
                if 'ndwi' in name:
                    shutil.copy(path, ndwi_path)
                    break
        
        # Verify we have essential files
        if not ndvi_path.exists():
            raise HTTPException(
                status_code=400, 
                detail="NDVI file required. Please upload ndvi.tif or provide Sentinel bands to compute it."
            )
        
        print(f"\n🔧 Step 2/4: Preprocessing into 7-channel patches...")
        
        # Preprocessing configuration
        config = {
            'patch_size': 64,
            'stride': 32,
            'target_patches': 100,  # Extract more patches for better coverage
            'min_valid_ratio': 0.7
        }
        
        # Run preprocessing
        try:
            patches, metadata = prepare_patches_from_upload(
                hydro_dir=temp_dir,
                sentinel_dir=temp_dir,
                lidar_dir=lidar_dir,
                output_dir=output_dir,
                config=config
            )
            
            print(f"   ✅ Extracted {len(patches)} patches")
            print(f"   ✅ Patch shape: {patches.shape}")
            
        except Exception as e:
            print(f"   ❌ Preprocessing failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Preprocessing failed: {str(e)}. Please check your input files."
            )
        
        print(f"\n🤖 Step 3/4: Running AI inference (batched)...")
        
        # Run prediction using batch queue
        results = await batch_queue.predict(patches, metadata, threshold=0.5)
        
        print(f"   ✅ Detected {results['summary']['anomalies_detected']} anomalies")
        
        print(f"\n📊 Step 4/4: Formatting results...")
        
        # Format top candidates with coordinates
        top_candidates = []
        for i, candidate in enumerate(results['summary']['top_candidates'][:10]):
            # Get patch center pixel coordinates
            row = candidate.get('row', 0) + 32
            col = candidate.get('col', 0) + 32
            
            # For now, use placeholder coordinates
            # In production, you'd convert pixel coords to lat/lon using the raster transform
            top_candidates.append({
                "rank": i + 1,
                "patch_id": candidate['patch_id'],
                "confidence": float(candidate['confidence']),
                "coordinates": {
                    "lat": 0.0,  # TODO: Convert from pixel coordinates
                    "lng": 0.0
                },
                "pixel_location": {
                    "row": int(row),
                    "col": int(col)
                }
            })
        
        # Create response
        response = {
            "success": True,
            "summary": {
                "total_patches": results['summary']['total_patches'],
                "anomalies_detected": results['summary']['anomalies_detected'],
                "anomaly_percentage": float(results['summary']['anomaly_percentage']),
                "mean_confidence": float(results['summary']['mean_confidence'])
            },
            "top_candidates": top_candidates,
            "processing_info": {
                "patch_size": config['patch_size'],
                "total_patches_analyzed": len(patches),
                "channels": 7
            }
        }
        
        print(f"\n{'='*60}")
        print("✅ ANALYSIS COMPLETE")
        print(f"{'='*60}\n")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary files
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


# ==============================================================================
# LEGACY ENDPOINTS (for pre-processed data) - WITH BATCHING
# ==============================================================================

@app.post("/api/ml/predict")
async def predict_patches(
    data_file: UploadFile = File(...)
):
    """
    Predict on preprocessed patches (LEGACY ENDPOINT)
    
    Expects .npz file with:
    - 'patches': array of shape (N, 7, 64, 64)
    - 'metadata': list of dicts with patch info
    """
    if predictor is None or batch_queue is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Read uploaded .npz file
        contents = await data_file.read()
        
        # Save temporarily
        temp_path = f"/tmp/{data_file.filename}"
        with open(temp_path, 'wb') as f:
            f.write(contents)
        
        # Load data
        data = np.load(temp_path, allow_pickle=True)
        patches = data['patches']
        metadata = data.get('metadata', [{'patch_id': f'p{i}'} for i in range(len(patches))])
        
        # Convert metadata from numpy to list if needed
        if isinstance(metadata, np.ndarray):
            metadata = metadata.tolist()
        
        # Run prediction using batch queue
        results = await batch_queue.predict(patches, metadata, threshold=0.5)
        
        return {
            "success": True,
            "summary": results['summary'],
            "top_candidates": results['summary']['top_candidates'][:10],
            "total_predictions": len(results['predictions'])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ml/predict-single")
async def predict_single(
    patch_file: UploadFile = File(...)
):
    """
    Predict on a single patch (LEGACY ENDPOINT)
    
    Expects .npy file with shape (7, 64, 64)
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        contents = await patch_file.read()
        temp_path = f"/tmp/{patch_file.filename}"
        
        with open(temp_path, 'wb') as f:
            f.write(contents)
        
        patch = np.load(temp_path)
        
        if patch.shape != (7, 64, 64):
            raise HTTPException(
                status_code=400, 
                detail=f"Expected shape (7, 64, 64), got {patch.shape}"
            )
        
        # Single patch prediction - use direct predictor (no batching needed)
        result = await asyncio.to_thread(predictor.predict_single_patch, patch)
        
        return {
            "success": True,
            "gate_prediction": result['gate_prediction'],
            "is_anomaly": result['is_anomaly'],
            "model_scores": result['model_scores']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# NEW ENHANCED ENDPOINTS - TERRAIN OVERLAYS & PATCH DETAILS
# ==============================================================================

@app.get("/api/terrain/{aoi_id}")
async def get_terrain_overlay(
    aoi_id: str,
    layer_type: str = "relief"  # 'relief', 'hillshade', 'basic'
):
    """
    Generate terrain overlay from DTM for a specific AOI
    Returns base64-encoded PNG image with bounds
    
    Layer types:
    - relief: Local Relief Model (archaeological features pop!)
    - hillshade: Multi-Directional Hillshade (3D terrain feel)
    - basic: Standard terrain coloring
    """
    try:
        # Get project root
        project_root = Path(__file__).parent.parent
        
        # Try multiple possible locations for Test_dataset
        possible_paths = [
            project_root / 'Test_data' / 'Test Dataset' / aoi_id / 'meta',
            project_root / 'Test_data' / 'Test_Dataset' / aoi_id / 'meta',
            project_root / 'Test_Dataset' / aoi_id / 'meta',
            project_root / 'Test_data' / aoi_id / 'meta',
        ]
        
        meta_dir = None
        for path in possible_paths:
            if path.exists():
                tif_files = list(path.glob('*.tif'))
                if tif_files:
                    meta_dir = path
                    break
        
        if meta_dir is None or not meta_dir.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"No DTM data found for {aoi_id}"
            )
        
        # Find DTM file
        tif_files = list(meta_dir.glob('*.tif'))
        
        if not tif_files:
            raise HTTPException(
                status_code=404, 
                detail=f"No .tif file found in {meta_dir}"
            )
        
        dtm_path = tif_files[0]
        
        # Load DTM with rasterio (CPU-intensive, run in thread)
        def load_and_process_dtm():
            with rasterio.open(dtm_path) as src:
                dtm = src.read(1).astype(np.float32)
                bounds = src.bounds
                crs = src.crs
                
                # Handle nodata
                if src.nodata is not None:
                    dtm[dtm == src.nodata] = np.nan
            
            valid_mask = ~np.isnan(dtm)
            
            if not valid_mask.any():
                raise ValueError("All DTM values are NaN")
            
            dtm_filled = dtm.copy()
            dtm_filled[~valid_mask] = np.nanmedian(dtm)
            
            # Generate overlay based on type
            if layer_type == "relief":
                # Local Relief Model (ARCHAEOLOGICAL GOLD!)
                dtm_smooth = gaussian_filter(dtm_filled, sigma=10)
                local_relief = dtm_filled - dtm_smooth
                
                relief_clipped = np.clip(local_relief, -2, 2)
                relief_norm = (relief_clipped + 2) / 4
                
                rdbu_cmap = cm.get_cmap('RdBu_r')
                relief_rgba = rdbu_cmap(relief_norm)
                relief_rgb = (relief_rgba[:, :, :3] * 255).astype(np.uint8)
                relief_rgb[~valid_mask] = [128, 128, 128]
                
                overlay_img = Image.fromarray(relief_rgb, mode='RGB')
                
            elif layer_type == "hillshade":
                # Multi-Directional Hillshade
                dx = sobel(dtm_filled, axis=1) / 8.0
                dy = sobel(dtm_filled, axis=0) / 8.0
                slope = np.arctan(np.sqrt(dx**2 + dy**2))
                aspect = np.arctan2(-dy, dx)
                
                azimuths = [315, 45, 225, 135]
                altitude = 45
                hillshades = []
                
                for az_deg in azimuths:
                    azimuth = np.radians(az_deg)
                    alt_rad = np.radians(altitude)
                    hs = (np.sin(alt_rad) * np.sin(slope) + 
                          np.cos(alt_rad) * np.cos(slope) * 
                          np.cos(azimuth - aspect))
                    hillshades.append(hs)
                
                hillshade_multi = np.mean(hillshades, axis=0)
                hillshade_multi = np.clip(hillshade_multi, -1, 1)
                hillshade_multi = ((hillshade_multi + 1) / 2 * 255).astype(np.uint8)
                hillshade_multi[~valid_mask] = 128
                
                # Add color tinting
                dtm_norm = (dtm - np.nanpercentile(dtm[valid_mask], 2)) / \
                           (np.nanpercentile(dtm[valid_mask], 98) - 
                            np.nanpercentile(dtm[valid_mask], 2))
                dtm_norm = np.clip(dtm_norm, 0, 1)
                
                terrain_cmap = cm.get_cmap('terrain')
                terrain_rgba = terrain_cmap(dtm_norm)
                terrain_rgb = (terrain_rgba[:, :, :3] * 255).astype(np.uint8)
                
                hillshade_rgb = np.stack([hillshade_multi, hillshade_multi, hillshade_multi], axis=-1)
                hillshade_rgb = (hillshade_rgb * 0.75 + terrain_rgb * 0.25).astype(np.uint8)
                hillshade_rgb[~valid_mask] = [128, 128, 128]
                
                overlay_img = Image.fromarray(hillshade_rgb, mode='RGB')
                
            else:  # basic terrain
                dtm_norm = (dtm - np.nanpercentile(dtm[valid_mask], 2)) / \
                           (np.nanpercentile(dtm[valid_mask], 98) - 
                            np.nanpercentile(dtm[valid_mask], 2))
                dtm_norm = np.clip(dtm_norm, 0, 1)
                
                terrain_cmap = cm.get_cmap('terrain')
                terrain_rgba = terrain_cmap(dtm_norm)
                terrain_rgb = (terrain_rgba[:, :, :3] * 255).astype(np.uint8)
                terrain_rgb[~valid_mask] = [128, 128, 128]
                
                overlay_img = Image.fromarray(terrain_rgb, mode='RGB')
            
            # Convert to base64
            buffered = io.BytesIO()
            overlay_img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Convert CRS if needed
            if crs.to_string() != 'EPSG:4326':
                from rasterio.warp import transform as transform_coords
                xs = [bounds.left, bounds.right]
                ys = [bounds.bottom, bounds.top]
                lons, lats = transform_coords(crs, 'EPSG:4326', xs, ys)
                min_lon, max_lon = min(lons), max(lons)
                min_lat, max_lat = min(lats), max(lats)
            else:
                min_lon, max_lon = bounds.left, bounds.right
                min_lat, max_lat = bounds.bottom, bounds.top
            
            return img_base64, min_lat, max_lat, min_lon, max_lon
        
        # Run in thread pool
        img_base64, min_lat, max_lat, min_lon, max_lon = await asyncio.to_thread(load_and_process_dtm)
        
        return {
            "success": True,
            "aoi_id": aoi_id,
            "layer_type": layer_type,
            "image": f"data:image/png;base64,{img_base64}",
            "bounds": {
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating terrain: {str(e)}"
        )


@app.get("/api/patch/{aoi_id}/{patch_id}/visualization")
async def get_patch_visualization(
    aoi_id: str,
    patch_id: str
):
    """
    Generate 7-channel visualization for a specific patch
    Returns base64-encoded PNG
    """
    try:
        project_root = Path(__file__).parent.parent
        
        # Load patches file
        patches_file = project_root / 'Test_data' / 'patches_final_file' / f"{aoi_id}_all_patches.npz"
        
        if not patches_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Patches file not found for {aoi_id}"
            )
        
        def generate_visualization():
            # Load data
            data = np.load(patches_file, allow_pickle=True)
            patches = data['patches']
            metadata = data['metadata']
            
            # Find patch by ID
            patch_idx = None
            for idx, meta in enumerate(metadata):
                if meta['patch_id'] == patch_id:
                    patch_idx = idx
                    break
            
            if patch_idx is None:
                raise ValueError(f"Patch {patch_id} not found in {aoi_id}")
            
            patch = patches[patch_idx]
            meta = metadata[patch_idx]
            
            # Generate 7-channel visualization
            channel_names = ['DTM', 'Slope', 'Roughness', 'NDVI', 'NDWI', 'Flow Acc', 'Flow Dir']
            
            fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor='white')
            axes = axes.flatten()
            
            for i in range(7):
                ax = axes[i]
                data_channel = patch[i]
                cmap = ['terrain', 'YlOrRd', 'viridis', 'RdYlGn', 'Blues', 'cividis', 'twilight'][i]
                
                im = ax.imshow(data_channel, cmap=cmap, interpolation='bilinear')
                ax.set_title(channel_names[i], fontsize=12, fontweight='bold', pad=8)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            axes[7].axis('off')
            
            plt.suptitle(
                f"Patch {meta['patch_id']} | Row {meta['row']} | Col {meta['col']}", 
                fontsize=16, fontweight='bold', y=0.98
            )
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)
            
            img_base64 = base64.b64encode(buf.getvalue()).decode()
            
            return img_base64, meta
        
        # Run in thread pool
        img_base64, meta = await asyncio.to_thread(generate_visualization)
        
        return {
            "success": True,
            "aoi_id": aoi_id,
            "patch_id": patch_id,
            "image": f"data:image/png;base64,{img_base64}",
            "metadata": {
                "row": int(meta['row']),
                "col": int(meta['col']),
                "patch_id": meta['patch_id']
            }
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating patch visualization: {str(e)}"
        )


@app.get("/api/patch/{aoi_id}/{patch_id}/terrain3d")
async def get_patch_3d_terrain(
    aoi_id: str,
    patch_id: str
):
    """
    Generate 3D terrain visualization for a specific patch
    Returns Plotly JSON
    """
    try:
        project_root = Path(__file__).parent.parent
        
        # Load patches file
        patches_file = project_root / 'Test_data' / 'patches_final_file' / f"{aoi_id}_all_patches.npz"
        
        if not patches_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Patches file not found for {aoi_id}"
            )
        
        def generate_3d_terrain():
            # Load data
            data = np.load(patches_file, allow_pickle=True)
            patches = data['patches']
            metadata = data['metadata']
            
            # Find patch by ID
            patch_idx = None
            for idx, meta in enumerate(metadata):
                if meta['patch_id'] == patch_id:
                    patch_idx = idx
                    break
            
            if patch_idx is None:
                raise ValueError(f"Patch {patch_id} not found")
            
            patch = patches[patch_idx]
            meta = metadata[patch_idx]
            
            # Extract DTM (channel 0)
            dtm = patch[0]
            dtm_clean = np.nan_to_num(dtm, nan=np.nanmedian(dtm))
            
            rows, cols = dtm.shape
            x, y = np.arange(cols), np.arange(rows)
            X, Y = np.meshgrid(x, y)
            
            # Create 3D surface plot
            fig = go.Figure(data=[go.Surface(
                z=dtm_clean,
                x=X,
                y=Y,
                colorscale='earth',
                showscale=True,
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
            
            fig.update_layout(
                title=f"3D Terrain · Patch {meta['patch_id']}",
                scene=dict(
                    xaxis_title='X (pixels)',
                    yaxis_title='Y (pixels)',
                    zaxis_title='Elevation (m)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=0.5)
                ),
                height=600,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            # Convert to JSON
            fig_json = fig.to_json()
            
            return fig_json, meta
        
        # Run in thread pool
        fig_json, meta = await asyncio.to_thread(generate_3d_terrain)
        
        return {
            "success": True,
            "aoi_id": aoi_id,
            "patch_id": patch_id,
            "plotly_json": json.loads(fig_json)
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating 3D terrain: {str(e)}"
        )


@app.get("/api/patch/{aoi_id}/{patch_id}/scores")
async def get_patch_scores(
    aoi_id: str,
    patch_id: str
):
    """
    Get model scores for a specific patch
    Returns individual model scores and GATE final score
    """
    try:
        project_root = Path(__file__).parent.parent
        
        # Load unified probability matrix
        matrix_file = project_root / 'Test_data' / 'test_unified_probablity_matrices_with_gate' / f"{aoi_id}_unified_prob_matrix.npz"
        
        if not matrix_file.exists():
            return {
                "success": False,
                "message": "Model scores not available for this AOI"
            }
        
        def load_scores():
            # Load patches metadata to find index
            patches_file = project_root / 'Test_data' / 'patches_final_file' / f"{aoi_id}_all_patches.npz"
            data = np.load(patches_file, allow_pickle=True)
            metadata = data['metadata']
            
            # Find patch index
            patch_idx = None
            for idx, meta in enumerate(metadata):
                if meta['patch_id'] == patch_id:
                    patch_idx = idx
                    break
            
            if patch_idx is None:
                raise ValueError(f"Patch {patch_id} not found")
            
            # Load unified matrix
            matrix_data = np.load(matrix_file)
            unified_matrix = matrix_data['unified_prob_matrix']
            
            # Extract scores for this patch
            patch_matrix = unified_matrix[patch_idx]
            
            ae_score = float(np.mean(patch_matrix[:, :, 0]))
            if_score = float(np.mean(patch_matrix[:, :, 1]))
            kmeans_score = float(np.mean(patch_matrix[:, :, 2]))
            similarity_score = float(np.mean(patch_matrix[:, :, 3]))
            gate_score = float(np.mean(patch_matrix[:, :, 4]))
            
            return ae_score, if_score, kmeans_score, similarity_score, gate_score
        
        # Run in thread pool
        ae_score, if_score, kmeans_score, similarity_score, gate_score = await asyncio.to_thread(load_scores)
        
        return {
            "success": True,
            "aoi_id": aoi_id,
            "patch_id": patch_id,
            "scores": {
                "autoencoder": ae_score,
                "isolation_forest": if_score,
                "kmeans": kmeans_score,
                "similarity": similarity_score,
                "gate_final": gate_score
            }
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model scores: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )