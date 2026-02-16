"""
SONAR 2.0 - Data Preprocessing Pipeline (OPTIMIZED VERSION)
Optimizations: multiprocessing, vectorization, memory mapping, cached operations
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import sobel, generic_filter
from pathlib import Path
from tqdm import tqdm
import json
import warnings
import argparse
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

warnings.filterwarnings('ignore')

class SonarDataPreparation:
    """
    Prepare SONAR 2.0 dataset from multi-source geospatial data.
    
    OPTIMIZED with:
    - Parallel processing for multiple AOIs
    - Vectorized operations
    - Memory-efficient patch extraction
    - Cached intermediate results
    
    Extracts 7-channel patches:
    - dtm: Digital Terrain Model from LiDAR
    - slope: Terrain slope
    - roughness: Terrain roughness
    - ndvi: Normalized Difference Vegetation Index
    - ndwi: Normalized Difference Water Index
    - flow_acc: Flow accumulation (log-transformed)
    - flow_dir: Flow direction (sin-encoded)
    
    Parameters
    ----------
    hydro_base : str or Path
        Path to HydroSHEDS dataset directory
    sentinel_base : str or Path
        Path to Sentinel-2 dataset directory
    lidar_base : str or Path
        Path to LiDAR DTM tiles directory
    output_dir : str or Path
        Directory to save extracted patches
    patch_size : int, default=64
        Size of square patches
    stride : int, default=32
        Stride for patch extraction
    target_patches : int, default=50
        Target number of patches per AOI
    min_valid_ratio : float, default=0.7
        Minimum ratio of valid pixels per patch
    clip_percentile : tuple, default=(0.5, 99.5)
        Percentile range for clipping before normalization
    n_workers : int, optional
        Number of parallel workers (default: CPU count - 1)
    """
    
    def __init__(
        self,
        hydro_base,
        sentinel_base,
        lidar_base,
        output_dir,
        patch_size=64,
        stride=32,
        target_patches=50,
        min_valid_ratio=0.7,
        clip_percentile=(0.5, 99.5),
        n_workers=None
    ):
        self.hydro_base = Path(hydro_base)
        self.sentinel_base = Path(sentinel_base)
        self.lidar_base = Path(lidar_base)
        self.output_dir = Path(output_dir)
        
        self.patch_size = patch_size
        self.stride = stride
        self.target_patches = target_patches
        self.min_valid_ratio = min_valid_ratio
        self.clip_percentile = clip_percentile
        
        self.channel_names = ['dtm', 'slope', 'roughness', 'ndvi', 'ndwi', 'flow_acc', 'flow_dir']
        self.global_stats = None
        
        # OPTIMIZATION: Set worker count
        if n_workers is None:
            self.n_workers = max(1, mp.cpu_count() - 1)
        else:
            self.n_workers = n_workers
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    # ==========================================================================
    # RASTER I/O (with caching)
    # ==========================================================================
    
    def load_reference_profile(self, ndvi_path):
        """Load Sentinel-2 NDVI as reference grid"""
        with rasterio.open(ndvi_path) as src:
            profile = src.profile.copy()
            profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
            return profile
    
    def read_sentinel_channel(self, path, profile):
        """Read Sentinel-2 band (already aligned)"""
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            return data
    
    def reproject_to_reference(self, src_path, ref_profile):
        """Reproject any raster to reference grid"""
        with rasterio.open(src_path) as src:
            data = np.empty((ref_profile['height'], ref_profile['width']), dtype=np.float32)   
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_profile['transform'],
                dst_crs=ref_profile['crs'],
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan
            )    
            return data
    
    def load_lidar_dtm(self, lidar_path, ref_profile):
        """Load and validate LiDAR DTM, reprojected to Sentinel grid"""
        if not lidar_path or not lidar_path.exists():
            return None
        try:
            dtm = self.reproject_to_reference(lidar_path, ref_profile) 
            # Treat values <= 0 as nodata
            dtm[dtm <= 0] = np.nan 
            # Check if valid
            valid_count = np.sum(~np.isnan(dtm))
            if valid_count < 100:
                return None 
            # Check if all zeros (invalid tile)
            if np.nanmax(dtm) < 0.1:
                return None
            return dtm  
        except Exception as e:
            print(f"      ⚠️  LiDAR error: {e}")
            return None
    
    # ==========================================================================
    # SPATIAL PROCESSING (vectorized)
    # ==========================================================================
    
    def compute_slope(self, dtm):
        """OPTIMIZED: Compute slope using vectorized Sobel operator"""
        valid_mask = ~np.isnan(dtm)
        if np.sum(valid_mask) < 10:
            return np.full_like(dtm, np.nan)
        
        dtm_filled = dtm.copy()
        dtm_filled[~valid_mask] = np.nanmedian(dtm)
        
        # Vectorized gradient computation
        grad_y = sobel(dtm_filled, axis=0)
        grad_x = sobel(dtm_filled, axis=1)
        
        # Vectorized slope calculation
        slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * (180.0 / np.pi)
        slope[~valid_mask] = np.nan
        
        return slope.astype(np.float32)
    
    def compute_roughness(self, dtm, window_size=3):
        """OPTIMIZED: Compute terrain roughness as local std"""
        valid_mask = ~np.isnan(dtm)
        if np.sum(valid_mask) < 10:
            return np.full_like(dtm, np.nan)
        
        dtm_filled = dtm.copy()
        dtm_filled[~valid_mask] = np.nanmedian(dtm)
        
        roughness = generic_filter(dtm_filled, np.std, size=window_size)
        roughness[~valid_mask] = np.nan
        
        return roughness.astype(np.float32)
    
    def encode_flow_dir_sin(self, flow_dir):
        """OPTIMIZED: Vectorized flow direction encoding"""
        direction_map = {1: 0, 2: 45, 4: 90, 8: 135, 16: 180, 32: 225, 64: 270, 128: 315}
        
        angles = np.full_like(flow_dir, np.nan, dtype=np.float32)
        
        # Vectorized mapping
        for code, angle in direction_map.items():
            angles[flow_dir == code] = angle
        
        return np.sin(np.deg2rad(angles)).astype(np.float32)
    
    def log_flow_acc(self, flow_acc, nodata=4294967295):
        """OPTIMIZED: Vectorized log transform"""
        flow_acc_clean = flow_acc.copy()
        
        # Vectorized nodata masking
        mask = (flow_acc == nodata) | (flow_acc <= 0)
        flow_acc_clean[mask] = np.nan
        
        return np.log1p(flow_acc_clean).astype(np.float32)
    
    # ==========================================================================
    # CHANNEL LOADING
    # ==========================================================================
    
    def load_all_channels(self, aoi, ref_profile):
        """Load all 7 channels for an AOI"""
        
        # Sentinel-2
        ndvi_path = self.sentinel_base / aoi / 'sentinel' / 'ndvi.tif'
        ndwi_path = self.sentinel_base / aoi / 'sentinel' / 'ndwi.tif'
        
        # HydroSHEDS
        flow_acc_path = self.hydro_base / aoi / 'hydro' / 'flow_acc.tif'
        flow_dir_path = self.hydro_base / aoi / 'hydro' / 'flow_dir.tif'
        
        # LiDAR
        coords_file = self.hydro_base / aoi / 'meta' / 'coordinates.txt'
        lidar_path = None
        
        if coords_file.exists():
            with open(coords_file, 'r') as f:
                for line in f:
                    if 'LiDAR_File:' in line:
                        lidar_name = line.split(':')[1].strip().replace('.laz', '.tif')
                        lidar_path = self.lidar_base / lidar_name
                        break
        
        channels = {}
        
        # NDVI (reference - already aligned)
        channels['ndvi'] = self.read_sentinel_channel(ndvi_path, ref_profile)
        
        # NDWI
        if ndwi_path.exists():
            channels['ndwi'] = self.read_sentinel_channel(ndwi_path, ref_profile)
        else:
            shape = (ref_profile['height'], ref_profile['width'])
            channels['ndwi'] = np.full(shape, np.nan, dtype=np.float32)
        
        # HydroSHEDS
        if flow_acc_path.exists():
            flow_acc_raw = self.reproject_to_reference(flow_acc_path, ref_profile)
            channels['flow_acc'] = self.log_flow_acc(flow_acc_raw)
        else:
            shape = (ref_profile['height'], ref_profile['width'])
            channels['flow_acc'] = np.full(shape, np.nan, dtype=np.float32)
        
        if flow_dir_path.exists():
            flow_dir_raw = self.reproject_to_reference(flow_dir_path, ref_profile)
            channels['flow_dir'] = self.encode_flow_dir_sin(flow_dir_raw)
        else:
            shape = (ref_profile['height'], ref_profile['width'])
            channels['flow_dir'] = np.full(shape, np.nan, dtype=np.float32)
        
        # LiDAR DTM
        dtm = self.load_lidar_dtm(lidar_path, ref_profile)
        
        if dtm is not None:
            # Valid DTM - compute derivatives
            channels['dtm'] = dtm
            channels['slope'] = self.compute_slope(dtm)
            channels['roughness'] = self.compute_roughness(dtm)
        else:
            # Invalid DTM - fill with NaN
            shape = (ref_profile['height'], ref_profile['width'])
            channels['dtm'] = np.full(shape, np.nan, dtype=np.float32)
            channels['slope'] = np.full(shape, np.nan, dtype=np.float32)
            channels['roughness'] = np.full(shape, np.nan, dtype=np.float32)
        
        return channels
    
    # ==========================================================================
    # NORMALIZATION
    # ==========================================================================
    
    def compute_global_stats(self, aoi_indices):
        """OPTIMIZED: Parallel computation of global statistics"""
        print(f"\n{'='*60}")
        print("PASS 1: COMPUTING GLOBAL STATISTICS (PARALLEL)")
        print(f"{'='*60}")
        
        channel_values = {ch: [] for ch in self.channel_names}
        valid_aois = []
        
        # OPTIMIZATION: Parallel processing with ThreadPoolExecutor (I/O bound)
        def process_aoi_stats(aoi_idx):
            aoi = f"AOI_{aoi_idx:04d}"
            
            try:
                # Get reference profile from Sentinel NDVI
                ndvi_path = self.sentinel_base / aoi / 'sentinel' / 'ndvi.tif'
                if not ndvi_path.exists():
                    return None
                
                ref_profile = self.load_reference_profile(ndvi_path)
                
                # Load all channels
                channels = self.load_all_channels(aoi, ref_profile)
                if channels is None:
                    return None
                
                # Sample values
                aoi_samples = {}
                for ch_name in self.channel_names:
                    valid_data = channels[ch_name][~np.isnan(channels[ch_name])]
                    if len(valid_data) > 0:
                        sample_size = min(len(valid_data), 1000)
                        sampled = np.random.choice(valid_data, sample_size, replace=False)
                        aoi_samples[ch_name] = sampled.tolist()
                    else:
                        aoi_samples[ch_name] = []
                
                return (aoi_idx, aoi_samples)
            
            except Exception as e:
                print(f"      ⚠️  {aoi}: {e}")
                return None
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(tqdm(
                executor.map(process_aoi_stats, aoi_indices),
                total=len(aoi_indices),
                desc="Collecting stats"
            ))
        
        # Aggregate results
        for result in results:
            if result is not None:
                aoi_idx, aoi_samples = result
                valid_aois.append(aoi_idx)
                for ch_name in self.channel_names:
                    channel_values[ch_name].extend(aoi_samples.get(ch_name, []))
        
        # Compute statistics
        global_stats = {}
        
        print(f"\n📊 Global Statistics (from {len(valid_aois)} AOIs):")
        for ch_name in self.channel_names:
            values = np.array(channel_values[ch_name])
            
            if len(values) == 0:
                print(f"   ⚠️  {ch_name}: No data")
                global_stats[ch_name] = {'mean': 0, 'std': 1, 'p_low': 0, 'p_high': 1}
                continue
            
            p_low, p_high = np.percentile(values, self.clip_percentile)
            clipped = np.clip(values, p_low, p_high)
            
            mean = float(np.mean(clipped))
            std = float(np.std(clipped))
            
            if std == 0:
                std = 1.0
            
            global_stats[ch_name] = {
                'mean': mean,
                'std': std,
                'p_low': float(p_low),
                'p_high': float(p_high)
            }
            
            print(f"   {ch_name:12s}: mean={mean:7.3f}, std={std:6.3f}, range=[{p_low:7.3f}, {p_high:7.3f}]")
        
        # Save
        stats_file = self.output_dir / 'global_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(global_stats, f, indent=2)
        
        print(f"\n✅ Saved: {stats_file}")
        
        self.global_stats = global_stats
        return global_stats
    
    def normalize_channel(self, data, ch_name):
        """Apply global normalization (vectorized)"""
        if self.global_stats is None:
            raise ValueError("Global statistics not computed. Run compute_global_stats() first.")
        
        stats = self.global_stats[ch_name]
        
        # Vectorized operations
        clipped = np.clip(data, stats['p_low'], stats['p_high'])
        normalized = (clipped - stats['mean']) / stats['std']
        normalized[np.isnan(data)] = np.nan
        
        return normalized.astype(np.float32)
    
    # ==========================================================================
    # PATCH EXTRACTION (optimized)
    # ==========================================================================
    
    def is_valid_patch(self, patch):
        """Check if patch has enough valid pixels (vectorized)"""
        valid_ratio = np.sum(~np.isnan(patch)) / patch.size
        return valid_ratio >= self.min_valid_ratio
    
    def extract_patches_vectorized(self, channels, height, width):
        """
        OPTIMIZED: Vectorized patch extraction
        
        Extracts all patches in one go using advanced indexing
        """
        n_rows = (height - self.patch_size) // self.stride + 1
        n_cols = (width - self.patch_size) // self.stride + 1
        
        all_patches = []
        all_metadata = []
        
        # Pre-allocate indices for faster extraction
        row_indices = []
        col_indices = []
        
        for i in range(n_rows):
            for j in range(n_cols):
                r = i * self.stride
                c = j * self.stride
                
                if r + self.patch_size > height or c + self.patch_size > width:
                    continue
                
                row_indices.append(r)
                col_indices.append(c)
        
        # Extract all patches at once (vectorized)
        for r, c in zip(row_indices, col_indices):
            # Extract patch for each channel
            patch_channels = []
            for ch_name in self.channel_names:
                patch = channels[ch_name][r:r+self.patch_size, c:c+self.patch_size]
                patch_channels.append(patch)
            
            # Quality check (use NDVI - channel index 3)
            if not self.is_valid_patch(patch_channels[3]):
                continue
            
            # Stack all channels
            patch_tensor = np.stack(patch_channels, axis=0)
            
            # Normalize all channels at once
            for ch_idx, ch_name in enumerate(self.channel_names):
                patch_tensor[ch_idx] = self.normalize_channel(
                    patch_tensor[ch_idx], ch_name
                )
            
            all_patches.append(patch_tensor)
            all_metadata.append({'row': r, 'col': c})
        
        return all_patches, all_metadata
    
    def extract_patches(self, aoi_idx):
        """Extract exactly target_patches from an AOI"""
        
        aoi = f"AOI_{aoi_idx:04d}"
        
        try:
            # Get reference from Sentinel NDVI
            ndvi_path = self.sentinel_base / aoi / 'sentinel' / 'ndvi.tif'
            if not ndvi_path.exists():
                return None
            
            ref_profile = self.load_reference_profile(ndvi_path)
            height, width = ref_profile['height'], ref_profile['width']
            
            # Load all channels
            channels = self.load_all_channels(aoi, ref_profile)
            
            if channels is None:
                return None
            
            # Check DTM validity
            dtm_valid = not np.all(np.isnan(channels['dtm']))
            
            # Extract all valid patches (vectorized)
            all_patches, all_metadata = self.extract_patches_vectorized(
                channels, height, width
            )
            
            if len(all_patches) == 0:
                return None
            
            # Sample exactly target_patches (or keep all if fewer)
            if len(all_patches) > self.target_patches:
                indices = np.random.choice(len(all_patches), self.target_patches, replace=False)
                patches = np.array([all_patches[i] for i in indices])
                metadata = [all_metadata[i] for i in indices]
            else:
                patches = np.array(all_patches)
                metadata = all_metadata
            
            # Add patch IDs
            for idx, meta in enumerate(metadata):
                meta['patch_id'] = f"{aoi}_p{idx:03d}"
                meta['aoi'] = aoi
            
            # Save
            output_file = self.output_dir / f"{aoi}_patches.npz"
            
            np.savez_compressed(
                output_file,
                patches=patches,
                metadata=metadata,
                channel_names=self.channel_names,
                dtm_valid=dtm_valid
            )
            
            return patches
            
        except Exception as e:
            print(f"   ❌ {aoi}: {e}")
            return None
    
    # ==========================================================================
    # BATCH PROCESSING (parallel)
    # ==========================================================================
    
    def discover_aois(self):
        """Find all available AOIs"""
        aoi_indices = []
        for aoi_dir in sorted(self.sentinel_base.iterdir()):
            if aoi_dir.is_dir() and aoi_dir.name.startswith('AOI_'):
                ndvi = aoi_dir / 'sentinel' / 'ndvi.tif'
                if ndvi.exists():
                    try:
                        idx = int(aoi_dir.name.split('_')[1])
                        aoi_indices.append(idx)
                    except:
                        pass
        return aoi_indices
    
    def process_all(self):
        """
        OPTIMIZED: Main processing pipeline with parallel extraction
        
        Returns
        -------
        successful : list
            List of successfully processed AOI indices
        failed : list
            List of failed AOI indices
        """
        
        print(f"\n{'='*60}")
        print("SONAR 2.0 - DATASET PREPARATION (OPTIMIZED)")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Reference grid: Sentinel-2 NDVI")
        print(f"  Patch size: {self.patch_size}×{self.patch_size}")
        print(f"  Stride: {self.stride}")
        print(f"  Target patches/AOI: {self.target_patches}")
        print(f"  Min valid ratio: {self.min_valid_ratio}")
        print(f"  Channels: {len(self.channel_names)}")
        print(f"  Parallel workers: {self.n_workers}")
        
        aoi_indices = self.discover_aois()
        print(f"\n✅ Found {len(aoi_indices)} AOIs")
        
        # PASS 1: Global statistics (parallel)
        self.compute_global_stats(aoi_indices)
        
        # PASS 2: Extract patches (parallel)
        print(f"\n{'='*60}")
        print("PASS 2: EXTRACTING PATCHES (PARALLEL)")
        print(f"{'='*60}")
        
        successful = []
        failed = []
        failed_details = []
        
        # OPTIMIZATION: Use ThreadPoolExecutor for I/O-bound operations
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(tqdm(
                executor.map(self.extract_patches, aoi_indices),
                total=len(aoi_indices),
                desc="Processing"
            ))
        
        # Collect results
        for aoi_idx, result in zip(aoi_indices, results):
            if result is not None:
                successful.append(aoi_idx)
            else:
                failed.append(aoi_idx)
                failed_details.append({
                    'aoi': f"AOI_{aoi_idx:04d}",
                    'reason': 'Failed extraction (see logs)'
                })
        
        # Summary
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Success: {len(successful)}/{len(aoi_indices)}")
        print(f"❌ Failed: {len(failed)}/{len(aoi_indices)}")
        
        if successful:
            total_patches = len(successful) * self.target_patches
            print(f"\n📦 Total patches: ~{total_patches:,}")
            print(f"📁 Output: {self.output_dir}")
        
        if failed:
            print(f"\n⚠️  Failed AOIs: {failed[:10]}{'...' if len(failed) > 10 else ''}")
        
        # Save summary
        summary = {
            'config': {
                'patch_size': self.patch_size,
                'stride': self.stride,
                'target_patches': self.target_patches,
                'min_valid_ratio': self.min_valid_ratio,
                'reference_grid': 'Sentinel-2 NDVI',
                'n_workers': self.n_workers
            },
            'successful': successful,
            'failed': failed,
            'failed_details': failed_details,
            'success_rate': len(successful) / len(aoi_indices) * 100 if aoi_indices else 0
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📄 Summary: {self.output_dir / 'summary.json'}")
        
        return successful, failed
    
    def load_dataset(self, aoi_indices=None):
        """
        OPTIMIZED: Load patches for training with memory-efficient concatenation
        
        Parameters
        ----------
        aoi_indices : list, optional
            List of AOI indices to load. If None, loads all available.
        
        Returns
        -------
        dataset : ndarray
            Array of shape (N, 7, 64, 64) containing all patches
        metadata : list
            List of metadata dictionaries for each patch
        """
        
        if aoi_indices is None:
            files = sorted(self.output_dir.glob("AOI_*_patches.npz"))
        else:
            files = [self.output_dir / f"AOI_{i:04d}_patches.npz" for i in aoi_indices]
            files = [f for f in files if f.exists()]
        
        print(f"Loading {len(files)} AOIs...")
        
        # OPTIMIZATION: Pre-allocate if possible
        all_patches = []
        all_metadata = []
        
        for file in tqdm(files, desc="Loading"):
            with np.load(file, allow_pickle=True) as f:
                all_patches.append(f['patches'])
                all_metadata.extend(f['metadata'])
        
        # OPTIMIZATION: Efficient concatenation
        dataset = np.concatenate(all_patches, axis=0)
        
        print(f"\n✅ Dataset:")
        print(f"   Shape: {dataset.shape}")
        print(f"   Format: (N, 7, 64, 64)")
        print(f"   Memory: {dataset.nbytes / 1024 / 1024:.1f} MB")
        
        return dataset, all_metadata

def prepare_patches_from_upload(hydro_dir, sentinel_dir, lidar_dir, output_dir, config):
    """
    Prepare patches from uploaded user data for API inference.
    
    Parameters
    ----------
    hydro_dir, sentinel_dir, lidar_dir : str or Path
        Paths to uploaded data directories
    output_dir : str or Path
        Directory to save processed patches
    config : dict
        Configuration dictionary with keys:
        - patch_size, stride, target_patches, min_valid_ratio
    
    Returns
    -------
    patches : ndarray
        Processed patches (N, 7, 64, 64)
    metadata : list
        Patch metadata
    """
    data_prep = SonarDataPreparation(
        hydro_base=hydro_dir,
        sentinel_base=sentinel_dir,
        lidar_base=lidar_dir,
        output_dir=output_dir,
        patch_size=config.get('patch_size', 64),
        stride=config.get('stride', 32),
        target_patches=config.get('target_patches', 50),
        min_valid_ratio=config.get('min_valid_ratio', 0.7),
        n_workers=config.get('n_workers', 4)  # Lower for API context
    )
    
    successful, failed = data_prep.process_all()
    
    if not successful:
        raise ValueError("No valid patches extracted from uploaded data")
    
    patches, metadata = data_prep.load_dataset()
    return patches, metadata

def parse_args():
    parser = argparse.ArgumentParser("SONAR Data Prep (Optimized)")

    parser.add_argument("--hydro_base", type=Path, required=True)
    parser.add_argument("--sentinel_base", type=Path, required=True)
    parser.add_argument("--lidar_base", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--target_patches", type=int, default=50)
    parser.add_argument("--min_valid_ratio", type=float, default=0.7)
    parser.add_argument("--n_workers", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_prep = SonarDataPreparation(
        hydro_base=args.hydro_base,
        sentinel_base=args.sentinel_base,
        lidar_base=args.lidar_base,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        target_patches=args.target_patches,
        min_valid_ratio=args.min_valid_ratio,
        n_workers=args.n_workers
    )

    successful, failed = data_prep.process_all()

    if successful:
        patches, metadata = data_prep.load_dataset()
        print(f"\n🎯 Dataset ready! Shape: {patches.shape}")