"""
SONAR 2.0 - Demo Data Generator
Processes all AOIs and creates demo data for the website
Run this script once to generate demo.json
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    PATCHES_DIR = Path(r'D:\All_Coding_stuff\SONAR 2.0\Project\Test_data\patches_final_file')
    PROB_MATRICES_DIR = Path(r'D:\All_Coding_stuff\SONAR 2.0\Project\Test_data\test_unified_probablity_matrices_with_gate')
    DATASET_DIR = Path(r'D:\All_Coding_stuff\SONAR 2.0\Project\Test_data\Test Dataset')
    OUTPUT_FILE = Path(r'D:\All_Coding_stuff\SONAR 2.0\Project\Test_data\demo_data.json')
    # Detection threshold
    THRESHOLD = 0.5
    
    # AOIs to process (all available)
    AOI_IDS = ['AOI_0000', 'AOI_0001', 'AOI_0003', 'AOI_0004', 
               'AOI_0006', 'AOI_0007', 'AOI_0008']

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_aoi_metadata(aoi_id: str, dataset_dir: Path) -> Dict:
    """Load geographic metadata from metadata.json"""
    meta_file = dataset_dir / aoi_id / 'meta' / 'metadata.json'
    
    if not meta_file.exists():
        print(f"⚠️  Metadata not found for {aoi_id}, using defaults")
        return {
            'aoi_id': aoi_id,
            'bounds': [0, 0, 0, 0],
            'min_lat': 0,
            'max_lat': 0,
            'min_lon': 0,
            'max_lon': 0
        }
    
    with open(meta_file, 'r') as f:
        return json.load(f)

def load_patches_and_probabilities(aoi_id: str, config: Config) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load patches and probability matrices for an AOI"""
    
    # Load patches
    patches_file = config.PATCHES_DIR / f"{aoi_id}_all_patches.npz"
    with np.load(patches_file, allow_pickle=True) as data:
        patches = data['patches']
        metadata = list(data['metadata'])
    
    # Load probability matrices
    prob_file = config.PROB_MATRICES_DIR / f"{aoi_id}_unified_prob_matrix.npz"
    with np.load(prob_file, allow_pickle=True) as data:
        prob_matrix = data['unified_matrix']  # Shape: (N, 64, 64, 5)
    
    return patches, prob_matrix, metadata

def pixel_to_latlon(row: int, col: int, aoi_metadata: Dict, raster_shape: Tuple[int, int]) -> Tuple[float, float]:
    """
    Convert pixel coordinates to lat/lon
    
    Args:
        row, col: Pixel coordinates
        aoi_metadata: AOI geographic bounds
        raster_shape: (height, width) of the raster
    
    Returns:
        (lat, lon) tuple
    """
    height, width = raster_shape
    
    # Get bounds
    min_lon = aoi_metadata['min_lon']
    max_lon = aoi_metadata['max_lon']
    min_lat = aoi_metadata['min_lat']
    max_lat = aoi_metadata['max_lat']
    
    # Convert pixel to geographic coordinates
    lon = min_lon + (col / width) * (max_lon - min_lon)
    lat = max_lat - (row / height) * (max_lat - min_lat)  # Flip Y axis
    
    return lat, lon

# ==============================================================================
# ANOMALY EXTRACTION
# ==============================================================================

def extract_anomalies_from_aoi(
    aoi_id: str,
    patches: np.ndarray,
    prob_matrix: np.ndarray,
    metadata: List[Dict],
    aoi_metadata: Dict,
    threshold: float = 0.5
) -> List[Dict]:
    """
    Extract anomalies from an AOI
    
    Returns list of anomaly dictionaries with coordinates
    """
    anomalies = []
    
    # Extract gate predictions (channel 4)
    gate_predictions = prob_matrix[:, :, :, 4]  # Shape: (N, 64, 64)
    
    # Get mean score per patch
    patch_scores = np.mean(gate_predictions, axis=(1, 2))  # Shape: (N,)
    
    # Estimate full raster dimensions from patches
    max_row = max(m['row'] + 64 for m in metadata)
    max_col = max(m['col'] + 64 for m in metadata)
    raster_shape = (max_row, max_col)
    
    # Find anomalies
    anomaly_indices = np.where(patch_scores >= threshold)[0]
    
    for idx in anomaly_indices:
        meta = metadata[idx]
        score = float(patch_scores[idx])
        
        # Get patch center coordinates
        patch_row = meta['row'] + 32  # Center of 64x64 patch
        patch_col = meta['col'] + 32
        
        # Convert to lat/lon
        lat, lon = pixel_to_latlon(patch_row, patch_col, aoi_metadata, raster_shape)
        
        anomaly = {
            'aoi_id': aoi_id,
            'patch_id': meta['patch_id'],
            'confidence': round(score, 4),
            'coordinates': {
                'lat': round(lat, 6),
                'lng': round(lon, 6)
            },
            'pixel_location': {
                'row': int(patch_row),
                'col': int(patch_col)
            },
            'patch_bounds': {
                'row_start': int(meta['row']),
                'row_end': int(meta['row'] + 64),
                'col_start': int(meta['col']),
                'col_end': int(meta['col'] + 64)
            }
        }
        
        anomalies.append(anomaly)
    
    return anomalies

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def generate_demo_data(config: Config):
    """
    Process all AOIs and generate demo data
    """
    print("\n" + "="*70)
    print("🗺️  SONAR 2.0 - Demo Data Generator")
    print("="*70)
    
    all_anomalies = []
    aoi_info = []
    total_patches = 0
    total_anomalies = 0
    
    for aoi_id in config.AOI_IDS:
        print(f"\n📍 Processing {aoi_id}...")
        
        try:
            # Load geographic metadata
            aoi_metadata = load_aoi_metadata(aoi_id, config.DATASET_DIR)
            
            # Load patches and probabilities
            patches, prob_matrix, metadata = load_patches_and_probabilities(aoi_id, config)
            
            print(f"   ✅ Loaded {len(patches)} patches")
            print(f"   ✅ Probability matrix shape: {prob_matrix.shape}")
            
            # Extract anomalies
            anomalies = extract_anomalies_from_aoi(
                aoi_id, patches, prob_matrix, metadata, 
                aoi_metadata, config.THRESHOLD
            )
            
            print(f"   🔴 Detected {len(anomalies)} anomalies (threshold: {config.THRESHOLD})")
            
            # Store AOI info
            aoi_info.append({
                'aoi_id': aoi_id,
                'total_patches': len(patches),
                'anomalies_detected': len(anomalies),
                'bounds': {
                    'min_lat': aoi_metadata['min_lat'],
                    'max_lat': aoi_metadata['max_lat'],
                    'min_lon': aoi_metadata['min_lon'],
                    'max_lon': aoi_metadata['max_lon']
                },
                'center': {
                    'lat': (aoi_metadata['min_lat'] + aoi_metadata['max_lat']) / 2,
                    'lng': (aoi_metadata['min_lon'] + aoi_metadata['max_lon']) / 2
                }
            })
            
            all_anomalies.extend(anomalies)
            total_patches += len(patches)
            total_anomalies += len(anomalies)
            
        except Exception as e:
            print(f"   ❌ Error processing {aoi_id}: {e}")
            continue
    
    # Create demo data structure
    demo_data = {
        'title': 'SONAR 2.0 - Multi-AOI Archaeological Analysis',
        'description': 'Comprehensive analysis across 7 Areas of Interest showing detected archaeological anomalies',
        'generated_at': '2024-02-11',
        'summary': {
            'total_aois': len(config.AOI_IDS),
            'total_patches': total_patches,
            'anomalies_detected': total_anomalies,
            'anomaly_percentage': round((total_anomalies / total_patches) * 100, 2) if total_patches > 0 else 0,
            'mean_confidence': round(np.mean([a['confidence'] for a in all_anomalies]), 4) if all_anomalies else 0
        },
        'aois': aoi_info,
        'top_candidates': sorted(all_anomalies, key=lambda x: x['confidence'], reverse=True)[:50],
        'all_anomalies': all_anomalies
    }
    
    # Save to JSON
    with open(config.OUTPUT_FILE, 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✅ Demo Data Generated Successfully!")
    print(f"{'='*70}")
    print(f"\n📊 Summary:")
    print(f"   Total AOIs processed: {len(config.AOI_IDS)}")
    print(f"   Total patches analyzed: {total_patches:,}")
    print(f"   Total anomalies detected: {total_anomalies:,}")
    print(f"   Anomaly rate: {demo_data['summary']['anomaly_percentage']:.2f}%")
    print(f"   Mean confidence: {demo_data['summary']['mean_confidence']:.4f}")
    print(f"\n💾 Saved to: {config.OUTPUT_FILE}")
    print(f"   File size: {config.OUTPUT_FILE.stat().st_size / 1024:.1f} KB\n")

# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    config = Config()
    
    # Verify directories exist
    if not config.PATCHES_DIR.exists():
        print(f"❌ Patches directory not found: {config.PATCHES_DIR}")
        exit(1)
    
    if not config.PROB_MATRICES_DIR.exists():
        print(f"❌ Probability matrices directory not found: {config.PROB_MATRICES_DIR}")
        exit(1)
    
    if not config.DATASET_DIR.exists():
        print(f"❌ Dataset directory not found: {config.DATASET_DIR}")
        exit(1)
    
    # Generate demo data
    generate_demo_data(config)