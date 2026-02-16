"""
SONAR 2.0 - Post-Processing Module
Transforms raw predictor output into frontend-ready formats
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
from matplotlib.patches import Rectangle
import io
import base64
from typing import Dict, List, Tuple, Optional
import json
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class PostProcessor:
    """
    Converts predictor raw outputs into presentation-ready formats:
    - GeoJSON for map markers
    - Heatmap images
    - Statistics summaries
    - Downloadable reports
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize post-processor
        
        Args:
            config: Optional configuration dict with:
                - heatmap_cmap: colormap for heatmaps (default: 'hot')
                - dpi: image resolution (default: 150)
                - reference_transform: rasterio transform for coordinate conversion
                - reference_crs: coordinate reference system
        """
        self.config = config or {}
        self.heatmap_cmap = self.config.get('heatmap_cmap', 'hot')
        self.dpi = self.config.get('dpi', 150)
        self.reference_transform = self.config.get('reference_transform')
        self.reference_crs = self.config.get('reference_crs')
    
    # ==========================================================================
    # COORDINATE CONVERSION
    # ==========================================================================
    
    def pixel_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to lat/lon
        
        Args:
            row, col: Pixel coordinates
        
        Returns:
            (lat, lon) tuple
        """
        if self.reference_transform is None:
            # Fallback: return normalized coordinates
            return float(row), float(col)
        
        try:
            import rasterio
            from rasterio.warp import transform as transform_coords
            
            # Convert pixel to projected coordinates
            x, y = rasterio.transform.xy(self.reference_transform, row, col)
            
            # Convert to WGS84 if needed
            if self.reference_crs and self.reference_crs != 'EPSG:4326':
                lon, lat = transform_coords(self.reference_crs, 'EPSG:4326', [x], [y])
                return lat[0], lon[0]
            else:
                return y, x  # y=lat, x=lon
        
        except Exception as e:
            print(f"Warning: Coordinate conversion failed: {e}")
            return float(row), float(col)
    
    # ==========================================================================
    # GEOJSON GENERATION
    # ==========================================================================
    
    def create_geojson(
        self, 
        predictions: List[Dict], 
        threshold: float = 0.5
    ) -> Dict:
        """
        Create GeoJSON FeatureCollection for map visualization
        
        Args:
            predictions: List of prediction dicts from predictor
            threshold: Only include predictions above this threshold
        
        Returns:
            GeoJSON dict ready for frontend mapping libraries
        """
        features = []
        
        for pred in predictions:
            if pred['gate_prediction'] < threshold:
                continue
            
            # Get center of patch
            row = pred.get('row', 0)
            col = pred.get('col', 0)
            center_row = row + 32  # 64x64 patch, center at 32
            center_col = col + 32
            
            lat, lon = self.pixel_to_latlon(center_row, center_col)
            
            # Create feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "patch_id": pred.get('patch_id', 'unknown'),
                    "confidence": round(pred['gate_prediction'], 4),
                    "is_anomaly": pred['is_anomaly'],
                    "row": int(row),
                    "col": int(col),
                    "model_scores": {
                        k: round(v, 4) 
                        for k, v in pred['model_scores'].items()
                    }
                }
            }
            
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    # ==========================================================================
    # HEATMAP GENERATION
    # ==========================================================================
    
    def create_prediction_heatmap(
        self,
        predictions: List[Dict],
        aoi_shape: Tuple[int, int],
        patch_size: int = 64
    ) -> str:
        """
        Create prediction heatmap as base64 image
        
        Args:
            predictions: List of prediction dicts
            aoi_shape: (height, width) of full AOI
            patch_size: Size of patches (default 64)
        
        Returns:
            Base64-encoded PNG image string
        """
        # Initialize empty heatmap
        heatmap = np.zeros(aoi_shape, dtype=np.float32)
        count_map = np.zeros(aoi_shape, dtype=np.int32)
        
        # Fill heatmap with predictions
        for pred in predictions:
            row = pred.get('row', 0)
            col = pred.get('col', 0)
            score = pred['gate_prediction']
            
            # Fill patch region
            r_end = min(row + patch_size, aoi_shape[0])
            c_end = min(col + patch_size, aoi_shape[1])
            
            heatmap[row:r_end, col:c_end] += score
            count_map[row:r_end, col:c_end] += 1
        
        # Average overlapping regions
        mask = count_map > 0
        heatmap[mask] /= count_map[mask]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        
        im = ax.imshow(
            heatmap, 
            cmap=self.heatmap_cmap, 
            vmin=0, 
            vmax=1,
            interpolation='bilinear',
            aspect='auto'
        )
        
        ax.set_title('GATE Prediction Heatmap', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Column (pixels)', fontsize=12)
        ax.set_ylabel('Row (pixels)', fontsize=12)
        ax.grid(False)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Anomaly Probability', fontsize=12, rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def create_top_candidates_grid(
        self,
        predictions: List[Dict],
        patches: np.ndarray,
        top_n: int = 16
    ) -> str:
        """
        Create grid visualization of top candidate patches
        
        Args:
            predictions: List of prediction dicts
            patches: Original patches array (N, 7, 64, 64)
            top_n: Number of top candidates to show
        
        Returns:
            Base64-encoded PNG image
        """
        # Sort by confidence
        sorted_preds = sorted(predictions, key=lambda x: x['gate_prediction'], reverse=True)
        top_preds = sorted_preds[:top_n]
        
        # Create grid
        grid_size = int(np.ceil(np.sqrt(top_n)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16), facecolor='white')
        axes = axes.flatten()
        
        for idx, pred in enumerate(top_preds):
            ax = axes[idx]
            
            # Find patch index in original array
            patch_id = pred.get('patch_id', '')
            # Assuming patch indices align with predictions order
            patch_idx = predictions.index(pred)
            
            # Get DTM channel (first channel)
            dtm = patches[patch_idx, 0, :, :]
            
            # Plot
            im = ax.imshow(dtm, cmap='terrain', interpolation='bilinear')
            
            # Add border based on confidence
            score = pred['gate_prediction']
            color = 'red' if score > 0.7 else 'orange' if score > 0.5 else 'yellow'
            rect = Rectangle((0, 0), 63, 63, linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Title
            ax.set_title(
                f"{patch_id}\nScore: {score:.3f}",
                fontsize=10,
                fontweight='bold'
            )
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(len(top_preds), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(
            f'Top {len(top_preds)} Archaeological Candidates', 
            fontsize=18, 
            fontweight='bold',
            y=0.995
        )
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    # ==========================================================================
    # STATISTICS FORMATTING
    # ==========================================================================
    
    def format_statistics(self, summary: Dict, threshold: float = 0.5) -> Dict:
        """
        Format summary statistics for frontend display
        
        Args:
            summary: Summary dict from predictor.predict_aoi()
            threshold: Detection threshold used
        
        Returns:
            Formatted statistics dict
        """
        return {
            "overview": {
                "total_patches": summary['total_patches'],
                "anomalies_detected": summary['anomalies_detected'],
                "anomaly_percentage": round(summary['anomaly_percentage'], 2),
                "threshold": threshold
            },
            "confidence_stats": {
                "mean": round(summary['mean_confidence'], 4),
                "max": round(summary['max_confidence'], 4),
                "min": round(summary['min_confidence'], 4)
            },
            "top_candidates": [
                {
                    "rank": idx + 1,
                    "patch_id": cand.get('patch_id', 'unknown'),
                    "position": {
                        "row": cand.get('row', -1), 
                        "col": cand.get('col', -1)
                    },
                    "confidence": round(cand.get('confidence', cand.get('gate_prediction', 0)), 4),
                    "is_anomaly": cand.get('is_anomaly', False)
                }
                for idx, cand in enumerate(summary['top_candidates'][:10])
            ]
        }
    
    def create_confidence_distribution(self, predictions: List[Dict]) -> str:
        """
        Create histogram of confidence scores
        
        Args:
            predictions: List of prediction dicts
        
        Returns:
            Base64-encoded PNG histogram
        """
        scores = [p['gate_prediction'] for p in predictions]
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        
        ax.hist(scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
        
        ax.set_xlabel('GATE Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Confidence Scores', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    # ==========================================================================
    # DOWNLOADABLE REPORTS
    # ==========================================================================
    
    def create_csv_report(self, predictions: List[Dict]) -> str:
        """
        Create CSV report of all predictions
        
        Args:
            predictions: List of prediction dicts
        
        Returns:
            CSV string
        """
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'Patch ID', 'Row', 'Col', 'GATE Confidence', 'Is Anomaly',
            'Autoencoder Score', 'IForest Score', 'KMeans Score', 'Arch Similarity'
        ])
        
        # Data rows
        for pred in predictions:
            writer.writerow([
                pred.get('patch_id', 'unknown'),
                pred.get('row', -1),
                pred.get('col', -1),
                round(pred['gate_prediction'], 4),
                pred['is_anomaly'],
                round(pred['model_scores']['autoencoder'], 4),
                round(pred['model_scores']['iforest'], 4),
                round(pred['model_scores']['kmeans'], 4),
                round(pred['model_scores']['arch_similarity'], 4)
            ])
        
        return output.getvalue()
    
    def create_json_report(self, result: Dict) -> str:
        """
        Create complete JSON report
        
        Args:
            result: Full result dict from predictor.predict_aoi()
        
        Returns:
            JSON string
        """
        # Deep copy to avoid modifying original
        report = {
            "metadata": {
                "aoi": result.get('metadata', [{}])[0].get('aoi', 'unknown'),
                "total_patches": result['summary']['total_patches'],
                "analysis_timestamp": None  # Add if needed
            },
            "summary": self.format_statistics(result['summary']),
            "predictions": [
                {
                    "patch_id": p.get('patch_id', 'unknown'),
                    "position": {"row": p.get('row', -1), "col": p.get('col', -1)},
                    "gate_confidence": round(p['gate_prediction'], 4),
                    "is_anomaly": p['is_anomaly'],
                    "model_breakdown": {
                        k: round(v, 4) for k, v in p['model_scores'].items()
                    }
                }
                for p in result['predictions']
            ]
        }
        
        return json.dumps(report, indent=2)
    
    # ==========================================================================
    # COMPLETE RESPONSE BUILDER
    # ==========================================================================
    
    def build_api_response(
        self,
        predictions: List[Dict],
        summary: Dict,
        metadata: List[Dict],
        patches: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        aoi_shape: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Build complete API response with all visualizations and data
        
        Args:
            predictions: Predictions from predictor
            summary: Summary stats from predictor
            metadata: Patch metadata
            patches: Original patches (optional, for visualizations)
            threshold: Detection threshold
            aoi_shape: Shape of full AOI (optional, for heatmap)
        
        Returns:
            Complete response dict ready for API/frontend
        """
        response = {
            "status": "success",
            "data": {
                "geojson": self.create_geojson(predictions, threshold),
                "statistics": self.format_statistics(summary, threshold),
                "visualizations": {}
            },
            "downloads": {}
        }
        
        # Add heatmap if AOI shape provided
        if aoi_shape:
            response["data"]["visualizations"]["heatmap"] = self.create_prediction_heatmap(
                predictions, aoi_shape
            )
        
        # Add top candidates visualization if patches provided
        if patches is not None:
            response["data"]["visualizations"]["top_candidates"] = self.create_top_candidates_grid(
                predictions, patches, top_n=16
            )
        
        # Add confidence distribution
        response["data"]["visualizations"]["confidence_distribution"] = \
            self.create_confidence_distribution(predictions)
        
        # Add downloadable reports
        response["downloads"]["csv"] = self.create_csv_report(predictions)
        response["downloads"]["json"] = self.create_json_report({
            'predictions': predictions,
            'summary': summary,
            'metadata': metadata
        })
        
        return response


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def postprocess_predictions(
    predictions: List[Dict],
    summary: Dict,
    metadata: List[Dict],
    patches: Optional[np.ndarray] = None,
    config: Optional[Dict] = None
) -> Dict:
    """
    One-shot postprocessing function
    
    Args:
        predictions: From predictor.predict_aoi()
        summary: From predictor.predict_aoi()
        metadata: Original patch metadata
        patches: Optional patches array for visualizations
        config: Optional postprocessor config
    
    Returns:
        Complete API response
    """
    processor = PostProcessor(config)
    
    # Determine AOI shape from metadata
    if metadata:
        max_row = max(m.get('row', 0) + 64 for m in metadata)
        max_col = max(m.get('col', 0) + 64 for m in metadata)
        aoi_shape = (max_row, max_col)
    else:
        aoi_shape = None
    
    return processor.build_api_response(
        predictions=predictions,
        summary=summary,
        metadata=metadata,
        patches=patches,
        aoi_shape=aoi_shape
    )


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    # Test with dummy data
    dummy_predictions = [
        {
            'patch_id': f'AOI_0001_p{i:03d}',
            'row': i * 32,
            'col': i * 32,
            'gate_prediction': np.random.rand(),
            'is_anomaly': np.random.rand() > 0.5,
            'model_scores': {
                'autoencoder': np.random.rand(),
                'iforest': np.random.rand(),
                'kmeans': np.random.rand(),
                'arch_similarity': np.random.rand()
            }
        }
        for i in range(50)
    ]
    
    dummy_summary = {
        'total_patches': 50,
        'anomalies_detected': 15,
        'anomaly_percentage': 30.0,
        'mean_confidence': 0.45,
        'max_confidence': 0.95,
        'min_confidence': 0.05,
        'top_candidates': dummy_predictions[:20]
    }
    
    processor = PostProcessor()
    
    print("Testing PostProcessor...")
    print("=" * 60)
    
    # Test GeoJSON
    geojson = processor.create_geojson(dummy_predictions, threshold=0.5)
    print(f"✅ GeoJSON created: {len(geojson['features'])} features")
    
    # Test statistics
    stats = processor.format_statistics(dummy_summary)
    print(f"✅ Statistics formatted: {stats['overview']}")
    
    # Test CSV
    csv_report = processor.create_csv_report(dummy_predictions)
    print(f"✅ CSV report created: {len(csv_report)} chars")
    
    print("=" * 60)
    print("All tests passed! ✨")