"""
SONAR 2.0 - Data Validation Schemas
Comprehensive schemas for preprocessing, prediction, and API validation
"""

from typing import List, Dict, Optional, Tuple, Union
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
import numpy as np


# ==============================================================================
# CONFIGURATION SCHEMAS
# ==============================================================================

class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing pipeline"""
    
    patch_size: int = Field(
        default=64, 
        ge=32, 
        le=512,
        description="Size of square patches (must be power of 2)"
    )
    stride: int = Field(
        default=32, 
        ge=8, 
        le=256,
        description="Stride for patch extraction"
    )
    target_patches: int = Field(
        default=50, 
        ge=1, 
        le=1000,
        description="Target number of patches per AOI"
    )
    min_valid_ratio: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        description="Minimum ratio of valid pixels per patch"
    )
    clip_percentile: Tuple[float, float] = Field(
        default=(0.5, 99.5),
        description="Percentile range for clipping before normalization"
    )
    
    @validator('patch_size')
    def validate_patch_size(cls, v):
        """Ensure patch_size is power of 2"""
        if v & (v - 1) != 0:
            raise ValueError(f"patch_size must be power of 2, got {v}")
        return v
    
    @validator('stride')
    def validate_stride(cls, v, values):
        """Ensure stride is <= patch_size"""
        if 'patch_size' in values and v > values['patch_size']:
            raise ValueError(f"stride ({v}) cannot be larger than patch_size ({values['patch_size']})")
        return v
    
    @validator('clip_percentile')
    def validate_percentile(cls, v):
        """Ensure valid percentile range"""
        if not (0 <= v[0] < v[1] <= 100):
            raise ValueError(f"Invalid percentile range: {v}. Must be 0 <= low < high <= 100")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "patch_size": 64,
                "stride": 32,
                "target_patches": 50,
                "min_valid_ratio": 0.7,
                "clip_percentile": [0.5, 99.5]
            }
        }


class ModelConfig(BaseModel):
    """Configuration for model inference"""
    
    autoencoder_path: str = Field(..., description="Path to autoencoder checkpoint (.pt)")
    iforest_path: str = Field(..., description="Path to Isolation Forest model (.pkl)")
    kmeans_path: str = Field(..., description="Path to K-Means model (.pkl)")
    gate_path: str = Field(..., description="Path to GATE meta-learner (.pkl)")
    scaler_path: str = Field(..., description="Path to feature scaler (.pkl)")
    reference_embeddings_path: Optional[str] = Field(
        None, 
        description="Path to reference archaeological embeddings (.npy)"
    )
    device: str = Field(
        default="cpu", 
        description="Device for inference: 'cuda' or 'cpu'"
    )
    batch_size: int = Field(
        default=32, 
        ge=1, 
        le=512,
        description="Batch size for inference"
    )
    
    @validator('device')
    def validate_device(cls, v):
        """Validate device type"""
        if v not in ['cuda', 'cpu', 'mps']:
            raise ValueError(f"device must be 'cuda', 'cpu', or 'mps', got '{v}'")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "autoencoder_path": "checkpoints/autoencoder.pt",
                "iforest_path": "checkpoints/iforest.pkl",
                "kmeans_path": "checkpoints/kmeans.pkl",
                "gate_path": "checkpoints/gate.pkl",
                "scaler_path": "checkpoints/scaler.pkl",
                "device": "cuda",
                "batch_size": 32
            }
        }


# ==============================================================================
# DATA SCHEMAS
# ==============================================================================

class ChannelNames(str, Enum):
    """Valid channel names for SONAR 2.0"""
    DTM = "dtm"
    SLOPE = "slope"
    ROUGHNESS = "roughness"
    NDVI = "ndvi"
    NDWI = "ndwi"
    FLOW_ACC = "flow_acc"
    FLOW_DIR = "flow_dir"


class PatchMetadata(BaseModel):
    """Metadata for a single patch"""
    
    patch_id: str = Field(..., description="Unique patch identifier (e.g., 'AOI_0001_p042')")
    aoi: str = Field(..., description="AOI identifier (e.g., 'AOI_0001')")
    row: int = Field(..., ge=0, description="Row position in original raster")
    col: int = Field(..., ge=0, description="Column position in original raster")
    
    class Config:
        schema_extra = {
            "example": {
                "patch_id": "AOI_0001_p042",
                "aoi": "AOI_0001",
                "row": 128,
                "col": 256
            }
        }


class PatchData(BaseModel):
    """Single patch with metadata"""
    
    data: List[List[List[float]]] = Field(
        ..., 
        description="Patch data as nested list: [7, 64, 64]"
    )
    metadata: PatchMetadata
    channel_names: List[ChannelNames] = Field(
        default=[
            ChannelNames.DTM, 
            ChannelNames.SLOPE, 
            ChannelNames.ROUGHNESS,
            ChannelNames.NDVI, 
            ChannelNames.NDWI, 
            ChannelNames.FLOW_ACC, 
            ChannelNames.FLOW_DIR
        ]
    )
    
    @validator('data')
    def validate_shape(cls, v):
        """Validate patch dimensions"""
        if len(v) != 7:
            raise ValueError(f"Expected 7 channels, got {len(v)}")
        
        height = len(v[0])
        width = len(v[0][0]) if height > 0 else 0
        
        for i, channel in enumerate(v):
            if len(channel) != height:
                raise ValueError(f"Channel {i} has inconsistent height")
            for row in channel:
                if len(row) != width:
                    raise ValueError(f"Channel {i} has inconsistent width")
        
        return v


class GlobalStatistics(BaseModel):
    """Global normalization statistics for a channel"""
    
    mean: float = Field(..., description="Global mean")
    std: float = Field(..., description="Global standard deviation")
    p_low: float = Field(..., description="Lower percentile for clipping")
    p_high: float = Field(..., description="Upper percentile for clipping")
    
    @validator('std')
    def validate_std(cls, v):
        """Ensure std is positive"""
        if v <= 0:
            raise ValueError(f"std must be positive, got {v}")
        return v


class DatasetStatistics(BaseModel):
    """Complete dataset statistics for all channels"""
    
    dtm: GlobalStatistics
    slope: GlobalStatistics
    roughness: GlobalStatistics
    ndvi: GlobalStatistics
    ndwi: GlobalStatistics
    flow_acc: GlobalStatistics
    flow_dir: GlobalStatistics
    
    class Config:
        schema_extra = {
            "example": {
                "dtm": {"mean": 125.5, "std": 45.2, "p_low": 10.0, "p_high": 250.0},
                "slope": {"mean": 15.3, "std": 8.7, "p_low": 0.0, "p_high": 45.0},
                "roughness": {"mean": 2.1, "std": 1.3, "p_low": 0.0, "p_high": 10.0},
                "ndvi": {"mean": 0.35, "std": 0.25, "p_low": -0.2, "p_high": 0.9},
                "ndwi": {"mean": 0.15, "std": 0.18, "p_low": -0.3, "p_high": 0.8},
                "flow_acc": {"mean": 5.2, "std": 2.1, "p_low": 0.0, "p_high": 12.0},
                "flow_dir": {"mean": 0.0, "std": 0.7, "p_low": -1.0, "p_high": 1.0}
            }
        }


# ==============================================================================
# PREDICTION REQUEST/RESPONSE SCHEMAS
# ==============================================================================

class PredictionRequest(BaseModel):
    """Request for model prediction"""
    
    patches: List[PatchData] = Field(
        ..., 
        min_items=1,
        description="List of patches to process"
    )
    threshold: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0,
        description="Detection threshold for anomalies"
    )
    batch_size: Optional[int] = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for processing"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "patches": [
                    {
                        "data": [[[0.0] * 64] * 64] * 7,
                        "metadata": {
                            "patch_id": "AOI_0001_p001",
                            "aoi": "AOI_0001",
                            "row": 0,
                            "col": 0
                        }
                    }
                ],
                "threshold": 0.5,
                "batch_size": 32
            }
        }


class ModelScores(BaseModel):
    """Individual model prediction scores"""
    
    autoencoder: float = Field(..., ge=0.0, le=1.0, description="Autoencoder anomaly score")
    iforest: float = Field(..., ge=0.0, le=1.0, description="Isolation Forest score")
    kmeans: float = Field(..., ge=0.0, le=1.0, description="K-Means similarity score")
    arch_similarity: float = Field(..., ge=0.0, le=1.0, description="Archaeological similarity score")
    
    class Config:
        schema_extra = {
            "example": {
                "autoencoder": 0.75,
                "iforest": 0.68,
                "kmeans": 0.82,
                "arch_similarity": 0.45
            }
        }


class SinglePrediction(BaseModel):
    """Prediction result for a single patch"""
    
    patch_id: str = Field(..., description="Patch identifier")
    aoi: str = Field(..., description="AOI identifier")
    row: int = Field(..., ge=0, description="Row position")
    col: int = Field(..., ge=0, description="Column position")
    gate_prediction: float = Field(..., ge=0.0, le=1.0, description="Final GATE ensemble score")
    is_anomaly: bool = Field(..., description="Binary anomaly classification")
    model_scores: ModelScores = Field(..., description="Individual model scores")
    
    class Config:
        schema_extra = {
            "example": {
                "patch_id": "AOI_0001_p042",
                "aoi": "AOI_0001",
                "row": 128,
                "col": 256,
                "gate_prediction": 0.85,
                "is_anomaly": True,
                "model_scores": {
                    "autoencoder": 0.75,
                    "iforest": 0.68,
                    "kmeans": 0.82,
                    "arch_similarity": 0.45
                }
            }
        }


class TopCandidate(BaseModel):
    """Top anomaly candidate summary"""
    
    patch_id: str
    row: int
    col: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_anomaly: bool


class PredictionSummary(BaseModel):
    """Summary statistics for AOI predictions"""
    
    total_patches: int = Field(..., ge=0, description="Total patches processed")
    anomalies_detected: int = Field(..., ge=0, description="Number of anomalies found")
    anomaly_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of anomalous patches")
    mean_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean prediction confidence")
    max_confidence: float = Field(..., ge=0.0, le=1.0, description="Maximum confidence score")
    min_confidence: float = Field(..., ge=0.0, le=1.0, description="Minimum confidence score")
    top_candidates: List[TopCandidate] = Field(..., max_items=20, description="Top 20 anomaly candidates")
    
    @validator('anomalies_detected')
    def validate_anomaly_count(cls, v, values):
        """Ensure anomaly count doesn't exceed total patches"""
        if 'total_patches' in values and v > values['total_patches']:
            raise ValueError(f"anomalies_detected ({v}) cannot exceed total_patches ({values['total_patches']})")
        return v


class PredictionResponse(BaseModel):
    """Complete prediction response"""
    
    predictions: List[SinglePrediction] = Field(..., description="Per-patch predictions")
    summary: PredictionSummary = Field(..., description="Aggregate statistics")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "patch_id": "AOI_0001_p042",
                        "aoi": "AOI_0001",
                        "row": 128,
                        "col": 256,
                        "gate_prediction": 0.85,
                        "is_anomaly": True,
                        "model_scores": {
                            "autoencoder": 0.75,
                            "iforest": 0.68,
                            "kmeans": 0.82,
                            "arch_similarity": 0.45
                        }
                    }
                ],
                "summary": {
                    "total_patches": 50,
                    "anomalies_detected": 8,
                    "anomaly_percentage": 16.0,
                    "mean_confidence": 0.42,
                    "max_confidence": 0.95,
                    "min_confidence": 0.05,
                    "top_candidates": [
                        {
                            "patch_id": "AOI_0001_p042",
                            "row": 128,
                            "col": 256,
                            "confidence": 0.95,
                            "is_anomaly": True
                        }
                    ]
                },
                "processing_time": 2.34
            }
        }


# ==============================================================================
# UPLOAD/BATCH PROCESSING SCHEMAS
# ==============================================================================

class DataUploadPaths(BaseModel):
    """Paths to uploaded geospatial data"""
    
    hydro_dir: str = Field(..., description="Path to HydroSHEDS data directory")
    sentinel_dir: str = Field(..., description="Path to Sentinel-2 data directory")
    lidar_dir: str = Field(..., description="Path to LiDAR DTM tiles directory")
    output_dir: str = Field(..., description="Path for processed output")
    
    class Config:
        schema_extra = {
            "example": {
                "hydro_dir": "/mnt/user-data/uploads/hydro",
                "sentinel_dir": "/mnt/user-data/uploads/sentinel",
                "lidar_dir": "/mnt/user-data/uploads/lidar",
                "output_dir": "/mnt/user-data/outputs/processed"
            }
        }


class BatchProcessingRequest(BaseModel):
    """Request for batch preprocessing and prediction"""
    
    data_paths: DataUploadPaths
    preprocess_config: PreprocessingConfig
    prediction_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "data_paths": {
                    "hydro_dir": "/uploads/hydro",
                    "sentinel_dir": "/uploads/sentinel",
                    "lidar_dir": "/uploads/lidar",
                    "output_dir": "/outputs"
                },
                "preprocess_config": {
                    "patch_size": 64,
                    "stride": 32,
                    "target_patches": 50,
                    "min_valid_ratio": 0.7
                },
                "prediction_threshold": 0.5
            }
        }


class ProcessingSummary(BaseModel):
    """Summary of preprocessing results"""
    
    successful_aois: List[int] = Field(..., description="Successfully processed AOI indices")
    failed_aois: List[int] = Field(..., description="Failed AOI indices")
    total_patches_extracted: int = Field(..., ge=0, description="Total patches extracted")
    success_rate: float = Field(..., ge=0.0, le=100.0, description="Success rate percentage")
    
    class Config:
        schema_extra = {
            "example": {
                "successful_aois": [1, 2, 3, 5, 8],
                "failed_aois": [4, 6, 7],
                "total_patches_extracted": 250,
                "success_rate": 62.5
            }
        }


class BatchProcessingResponse(BaseModel):
    """Response for batch processing"""
    
    processing_summary: ProcessingSummary
    prediction_results: PredictionResponse
    global_statistics: DatasetStatistics
    
    class Config:
        schema_extra = {
            "example": {
                "processing_summary": {
                    "successful_aois": [1, 2, 3],
                    "failed_aois": [],
                    "total_patches_extracted": 150,
                    "success_rate": 100.0
                },
                "prediction_results": {
                    "predictions": [],
                    "summary": {
                        "total_patches": 150,
                        "anomalies_detected": 12,
                        "anomaly_percentage": 8.0,
                        "mean_confidence": 0.35,
                        "max_confidence": 0.92,
                        "min_confidence": 0.08,
                        "top_candidates": []
                    }
                },
                "global_statistics": {}
            }
        }


# ==============================================================================
# ERROR SCHEMAS
# ==============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information"""
    
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: Optional[Dict] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error_type": "ValidationError",
                "message": "Invalid patch dimensions",
                "details": {
                    "expected_shape": [7, 64, 64],
                    "received_shape": [7, 32, 32]
                }
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response"""
    
    success: bool = Field(default=False)
    error: ErrorDetail
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": {
                    "error_type": "ValidationError",
                    "message": "Invalid input data",
                    "details": {}
                }
            }
        }


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def validate_patch_array(patch: np.ndarray) -> bool:
    """
    Validate numpy array patch shape and data
    
    Args:
        patch: numpy array to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if patch.ndim != 3:
        raise ValueError(f"Patch must be 3D array, got {patch.ndim}D")
    
    if patch.shape[0] != 7:
        raise ValueError(f"Patch must have 7 channels, got {patch.shape[0]}")
    
    if patch.shape[1] != patch.shape[2]:
        raise ValueError(f"Patch must be square, got {patch.shape[1]}x{patch.shape[2]}")
    
    if not np.issubdtype(patch.dtype, np.floating):
        raise ValueError(f"Patch must be float type, got {patch.dtype}")
    
    return True


def validate_preprocessing_config(config: Dict) -> PreprocessingConfig:
    """
    Validate preprocessing configuration dictionary
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated PreprocessingConfig object
    """
    return PreprocessingConfig(**config)


def validate_model_config(config: Dict) -> ModelConfig:
    """
    Validate model configuration dictionary
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated ModelConfig object
    """
    return ModelConfig(**config)


# ==============================================================================
# EXPORT ALL SCHEMAS
# ==============================================================================

__all__ = [
    # Configuration
    'PreprocessingConfig',
    'ModelConfig',
    
    # Data
    'ChannelNames',
    'PatchMetadata',
    'PatchData',
    'GlobalStatistics',
    'DatasetStatistics',
    
    # Prediction
    'PredictionRequest',
    'ModelScores',
    'SinglePrediction',
    'TopCandidate',
    'PredictionSummary',
    'PredictionResponse',
    
    # Batch Processing
    'DataUploadPaths',
    'BatchProcessingRequest',
    'ProcessingSummary',
    'BatchProcessingResponse',
    
    # Errors
    'ErrorDetail',
    'ErrorResponse',
    
    # Helpers
    'validate_patch_array',
    'validate_preprocessing_config',
    'validate_model_config'
]