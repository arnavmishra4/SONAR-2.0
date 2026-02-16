"""
SONAR 2.0 - Core Prediction Engine (OPTIMIZED VERSION)
Pure inference logic - no UI, no API routes
Optimizations: batching, torch.compile, cached operations, vectorization
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import your model architectures
from model.ResUnet import ResUNetAutoencoder, ResUNetEncoder
from model.GATE import GateNetwork


class SONARPredictor:
    """
    Optimized inference class for SONAR 2.0
    Loads all models once, runs predictions efficiently with batching
    """
    
    def __init__(self, config: Dict):
        """
        Initialize predictor with all models
        
        Args:
            config: Dictionary with paths to model checkpoints
                   {
                       'autoencoder_path': 'checkpoints/best_model_aoi.pth',
                       'iforest_path': 'checkpoints/isolation_forest_model.pkl',
                       'kmeans_path': 'checkpoints/kmeans_model.pkl',
                       'gate_path': 'checkpoints/GATE_model.pt',
                       'device': 'cuda' or 'cpu'
                   }
        """
        self.device = torch.device(config.get('device', 'cpu'))
        
        print(f"🚀 Initializing SONAR 2.0 Predictor on {self.device}")
        
        # Load PyTorch models
        self.autoencoder = self._load_autoencoder(config['autoencoder_path'])
        self.encoder = self._load_encoder(config['autoencoder_path'])
        
        # Load sklearn models
        self.iforest = self._load_sklearn_model(config['iforest_path'])
        self.kmeans = self._load_sklearn_model(config['kmeans_path'])
        
        # Load GATE CNN model
        self.gate_cnn = self._load_gate_model(config['gate_path'])
        
        # Load reference embeddings for archaeological similarity
        self.reference_embeddings = self._load_reference_embeddings(
            config.get('reference_embeddings_path')
        )
        
        # OPTIMIZATION: Pre-compute K-Means cluster centers as torch tensor
        self.kmeans_centers_tensor = torch.FloatTensor(
            self.kmeans.cluster_centers_
        ).to(self.device)
        
        # OPTIMIZATION: Pre-compute cluster center norms for faster distance calculation
        self.kmeans_max_distance = float(
            np.max(np.linalg.norm(
                self.kmeans.cluster_centers_[None, :] - 
                self.kmeans.cluster_centers_[:, None],
                axis=2
            ))
        )
        
        # OPTIMIZATION: Convert reference embeddings to torch tensor if available
        if self.reference_embeddings is not None:
            self.reference_embeddings_tensor = torch.FloatTensor(
                self.reference_embeddings
            ).to(self.device)
            # Pre-compute norms for cosine similarity
            self.ref_norms = torch.norm(self.reference_embeddings_tensor, dim=1, keepdim=True)
        else:
            self.reference_embeddings_tensor = None
            self.ref_norms = None
        
        # OPTIMIZATION: torch.compile for PyTorch 2.0+ (if available)
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                print("⚡ Applying torch.compile optimizations...")
                self.autoencoder = torch.compile(self.autoencoder, mode='reduce-overhead')
                self.encoder = torch.compile(self.encoder, mode='reduce-overhead')
                self.gate_cnn = torch.compile(self.gate_cnn, mode='reduce-overhead')
                print("✅ torch.compile applied successfully")
            except Exception as e:
                print(f"⚠️  torch.compile failed: {e}, continuing without it")
        
        print("✅ All models loaded and optimized successfully")
    
    def _load_autoencoder(self, path: str) -> torch.nn.Module:
        """Load autoencoder model"""
        model = ResUNetAutoencoder(in_channels=7, latent_dim=256)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        # OPTIMIZATION: Set to inference mode (disables autograd)
        if hasattr(torch, 'inference_mode'):
            model = torch.nn.utils.parametrize.cached()(model)
        
        return model
    
    def _load_encoder(self, path: str) -> torch.nn.Module:
        """Load encoder (same checkpoint as autoencoder)"""
        encoder = ResUNetEncoder(in_channels=7, embedding_dim=128)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load only encoder parts (strict=False to ignore decoder weights)
        encoder.load_state_dict(state_dict, strict=False)
        encoder.to(self.device)
        encoder.eval()
        return encoder
    
    def _load_gate_model(self, path: str) -> torch.nn.Module:
        """Load GATE CNN model with flexible checkpoint loading"""
        model = GateNetwork(in_channels=4, dropout=0.3)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Try direct loading first
        try:
            model.load_state_dict(state_dict)
            print("✓ GATE model loaded directly")
        except RuntimeError as e:
            print(f"⚠️  Direct loading failed, trying key mapping...")
            
            # Map old keys to new keys
            new_state_dict = {}
            
            # The checkpoint has 'network.X' but model expects 'convY.Z' and 'classifier.Z'
            # We need to map based on the layer indices
            
            # First, let's just try loading with strict=False to see what works
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
            if len(missing) > 0:
                print(f"⚠️  Could not load GATE model properly.")
                print(f"    Missing keys: {len(missing)}")
                print(f"    Unexpected keys: {len(unexpected)}")
                print(f"    The model will use random initialization for GATE!")
                print(f"    This will affect prediction quality.")
            else:
                print("✓ GATE model loaded with partial match")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_sklearn_model(self, path: str):
        """Load sklearn/joblib models"""
        return joblib.load(path)
    
    def _load_reference_embeddings(self, path: Optional[str]) -> Optional[np.ndarray]:
        """Load reference archaeological site embeddings"""
        if path and Path(path).exists():
            return np.load(path)
        return None
    
    @torch.inference_mode()
    def _compute_autoencoder_scores_batch(self, patches_tensor: torch.Tensor) -> torch.Tensor:
        """
        OPTIMIZED: Compute autoencoder scores for batch
        
        Args:
            patches_tensor: (B, 7, 64, 64)
        
        Returns:
            scores: (B,) tensor of anomaly scores
        """
        reconstructed, _ = self.autoencoder(patches_tensor)
        mse = torch.mean((patches_tensor - reconstructed) ** 2, dim=[1, 2, 3])
        scores = torch.sigmoid(mse * 10)
        return scores
    
    @torch.inference_mode()
    def _compute_embeddings_batch(self, patches_tensor: torch.Tensor) -> torch.Tensor:
        """
        OPTIMIZED: Compute embeddings for batch
        
        Args:
            patches_tensor: (B, 7, 64, 64)
        
        Returns:
            embeddings: (B, 128) tensor
        """
        return self.encoder(patches_tensor)
    
    def _compute_iforest_scores_batch(self, embeddings_np: np.ndarray) -> np.ndarray:
        """
        OPTIMIZED: Compute Isolation Forest scores for batch
        
        Args:
            embeddings_np: (B, 128) numpy array
        
        Returns:
            scores: (B,) numpy array
        """
        iforest_scores = self.iforest.decision_function(embeddings_np)
        # Normalize to 0-1 using sigmoid
        scores = 1 / (1 + np.exp(-iforest_scores))
        return scores
    
    def _compute_kmeans_scores_batch(self, embeddings_tensor: torch.Tensor) -> torch.Tensor:
        """
        OPTIMIZED: Compute K-Means scores for batch using vectorized operations
        
        Args:
            embeddings_tensor: (B, 128) torch tensor
        
        Returns:
            scores: (B,) torch tensor
        """
        # Compute distances to all cluster centers: (B, K)
        # embeddings_tensor: (B, 128)
        # kmeans_centers_tensor: (K, 128)
        # distances: (B, K)
        distances = torch.cdist(
            embeddings_tensor.unsqueeze(0), 
            self.kmeans_centers_tensor.unsqueeze(0)
        ).squeeze(0)
        
        # Get minimum distance for each sample
        min_distances = torch.min(distances, dim=1)[0]
        
        # Normalize using pre-computed max distance
        scores = 1 - (min_distances / self.kmeans_max_distance)
        
        return scores
    
    def _compute_arch_similarity_batch(self, embeddings_tensor: torch.Tensor) -> torch.Tensor:
        """
        OPTIMIZED: Compute archaeological similarity scores for batch
        
        Args:
            embeddings_tensor: (B, 128) torch tensor
        
        Returns:
            scores: (B,) torch tensor
        """
        if self.reference_embeddings_tensor is None:
            return torch.zeros(embeddings_tensor.shape[0], device=self.device)
        
        # Compute cosine similarity: (B, num_references)
        # embeddings_tensor: (B, 128)
        # reference_embeddings_tensor: (num_references, 128)
        
        # Normalize embeddings
        embeddings_norm = torch.norm(embeddings_tensor, dim=1, keepdim=True)
        embeddings_normalized = embeddings_tensor / (embeddings_norm + 1e-8)
        
        # Normalize reference embeddings (already pre-computed)
        ref_normalized = self.reference_embeddings_tensor / (self.ref_norms + 1e-8)
        
        # Compute cosine similarity
        similarities = torch.mm(embeddings_normalized, ref_normalized.T)
        
        # Get max similarity for each sample
        max_similarities = torch.max(similarities, dim=1)[0]
        
        return max_similarities
    
    @torch.inference_mode()
    def _compute_gate_predictions_batch(self, prob_matrices: torch.Tensor) -> torch.Tensor:
        """
        OPTIMIZED: Compute GATE predictions for batch
        
        Args:
            prob_matrices: (B, 4, 64, 64) probability matrices
        
        Returns:
            predictions: (B,) tensor of final predictions
        """
        _, gate_probs = self.gate_cnn(prob_matrices)
        return gate_probs.squeeze(1)
    
    def predict_single_patch(self, patch: np.ndarray) -> Dict:
        """
        Run full ensemble prediction on a single patch
        (Calls optimized batch method with batch_size=1)
        
        Args:
            patch: numpy array of shape (7, 64, 64) - 7 channels
        
        Returns:
            {
                'gate_prediction': float (0-1),
                'is_anomaly': bool,
                'model_scores': {
                    'autoencoder': float,
                    'iforest': float,
                    'kmeans': float,
                    'arch_similarity': float
                },
                'embedding': np.ndarray (128-dim)
            }
        """
        # Convert to batch of 1
        patches = np.expand_dims(patch, axis=0)
        results = self.predict_batch(patches, batch_size=1)
        return results[0]
    
    def predict_batch(self, patches: np.ndarray, batch_size: int = 32) -> List[Dict]:
        """
        OPTIMIZED: Run predictions on multiple patches efficiently with true batching
        
        Args:
            patches: numpy array of shape (N, 7, 64, 64)
            batch_size: number of patches to process at once
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        num_patches = patches.shape[0]
        
        for i in range(0, num_patches, batch_size):
            batch = patches[i:i+batch_size]
            batch_results = self._predict_batch_optimized(batch)
            results.extend(batch_results)
        
        return results
    
    def _predict_batch_optimized(self, batch: np.ndarray) -> List[Dict]:
        """
        OPTIMIZED: Core batch prediction with vectorized operations
        
        Args:
            batch: numpy array of shape (B, 7, 64, 64)
        
        Returns:
            List of B prediction dictionaries
        """
        batch_size = batch.shape[0]
        
        # Convert to tensor once
        patches_tensor = torch.FloatTensor(batch).to(self.device)
        
        # 1. Autoencoder anomaly scores (batched)
        autoencoder_scores = self._compute_autoencoder_scores_batch(patches_tensor)
        
        # 2. Get embeddings (batched)
        embeddings_tensor = self._compute_embeddings_batch(patches_tensor)
        
        # Keep embeddings on GPU for further processing
        embeddings_np = embeddings_tensor.cpu().numpy()
        
        # 3. Isolation Forest scores (numpy, batched)
        iforest_scores = self._compute_iforest_scores_batch(embeddings_np)
        
        # 4. K-Means scores (torch, batched)
        kmeans_scores = self._compute_kmeans_scores_batch(embeddings_tensor)
        
        # 5. Archaeological similarity (torch, batched)
        arch_similarities = self._compute_arch_similarity_batch(embeddings_tensor)
        
        # 6. GATE CNN meta-learner (batched)
        # Create probability matrices (B, 4, 64, 64)
        prob_matrices = torch.zeros(batch_size, 4, 64, 64, device=self.device)
        
        # Fill with scores (broadcast across spatial dimensions)
        prob_matrices[:, 0, :, :] = autoencoder_scores.view(-1, 1, 1)
        prob_matrices[:, 1, :, :] = torch.FloatTensor(iforest_scores).to(self.device).view(-1, 1, 1)
        prob_matrices[:, 2, :, :] = kmeans_scores.view(-1, 1, 1)
        prob_matrices[:, 3, :, :] = arch_similarities.view(-1, 1, 1)
        
        # Get GATE predictions
        gate_predictions = self._compute_gate_predictions_batch(prob_matrices)
        
        # Convert all to numpy for results
        autoencoder_scores_np = autoencoder_scores.cpu().numpy()
        kmeans_scores_np = kmeans_scores.cpu().numpy()
        arch_similarities_np = arch_similarities.cpu().numpy()
        gate_predictions_np = gate_predictions.cpu().numpy()
        
        # Build results list
        results = []
        for idx in range(batch_size):
            gate_pred = float(gate_predictions_np[idx])
            
            results.append({
                'gate_prediction': gate_pred,
                'is_anomaly': bool(gate_pred > 0.5),
                'model_scores': {
                    'autoencoder': float(autoencoder_scores_np[idx]),
                    'iforest': float(iforest_scores[idx]),
                    'kmeans': float(kmeans_scores_np[idx]),
                    'arch_similarity': float(arch_similarities_np[idx])
                },
                'embedding': embeddings_np[idx]
            })
        
        return results
    
    def predict_aoi(
        self, 
        patches: np.ndarray, 
        metadata: List[Dict],
        threshold: float = 0.5
    ) -> Dict:
        """
        Run predictions on full Area of Interest (AOI)
        
        Args:
            patches: numpy array of shape (N, 7, 64, 64)
            metadata: list of patch metadata dicts with 'row', 'col', 'patch_id'
            threshold: detection threshold (default 0.5)
        
        Returns:
            {
                'predictions': List[Dict],  # per-patch predictions
                'summary': {
                    'total_patches': int,
                    'anomalies_detected': int,
                    'anomaly_percentage': float,
                    'mean_confidence': float,
                    'top_candidates': List[Dict]  # sorted by score
                },
                'metadata': List[Dict]  # original metadata
            }
        """
        print(f"🔎 Analyzing {len(patches)} patches...")
        
        # Run predictions (uses optimized batching internally)
        predictions = self.predict_batch(patches, batch_size=32)
        
        # Combine with metadata
        for pred, meta in zip(predictions, metadata):
            pred.update(meta)
        
        # OPTIMIZATION: Vectorized statistics using numpy
        all_scores = np.array([p['gate_prediction'] for p in predictions])
        
        # Calculate statistics
        anomalies = [p for p in predictions if p['gate_prediction'] >= threshold]
        
        # Sort by confidence (top 20)
        top_candidates = sorted(
            predictions, 
            key=lambda x: x['gate_prediction'], 
            reverse=True
        )[:20]
        
        summary = {
            'total_patches': len(patches),
            'anomalies_detected': len(anomalies),
            'anomaly_percentage': (len(anomalies) / len(patches)) * 100,
            'mean_confidence': float(np.mean(all_scores)),
            'max_confidence': float(np.max(all_scores)),
            'min_confidence': float(np.min(all_scores)),
            'top_candidates': [
                {
                    'patch_id': p.get('patch_id', 'unknown'),
                    'row': p.get('row', -1),
                    'col': p.get('col', -1),
                    'confidence': p['gate_prediction'],
                    'is_anomaly': p['is_anomaly']
                }
                for p in top_candidates
            ]
        }
        
        print(f"✅ Analysis complete: {len(anomalies)} anomalies detected")
        
        return {
            'predictions': predictions,
            'summary': summary,
            'metadata': metadata
        }


# ==============================================================================
# CONVENIENCE FUNCTION (used by API routes)
# ==============================================================================

def get_predictor(config: Dict) -> SONARPredictor:
    """
    Factory function to create predictor instance
    Called once by FastAPI on startup
    """
    return SONARPredictor(config)