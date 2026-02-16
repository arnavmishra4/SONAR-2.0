import torch
import torch.nn as nn

class GateNetwork(nn.Module):
    """
    GATE Network - MLP Architecture (ACTUAL trained model)
    
    Takes flattened probability matrices and predicts archaeological sites.
    
    Architecture:
    - Input: (batch, 4, 64, 64) probability matrices -> flattened to (batch, 16384)
    - MLP: 16384 -> 512 -> 256 -> 128 -> 64 -> 1
    - Output: (batch, 1) sigmoid probability
    
    This is the ACTUAL architecture your checkpoint was trained with!
    """
    
    def __init__(self, in_channels: int = 4, spatial_size: int = 64, dropout: float = 0.3):
        super().__init__()
        
        # Calculate flattened input size
        self.flatten_size = in_channels * spatial_size * spatial_size  # 4 * 64 * 64 = 16384
        
        self.network = nn.Sequential(
            nn.Flatten(),  # (batch, 4, 64, 64) -> (batch, 16384)
            
            # Layer 1: 16384 -> 512
            nn.Linear(self.flatten_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 2: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 3: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 4: 128 -> 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Output: 64 -> 1
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 4, 64, 64) probability matrices
        
        Returns:
            logits: (batch, 1) raw logits
            probs: (batch, 1) sigmoid probabilities
        """
        logits = self.network(x)
        probs = torch.sigmoid(logits)
        return logits, probs
    
    def predict(self, x):
        """Predict class labels (0 or 1)"""
        _, probs = self.forward(x)
        return (probs > 0.5).long()
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)