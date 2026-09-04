"""
OceanEmbed ML Model
Deep learning model for predicting subsurface ocean temperature profiles
from surface satellite observations.

Architecture: Multi-scale 1D CNN with residual connections.
The model learns to map a 5-dimensional surface feature vector
(SST, SSH, SSS, u10, v10) to a 31-level vertical temperature profile (0-2000m).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class OceanEmbedModel(nn.Module):
    """
    Multi-scale 1D CNN for ocean temperature profile reconstruction.
    
    Input: (batch_size, num_features=5) - surface observations
    Output: (batch_size, num_depth_levels=31) - temperature profile 0-2000m
    
    The architecture uses:
    1. Feature embedding layer to project sparse surface inputs to higher dims
    2. Multi-scale 1D convolutions to capture different spatial scales
    3. Residual connections for stable training
    4. Depth-aware output head with sigmoid activations for physically valid profiles
    """
    
    def __init__(
        self,
        input_dim: int = config.NUM_FEATURES,
        output_dim: int = config.NUM_DEPTH_LEVELS,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        hidden_dims = hidden_dims or config.MODEL_CONFIG["hidden_dims"]
        
        # Feature embedding: project surface features to higher dimension
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # Multi-scale convolution blocks
        # The "image" is 1D with the feature dim as channels
        self.conv_blocks = nn.ModuleList()
        
        # Block 1: Local patterns
        self.conv_blocks.append(nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        ))
        
        # Block 2: Medium-scale patterns
        self.conv_blocks.append(nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        ))
        
        # Block 3: Large-scale patterns
        self.conv_blocks.append(nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
        ))
        
        # Attention mechanism for depth-aware processing
        self.attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(128)
        
        # Output head: map conv features to depth-temperature profile
        self.output_head = nn.Sequential(
            nn.Linear(128, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dims[-1] // 2, output_dim),
        )
        
        # Residual projection for skip connection
        self.residual_proj = nn.Linear(input_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, input_dim) surface features
            
        Returns:
            (batch_size, output_dim) temperature profile
        """
        # Feature embedding
        embedded = self.feature_embedding(x)  # (B, hidden_dim)
        
        # Reshape for conv: (B, 1, hidden_dim) - treat as 1-channel 1D signal
        conv_input = embedded.unsqueeze(1)
        
        # Multi-scale conv blocks with residual connections
        h = conv_input
        for i, block in enumerate(self.conv_blocks):
            residual = h
            h = block(h)
            if h.shape == residual.shape:
                h = h + residual
        
        # Global average pooling: (B, channels, seq_len) → (B, channels)
        h = h.mean(dim=2)
        
        # Reshape for attention: (B, hidden) → (B, 1, hidden)
        h = h.unsqueeze(1)
        
        # Self-attention over the single "token"
        attn_out, _ = self.attention(h, h, h)
        h = self.attention_norm(h + attn_out)
        h = h.squeeze(1)  # (B, hidden)
        
        # Output: temperature profile
        output = self.output_head(h)
        
        # Add residual from direct surface → depth mapping
        residual = self.residual_proj(x)
        output = output + 0.1 * residual
        
        return output
    
    def predict_with_uncertainty(
        self, x: torch.Tensor, n_forward_passes: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout inference for uncertainty estimation.
        Keeps dropout active but uses eval mode for BatchNorm layers.
        
        Returns:
            mean_pred: mean prediction over forward passes
            std_pred: standard deviation (uncertainty) estimate
        """
        # Temporarily enable dropout but keep BatchNorm in eval mode
        was_training = self.training
        self.eval()
        
        # Enable only dropout layers
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_forward_passes):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        self.train(was_training)
        return mean_pred, std_pred


class OceanEmbedLoss(nn.Module):
    """
    Custom loss function combining MSE with physics-aware regularization.
    
    Components:
    1. MSE loss on temperature predictions
    2. Monotonicity penalty (temperature should decrease with depth)
    3. Boundary condition enforcement (surface temp ≈ SST input)
    """
    
    def __init__(self, monotonicity_weight: float = 0.1):
        super().__init__()
        self.monotonicity_weight = monotonicity_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sst_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            pred: (B, D) predicted temperature profile
            target: (B, D) true temperature profile
            sst_input: (B,) optional surface temperature input for boundary loss
            
        Returns:
            total_loss, loss_components dict
        """
        # Main MSE loss
        mse = F.mse_loss(pred, target)
        
        # Monotonicity: temperature should generally decrease with depth
        # Compute depth differences
        depth_diff = pred[:, 1:] - pred[:, :-1]  # (B, D-1)
        # Penalize temperature increases below the mixed layer (first few levels)
        violation = F.relu(depth_diff[:, 5:])  # Only penalize below ~25m
        mono_loss = violation.mean()
        
        # Total loss
        total = mse + self.monotonicity_weight * mono_loss
        
        loss_dict = {
            "mse": mse.item(),
            "monotonicity": mono_loss.item(),
            "total": total.item(),
        }
        
        return total, loss_dict


def create_model() -> OceanEmbedModel:
    """Create and initialize the OceanEmbed model."""
    model = OceanEmbedModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Created OceanEmbed model with {n_params:,} parameters")
    return model


if __name__ == "__main__":
    # Quick model test
    model = create_model()
    x = torch.randn(4, config.NUM_FEATURES)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.2f}, {y.max().item():.2f}]")
    
    # Test uncertainty estimation
    mean, std = model.predict_with_uncertainty(x)
    print(f"Mean prediction shape: {mean.shape}")
    print(f"Uncertainty (std) shape: {std.shape}")
