"""
Robust MLP classifier with enhanced regularization for better generalization.
Includes improved dropout, weight decay, batch normalization, and architecture improvements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class RobustMLPClassifier(nn.Module):
    """
    Enhanced MLP classifier with multiple regularization techniques
    for improved generalization on accent classification.
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.5,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 activation: str = 'relu',
                 use_residual: bool = False,
                 use_spectral_norm: bool = False):
        """
        Initialize robust MLP classifier.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability (higher for better regularization)
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            activation: Activation function ('relu', 'gelu', 'swish')
            use_residual: Whether to use residual connections
            use_spectral_norm: Whether to use spectral normalization
        """
        super(RobustMLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            linear = nn.Linear(prev_dim, hidden_dim)
            
            # Apply spectral normalization if requested
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            
            layers.append(linear)
            
            # Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'swish':
                layers.append(nn.SiLU())  # Swish/SiLU
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Dropout (higher rate for better regularization)
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation, no dropout)
        output_layer = nn.Linear(prev_dim, num_classes)
        if use_spectral_norm:
            output_layer = nn.utils.spectral_norm(output_layer)
        layers.append(output_layer)
        
        self.model = nn.Sequential(*layers)
        
        # Store dimensions for residual connections
        self.layer_dims = [input_dim] + hidden_dims
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for ReLU
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional residual connections.
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        if not self.use_residual:
            return self.model(x)
        
        # Forward with residual connections
        current = x
        layer_idx = 0
        
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                # Store input for potential residual connection
                residual_input = current
                current = layer(current)
                
                # Add residual connection if dimensions match
                if (layer_idx > 0 and 
                    residual_input.shape[-1] == current.shape[-1] and
                    layer_idx < len(self.hidden_dims)):  # Not output layer
                    current = current + residual_input
                
                layer_idx += 1
            else:
                current = layer(current)
        
        return current


class AdaptiveMLPClassifier(nn.Module):
    """
    Adaptive MLP that adjusts its complexity based on input uncertainty.
    Uses multiple prediction heads for ensemble-like behavior.
    """
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_dims: List[int] = [256, 128],
                 num_heads: int = 3,
                 dropout: float = 0.4):
        """
        Initialize adaptive MLP.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of classes
            hidden_dims: Hidden layer dimensions
            num_heads: Number of prediction heads
            dropout: Dropout rate
        """
        super(AdaptiveMLPClassifier, self).__init__()
        
        self.num_heads = num_heads
        
        # Shared feature extractor
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:  # All but last layer
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_features = nn.Sequential(*shared_layers)
        
        # Multiple prediction heads
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.BatchNorm1d(hidden_dims[-1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * (1.0 + i * 0.1)),  # Varying dropout
                nn.Linear(hidden_dims[-1], num_classes)
            )
            self.heads.append(head)
        
        # Attention mechanism for head weighting
        self.attention = nn.Sequential(
            nn.Linear(prev_dim, num_heads),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor, return_individual_heads: bool = False):
        """
        Forward pass with multiple heads and attention weighting.
        
        Args:
            x: Input features
            return_individual_heads: Whether to return individual head outputs
        
        Returns:
            Combined predictions or individual head outputs
        """
        # Extract shared features
        shared_feat = self.shared_features(x)
        
        # Get predictions from all heads
        head_outputs = []
        for head in self.heads:
            head_output = head(shared_feat)
            head_outputs.append(head_output)
        
        if return_individual_heads:
            return torch.stack(head_outputs, dim=1)  # (batch, num_heads, num_classes)
        
        # Compute attention weights
        attention_weights = self.attention(shared_feat)  # (batch, num_heads)
        
        # Weighted combination of head outputs
        stacked_outputs = torch.stack(head_outputs, dim=2)  # (batch, num_classes, num_heads)
        weighted_output = torch.bmm(
            stacked_outputs, 
            attention_weights.unsqueeze(2)
        ).squeeze(2)  # (batch, num_classes)
        
        return weighted_output


class UncertaintyAwareMLPClassifier(nn.Module):
    """
    MLP classifier that also predicts its own uncertainty.
    Helps identify when the model is likely to be wrong.
    """
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_dims: List[int] = [256, 128],
                 dropout: float = 0.4):
        """
        Initialize uncertainty-aware MLP.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of classes
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super(UncertaintyAwareMLPClassifier, self).__init__()
        
        # Shared feature extractor
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_features = nn.Sequential(*shared_layers)
        
        # Classification head
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # Uncertainty head (predicts log variance)
        self.uncertainty = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[-1] // 2, 1),
            nn.Softplus()  # Ensures positive output
        )
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = False):
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: Input features
            return_uncertainty: Whether to return uncertainty estimates
        
        Returns:
            Class logits and optionally uncertainty estimates
        """
        shared_feat = self.shared_features(x)
        
        # Get classification logits
        logits = self.classifier(shared_feat)
        
        if return_uncertainty:
            # Get uncertainty (aleatoric uncertainty)
            uncertainty = self.uncertainty(shared_feat)
            return logits, uncertainty
        
        return logits


def create_robust_model(input_dim: int, 
                       num_classes: int,
                       model_type: str = 'robust_mlp',
                       **kwargs) -> nn.Module:
    """
    Factory function to create different types of robust models.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of classes
        model_type: Type of model ('robust_mlp', 'adaptive_mlp', 'uncertainty_mlp')
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Initialized model
    """
    if model_type == 'robust_mlp':
        return RobustMLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'adaptive_mlp':
        return AdaptiveMLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'uncertainty_mlp':
        return UncertaintyAwareMLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("Testing robust MLP models...")
    
    batch_size = 16
    input_dim = 1536  # HuBERT features
    num_classes = 6   # Number of accents
    
    # Test robust MLP
    print("\n1. Robust MLP Classifier")
    robust_model = RobustMLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[256, 128],
        dropout=0.5,
        use_batch_norm=True,
        activation='relu'
    )
    
    x = torch.randn(batch_size, input_dim)
    output = robust_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in robust_model.parameters()):,}")
    
    # Test adaptive MLP
    print("\n2. Adaptive MLP Classifier")
    adaptive_model = AdaptiveMLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[256, 128],
        num_heads=3
    )
    
    output = adaptive_model(x)
    individual_heads = adaptive_model(x, return_individual_heads=True)
    print(f"Combined output shape: {output.shape}")
    print(f"Individual heads shape: {individual_heads.shape}")
    print(f"Parameters: {sum(p.numel() for p in adaptive_model.parameters()):,}")
    
    # Test uncertainty-aware MLP
    print("\n3. Uncertainty-Aware MLP Classifier")
    uncertainty_model = UncertaintyAwareMLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[256, 128]
    )
    
    logits, uncertainty = uncertainty_model(x, return_uncertainty=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Parameters: {sum(p.numel() for p in uncertainty_model.parameters()):,}")
    
    print("\n✓ All robust models tested successfully!")