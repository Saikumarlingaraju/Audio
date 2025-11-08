"""
Transformer classifier for accent classification.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    """Transformer encoder for accent classification."""
    
    def __init__(self, input_dim, num_classes, d_model=256, nhead=8,
                 num_layers=4, dim_feedforward=1024, dropout=0.3):
        """
        Initialize Transformer classifier.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
        """
        super(TransformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, seq_len, feature_dim)
            mask: Attention mask (optional)
        
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        # Project input to model dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        output = self.classifier(x)
        
        return output


if __name__ == "__main__":
    # Test model
    batch_size = 16
    seq_len = 100
    feature_dim = 40
    num_classes = 10
    
    model = TransformerClassifier(feature_dim, num_classes)
    x = torch.randn(batch_size, seq_len, feature_dim)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
