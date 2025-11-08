"""
CNN classifier for sequence-based accent classification.
"""

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """1D CNN for sequence-based accent classification."""
    
    def __init__(self, input_dim, num_classes, num_filters=[128, 256, 256],
                 kernel_sizes=[3, 3, 3], dropout=0.3, pool_size=2):
        """
        Initialize CNN classifier.
        
        Args:
            input_dim: Input feature dimension (e.g., n_mfcc or hidden_size)
            num_classes: Number of output classes
            num_filters: List of filter sizes for conv layers
            kernel_sizes: List of kernel sizes for conv layers
            dropout: Dropout probability
            pool_size: Max pooling size
        """
        super(CNNClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_dim
        
        for out_channels, kernel_size in zip(num_filters, kernel_sizes):
            conv_layers.append(nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(pool_size))
            conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(num_filters[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, feature_dim, seq_len)
        
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        # Apply conv layers
        x = self.conv_layers(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # Fully connected
        x = self.fc(x)
        
        return x


if __name__ == "__main__":
    # Test model
    batch_size = 16
    feature_dim = 40  # MFCC dimension
    seq_len = 100  # Number of frames
    num_classes = 10
    
    model = CNNClassifier(feature_dim, num_classes)
    x = torch.randn(batch_size, feature_dim, seq_len)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
