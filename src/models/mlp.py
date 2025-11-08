"""
MLP (Multi-Layer Perceptron) classifier for NLI.
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for accent classification."""
    
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256, 128],
                 dropout=0.3, activation='relu'):
        """
        Initialize MLP classifier.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes (languages)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super(MLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        return self.model(x)


if __name__ == "__main__":
    # Test model
    batch_size = 32
    input_dim = 200  # MFCC stats dimension
    num_classes = 10  # Number of languages
    
    model = MLPClassifier(input_dim, num_classes)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
