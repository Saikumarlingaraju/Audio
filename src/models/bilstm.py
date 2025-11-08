"""
BiLSTM classifier for sequence-based accent classification.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for accent classification."""
    
    def __init__(self, input_dim, num_classes, hidden_size=256, 
                 num_layers=2, dropout=0.3):
        """
        Initialize BiLSTM classifier.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(BiLSTMClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism (optional)
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def attention_pooling(self, lstm_output):
        """
        Apply attention mechanism for pooling.
        
        Args:
            lstm_output: LSTM output (batch_size, seq_len, hidden_size*2)
        
        Returns:
            Pooled output (batch_size, hidden_size*2)
        """
        # Compute attention scores
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum
        pooled = torch.sum(lstm_output * attention_weights, dim=1)
        
        return pooled
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, seq_len, feature_dim)
        
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_size*2)
        
        # Attention pooling
        pooled = self.attention_pooling(lstm_out)
        
        # Fully connected
        output = self.fc(pooled)
        
        return output


if __name__ == "__main__":
    # Test model
    batch_size = 16
    seq_len = 100
    feature_dim = 40
    num_classes = 10
    
    model = BiLSTMClassifier(feature_dim, num_classes)
    x = torch.randn(batch_size, seq_len, feature_dim)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
