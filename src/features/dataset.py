"""
PyTorch Dataset classes for NLI.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import numpy as np
from pathlib import Path


class NLIDataset(Dataset):
    """Dataset for Native Language Identification."""
    
    def __init__(self, features_list, label_encoder=None):
        """
        Initialize dataset.
        
        Args:
            features_list: List of feature dictionaries
            label_encoder: sklearn LabelEncoder for native language labels
        """
        self.features_list = features_list
        self.label_encoder = label_encoder
        
        # Extract unique languages and create encoder if not provided
        if self.label_encoder is None:
            from sklearn.preprocessing import LabelEncoder
            languages = [item['native_language'] for item in features_list]
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(languages)
        
        self.num_classes = len(self.label_encoder.classes_)
    
    def __len__(self):
        return len(self.features_list)
    
    def __getitem__(self, idx):
        item = self.features_list[idx]
        
        # Get features
        if 'features' in item:
            features = item['features']
        elif 'embeddings' in item:
            # For HuBERT embeddings, extract specific layer or use first available
            embeddings_dict = item['embeddings']
            # Use the first available layer
            layer_key = list(embeddings_dict.keys())[0]
            features = embeddings_dict[layer_key]
        else:
            raise ValueError(f"No features found in item {idx}")
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        
        # Get label
        language = item['native_language']
        label = self.label_encoder.transform([language])[0]
        label = torch.LongTensor([label])[0]
        
        return features, label, item.get('speaker_id', 'unknown')


def load_features_and_create_dataset(features_path, split_csv=None, label_encoder=None):
    """
    Load features and create dataset.
    
    Args:
        features_path: Path to pickled features file
        split_csv: Optional CSV file to filter features
        label_encoder: Optional label encoder
    
    Returns:
        dataset: NLIDataset instance
        label_encoder: The label encoder used
    """
    print(f"Loading features from: {features_path}")
    
    with open(features_path, 'rb') as f:
        features_list = pickle.load(f)
    
    print(f"✓ Loaded {len(features_list)} feature samples")
    
    # Filter by split if provided
    if split_csv:
        split_df = pd.read_csv(split_csv)
        # Get list of filepaths from split
        split_filepaths = set(split_df['filepath'].apply(lambda x: str(Path(x).stem)).tolist())
        
        # Filter features
        filtered_features = []
        for item in features_list:
            filepath_stem = str(Path(item['filepath']).stem)
            if filepath_stem in split_filepaths:
                filtered_features.append(item)
        
        features_list = filtered_features
        print(f"✓ Filtered to {len(features_list)} samples based on split CSV")
    
    # Create dataset
    dataset = NLIDataset(features_list, label_encoder=label_encoder)
    
    print(f"✓ Dataset created with {len(dataset)} samples")
    print(f"  Number of classes: {dataset.num_classes}")
    print(f"  Classes: {list(dataset.label_encoder.classes_)}")
    
    return dataset, dataset.label_encoder


if __name__ == "__main__":
    # Test dataset loading
    print("Dataset class created successfully")
    print("Use load_features_and_create_dataset() to load features")
