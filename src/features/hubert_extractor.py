"""
HuBERT embedding extraction from all layers with caching.
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')


class HuBERTExtractor:
    """Extract HuBERT embeddings from audio files."""
    
    def __init__(self, model_name="facebook/hubert-base-ls960", device=None):
        """
        Initialize HuBERT extractor.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading HuBERT model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load feature extractor and model
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Get model configuration
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        
        print(f"✓ Model loaded successfully")
        print(f"  Number of layers: {self.n_layers}")
        print(f"  Hidden size: {self.hidden_size}")
    
    def load_audio(self, filepath, target_sr=16000):
        """Load and resample audio file."""
        try:
            waveform, sr = torchaudio.load(filepath)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            
            # Convert to numpy and squeeze
            audio = waveform.squeeze().numpy()
            
            return audio, target_sr
            
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None, None
    
    @torch.no_grad()
    def extract_embeddings(self, audio, sr=16000, extract_layer=None, 
                          pooling='mean'):
        """
        Extract HuBERT embeddings from audio.
        
        Args:
            audio: Audio signal (numpy array)
            sr: Sampling rate
            extract_layer: Specific layer to extract (None = all layers)
            pooling: Pooling method ('mean', 'max', or None for frames)
        
        Returns:
            dict: Embeddings and metadata
        """
        try:
            # Prepare input
            inputs = self.feature_extractor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Extract hidden states
            hidden_states = outputs.hidden_states  # Tuple of (n_layers+1,) tensors
            
            # Extract embeddings
            embeddings = {}
            
            if extract_layer is not None:
                # Extract specific layer
                layer_output = hidden_states[extract_layer].squeeze(0).cpu().numpy()
                
                if pooling == 'mean':
                    embeddings[f'layer_{extract_layer}'] = np.mean(layer_output, axis=0)
                elif pooling == 'max':
                    embeddings[f'layer_{extract_layer}'] = np.max(layer_output, axis=0)
                else:
                    embeddings[f'layer_{extract_layer}'] = layer_output
            
            else:
                # Extract all layers
                for layer_idx in range(len(hidden_states)):
                    layer_output = hidden_states[layer_idx].squeeze(0).cpu().numpy()
                    
                    if pooling == 'mean':
                        embeddings[f'layer_{layer_idx}'] = np.mean(layer_output, axis=0)
                    elif pooling == 'max':
                        embeddings[f'layer_{layer_idx}'] = np.max(layer_output, axis=0)
                    else:
                        embeddings[f'layer_{layer_idx}'] = layer_output
            
            return embeddings
            
        except Exception as e:
            print(f"Error extracting embeddings: {str(e)}")
            return None
    
    def extract_from_file(self, filepath, extract_layer=None, pooling='mean'):
        """
        Extract HuBERT embeddings from audio file.
        
        Args:
            filepath: Path to audio file
            extract_layer: Specific layer to extract (None = all layers)
            pooling: Pooling method ('mean', 'max', or None)
        
        Returns:
            dict: Embeddings and metadata
        """
        # Load audio
        audio, sr = self.load_audio(filepath)
        
        if audio is None:
            return None
        
        # Extract embeddings
        embeddings = self.extract_embeddings(
            audio, 
            sr=sr, 
            extract_layer=extract_layer,
            pooling=pooling
        )
        
        if embeddings is None:
            return None
        
        return {
            'embeddings': embeddings,
            'filepath': str(filepath),
            'n_layers': len(embeddings),
            'hidden_size': self.hidden_size
        }


def extract_features_from_dataset(metadata_path, audio_dir, output_dir,
                                  extract_layer=None, pooling='mean',
                                  model_name="facebook/hubert-base-ls960",
                                  batch_size=8):
    """
    Extract HuBERT features from entire dataset.
    
    Args:
        metadata_path: Path to metadata CSV
        audio_dir: Directory containing audio files
        output_dir: Directory to save features
        extract_layer: Specific layer to extract (None = all layers)
        pooling: Pooling method ('mean', 'max', or None for frames)
        model_name: HuggingFace model name
        batch_size: Batch size for processing (currently unused, processes one at a time)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    print(f"✓ Loaded metadata: {len(df)} samples")
    
    # Initialize extractor
    extractor = HuBERTExtractor(model_name=model_name)
    
    print("\n" + "=" * 80)
    print("HuBERT Extraction Configuration")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Device: {extractor.device}")
    print(f"Number of layers: {extractor.n_layers}")
    print(f"Hidden size: {extractor.hidden_size}")
    print(f"Extract layer: {extract_layer if extract_layer is not None else 'All layers'}")
    print(f"Pooling: {pooling}")
    print("=" * 80 + "\n")
    
    # Extract features
    features_list = []
    failed_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting HuBERT embeddings"):
        # Get audio file path
        if 'filepath' in row:
            audio_path = Path(audio_dir) / row['filepath']
        else:
            continue
        
        if not audio_path.exists():
            failed_files.append(str(audio_path))
            continue
        
        # Extract features
        result = extractor.extract_from_file(
            audio_path, 
            extract_layer=extract_layer,
            pooling=pooling
        )
        
        if result is not None:
            # Add metadata
            result['speaker_id'] = row.get('speaker_id', 'unknown')
            result['native_language'] = row.get('native_language', 'unknown')
            result['age_group'] = row.get('age_group', 'unknown')
            result['utterance_type'] = row.get('utterance_type', 'unknown')
            
            features_list.append(result)
        else:
            failed_files.append(str(audio_path))
    
    print("\n" + "=" * 80)
    print("Extraction Complete")
    print("=" * 80)
    print(f"✓ Successfully extracted: {len(features_list)} files")
    print(f"✗ Failed: {len(failed_files)} files")
    
    if len(features_list) == 0:
        print("✗ No features extracted!")
        return
    
    # Save features
    if extract_layer is not None:
        output_file = output_dir / f"hubert_layer{extract_layer}_{pooling}.pkl"
    else:
        output_file = output_dir / f"hubert_all_layers_{pooling}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(features_list, f)
    
    print(f"✓ Features saved to: {output_file}")
    
    # Save feature info
    info = {
        'n_samples': len(features_list),
        'model_name': model_name,
        'n_layers': extractor.n_layers,
        'hidden_size': extractor.hidden_size,
        'extract_layer': extract_layer,
        'pooling': pooling
    }
    
    info_file = output_dir / f"hubert_info_{extract_layer if extract_layer is not None else 'all'}_{pooling}.pkl"
    with open(info_file, 'wb') as f:
        pickle.dump(info, f)
    
    print(f"✓ Feature info saved to: {info_file}")
    
    if failed_files:
        failed_file_path = output_dir / 'failed_files_hubert.txt'
        with open(failed_file_path, 'w') as f:
            f.write('\n'.join(failed_files))
        print(f"✗ Failed files list saved to: {failed_file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract HuBERT embeddings from audio dataset"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted features"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/hubert-base-ls960",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--extract_layer",
        type=int,
        default=None,
        help="Specific layer to extract (None = all layers)"
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "none"],
        help="Pooling method for temporal dimension"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run model on"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("HuBERT Feature Extraction")
    print("=" * 80)
    
    pooling = None if args.pooling == "none" else args.pooling
    
    extract_features_from_dataset(
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        extract_layer=args.extract_layer,
        pooling=pooling,
        model_name=args.model_name
    )
    
    print("\n✅ Feature extraction completed!")


if __name__ == "__main__":
    main()
