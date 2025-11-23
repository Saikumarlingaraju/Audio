"""Quick test of feature extraction"""
import sys
import numpy as np
from src.features.hubert_extractor import HuBERTExtractor

print("Initializing extractor...")
extractor = HuBERTExtractor()

test_file = 'data/raw/indian_accents/tamil/Tamil_speaker (1).wav'
print(f"\nTesting with: {test_file}")

print("Extracting features...")
result = extractor.extract_from_file(test_file, extract_layer=12, pooling='mean')

print(f"\nResult type: {type(result)}")

if result is None:
    print("ERROR: Result is None!")
elif isinstance(result, dict):
    print(f"Result is dict with keys: {list(result.keys())}")
    emb = result.get('embeddings', None)
    if emb is not None:
        print(f"Embeddings type: {type(emb)}")
        if isinstance(emb, dict):
            print(f"Embeddings is ALSO a dict with keys: {list(emb.keys())}")
            # Try to get the actual array from the nested dict
            if 'layer_12' in emb:
                emb_array = np.array(emb['layer_12'], dtype=np.float32)
            elif 12 in emb:
                emb_array = np.array(emb[12], dtype=np.float32)
            else:
                print(f"Keys in embeddings dict: {emb.keys()}")
                first_key = list(emb.keys())[0]
                print(f"Using first key: {first_key}")
                emb_array = np.array(emb[first_key], dtype=np.float32)
        else:
            emb_array = np.array(emb, dtype=np.float32)
        print(f"Embeddings shape: {emb_array.shape}")
        print(f"Embeddings dtype: {emb_array.dtype}")
        print(f"Embeddings range: [{emb_array.min():.4f}, {emb_array.max():.4f}]")
    else:
        print("ERROR: No embeddings in result dict!")
else:
    print(f"Result is {type(result)}")
    emb_array = np.array(result, dtype=np.float32)
    print(f"Shape: {emb_array.shape}")
    print(f"Dtype: {emb_array.dtype}")

print("\n✅ Test complete!")
