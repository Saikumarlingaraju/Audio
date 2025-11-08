"""
Quick test script to check predictions on training samples
"""
import pandas as pd
import requests
from pathlib import Path
import json

def test_training_predictions():
    # Load metadata
    meta = pd.read_csv('data/raw/indian_accents/metadata_with_splits.csv')
    
    # Pick 2 samples from each state in training set
    print("="*80)
    print("🧪 TESTING PREDICTIONS ON TRAINING SAMPLES")
    print("="*80)
    
    results = []
    for state in meta['native_language'].unique():
        state_samples = meta[(meta['split'] == 'train') & (meta['native_language'] == state)].head(2)
        
        for _, row in state_samples.iterrows():
            filepath = row['filepath']
            true_label = row['native_language']
            
            # Test via debug endpoint
            url = f"http://127.0.0.1:5000/debug_predict?path=data/raw/indian_accents/{filepath}"
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    pred = response.json()
                    predicted = pred['accent'].lower().replace(' ', '_')
                    confidence = pred['confidence']
                    
                    correct = predicted == true_label
                    results.append({
                        'file': Path(filepath).name,
                        'true': true_label,
                        'predicted': predicted,
                        'confidence': confidence,
                        'correct': correct
                    })
                    
                    status = "✅" if correct else "❌"
                    print(f"\n{status} {Path(filepath).name}")
                    print(f"   True: {true_label}")
                    print(f"   Predicted: {predicted} ({confidence:.1f}%)")
                else:
                    print(f"\n❌ Error for {filepath}: {response.status_code}")
                    print(f"   Response: {response.text}")
            except Exception as e:
                print(f"\n❌ Exception for {filepath}: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    if results:
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = (correct / total) * 100
        
        print(f"Total tested: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        # Show errors
        errors = [r for r in results if not r['correct']]
        if errors:
            print(f"\n❌ MISCLASSIFIED ({len(errors)} files):")
            for err in errors:
                print(f"   {err['file']}: {err['true']} → {err['predicted']} ({err['confidence']:.1f}%)")
    else:
        print("No results - check if server is running!")
    
    print("="*80)
    return results

if __name__ == "__main__":
    print("\n⚠️  Make sure Flask server is running: python app\\app_robust.py\n")
    import time
    time.sleep(2)
    
    results = test_training_predictions()
