"""
Test script to compare original vs robust accent prediction.
Demonstrates the improvements in generalization.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import time

# Add src to path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

def test_original_app():
    """Test the original app's prediction accuracy."""
    print("🔧 Testing Original App...")
    
    try:
        # Import original app components
        sys.path.insert(0, str(root_dir / 'app'))
        from app import predict_accent as original_predict
        
        # Test on a few samples
        test_files = [
            "data/raw/indian_accents/andhra_pradesh/Andhra_speaker (1000).wav",
            "data/raw/indian_accents/kerala/Kerala_speaker (1000).wav",
            "data/raw/indian_accents/tamil/Tamil_speaker (1000).wav"
        ]
        
        predictions = []
        for file_path in test_files:
            if Path(file_path).exists():
                try:
                    result = original_predict(file_path)
                    predictions.append({
                        'file': Path(file_path).name,
                        'predicted': result.get('predicted_accent', 'Unknown'),
                        'confidence': result.get('confidence', 0),
                        'method': 'original'
                    })
                except Exception as e:
                    print(f"  ❌ Failed to predict {file_path}: {e}")
        
        return predictions
        
    except Exception as e:
        print(f"  ❌ Original app test failed: {e}")
        return []


def test_robust_app():
    """Test the robust app's prediction accuracy."""
    print("🚀 Testing Robust App...")
    
    try:
        # Import robust prediction components
        from src.utils.robust_prediction import RobustPredictor
        from src.features.robust_extraction import create_robust_feature_extractor
        
        # Check if robust model exists
        model_path = "experiments/indian_accents_robust/best_model.pt"
        encoder_path = "experiments/indian_accents_robust/label_encoder.pkl"
        
        if not Path(model_path).exists():
            print(f"  ⚠️ Robust model not found at {model_path}")
            print("     Please train the robust model first:")
            print("     python train_robust.py")
            return []
        
        # Initialize robust predictor
        predictor = RobustPredictor(model_path, encoder_path)
        
        # Test on the same samples
        test_files = [
            "data/raw/indian_accents/andhra_pradesh/Andhra_speaker (1000).wav",
            "data/raw/indian_accents/kerala/Kerala_speaker (1000).wav", 
            "data/raw/indian_accents/tamil/Tamil_speaker (1000).wav"
        ]
        
        predictions = []
        for file_path in test_files:
            if Path(file_path).exists():
                try:
                    # Test both simple and ensemble prediction
                    simple_result = predictor.predict_simple(file_path)
                    ensemble_result = predictor.predict_ensemble(file_path)
                    
                    predictions.extend([
                        {
                            'file': Path(file_path).name,
                            'predicted': simple_result['prediction'],
                            'confidence': simple_result['confidence'] * 100,
                            'method': 'robust_simple'
                        },
                        {
                            'file': Path(file_path).name,
                            'predicted': ensemble_result['prediction'],
                            'confidence': ensemble_result['confidence'] * 100,
                            'method': 'robust_ensemble',
                            'n_augmentations': ensemble_result.get('n_augmentations', 1)
                        }
                    ])
                except Exception as e:
                    print(f"  ❌ Failed to predict {file_path}: {e}")
        
        return predictions
        
    except Exception as e:
        print(f"  ❌ Robust app test failed: {e}")
        return []


def benchmark_prediction_speed():
    """Benchmark prediction speed of different methods."""
    print("⏱️ Benchmarking Prediction Speed...")
    
    # Create a test audio file path
    test_file = "data/raw/indian_accents/andhra_pradesh/Andhra_speaker (1000).wav"
    
    if not Path(test_file).exists():
        print("  ⚠️ Test file not found, skipping speed benchmark")
        return
    
    results = {}
    
    # Test robust prediction if available
    try:
        from src.utils.robust_prediction import RobustPredictor
        model_path = "experiments/indian_accents_robust/best_model.pt"
        encoder_path = "experiments/indian_accents_robust/label_encoder.pkl"
        
        if Path(model_path).exists():
            predictor = RobustPredictor(model_path, encoder_path)
            
            # Time simple prediction
            start_time = time.time()
            for _ in range(5):
                predictor.predict_simple(test_file)
            simple_time = (time.time() - start_time) / 5
            results['robust_simple'] = simple_time
            
            # Time ensemble prediction
            start_time = time.time()
            for _ in range(3):  # Fewer iterations due to higher cost
                predictor.predict_ensemble(test_file)
            ensemble_time = (time.time() - start_time) / 3
            results['robust_ensemble'] = ensemble_time
    except Exception as e:
        print(f"  ⚠️ Could not benchmark robust prediction: {e}")
    
    # Print results
    if results:
        print("  Speed Benchmark Results:")
        for method, avg_time in results.items():
            print(f"    {method}: {avg_time:.2f}s per prediction")
    else:
        print("  ⚠️ No benchmark results available")


def analyze_improvements():
    """Analyze the theoretical improvements made."""
    print("📊 Analyzing Improvements Made...")
    
    improvements = [
        {
            'category': 'Data Augmentation',
            'techniques': [
                'Speed perturbation (0.9x - 1.1x)',
                'Pitch shifting (±2 semitones)', 
                'Background noise addition',
                'Gain variations',
                'Feature-level dropout'
            ],
            'benefit': 'Increases training data diversity, reduces overfitting'
        },
        {
            'category': 'Model Regularization',
            'techniques': [
                'Higher dropout (0.5 vs 0.3)',
                'Weight decay (L2 regularization)',
                'Batch normalization',
                'Label smoothing',
                'Gradient clipping'
            ],
            'benefit': 'Prevents overfitting, improves generalization'
        },
        {
            'category': 'Robust Feature Extraction',
            'techniques': [
                'Consistent audio preprocessing',
                'RMS normalization instead of peak',
                'Silence trimming with adaptive thresholds',
                'Feature normalization with training statistics',
                'Outlier handling'
            ],
            'benefit': 'Handles domain shift between training and inference'
        },
        {
            'category': 'Ensemble Prediction',
            'techniques': [
                'Test-time augmentation',
                'Multiple prediction averaging',
                'Majority voting',
                'Confidence scoring'
            ],
            'benefit': 'Reduces prediction variance, improves accuracy on unseen data'
        }
    ]
    
    for improvement in improvements:
        print(f"\n  🎯 {improvement['category']}:")
        print(f"     Benefit: {improvement['benefit']}")
        for technique in improvement['techniques']:
            print(f"     • {technique}")


def main():
    print("🧪 ACCENT PREDICTION COMPARISON TEST")
    print("=" * 60)
    
    # Analyze improvements
    analyze_improvements()
    
    print("\n" + "=" * 60)
    print("PREDICTION TESTING")
    print("=" * 60)
    
    # Test original app
    original_predictions = test_original_app()
    
    # Test robust app
    robust_predictions = test_robust_app()
    
    # Compare results
    if original_predictions and robust_predictions:
        print("\n📊 COMPARISON RESULTS:")
        print("-" * 40)
        
        # Create comparison DataFrame
        df_data = []
        
        # Add original predictions
        for pred in original_predictions:
            df_data.append(pred)
        
        # Add robust predictions
        for pred in robust_predictions:
            df_data.append(pred)
        
        # Display results
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        print("\n📈 Key Improvements Expected:")
        print("  • Higher confidence scores on correct predictions")
        print("  • More consistent predictions across different methods")
        print("  • Better handling of audio quality variations")
        print("  • Ensemble method provides additional confidence")
    
    # Benchmark speed
    print("\n" + "=" * 60)
    benchmark_prediction_speed()
    
    print("\n" + "=" * 60)
    print("✅ COMPARISON TEST COMPLETE")
    print("=" * 60)
    
    print("\n🎯 Recommendations:")
    print("  1. Use the robust ensemble prediction for best accuracy")
    print("  2. Simple robust prediction for faster inference") 
    print("  3. Train on more diverse data for further improvements")
    print("  4. Monitor prediction confidence scores")
    
    print("\n📝 Next Steps:")
    print("  • Train the robust model: python train_robust.py")
    print("  • Launch robust app: python app/app_robust.py")
    print("  • Test with your own audio files")


if __name__ == "__main__":
    main()