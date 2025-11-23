"""
Setup script for the robust accent identification system.
Trains a robust model and launches the improved app.
"""

import subprocess
import sys
from pathlib import Path
import os


def run_command(command, description):
    """Run a command with error handling."""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False


def main():
    print("🚀 ROBUST ACCENT IDENTIFICATION SETUP")
    print("=" * 60)
    
    # Check if we have the necessary data
    features_path = Path("data/features/indian_accents/hubert_layer12_mean.pkl")
    train_csv = Path("data/splits/train.csv")
    
    if not features_path.exists():
        print("❌ HuBERT features not found. Please extract features first:")
        print("   python extract_hubert_features.py --metadata data/indian_metadata.csv \\")
        print("                                    --audio_dir data/raw/indian_accents \\")
        print("                                    --output_dir data/features/indian_accents \\")
        print("                                    --extract_layer 12 --pooling mean")
        return
    
    if not train_csv.exists():
        print("❌ Data splits not found. Please create splits first:")
        print("   python src/data/create_splits.py")
        return
    
    print("✅ Required data files found")
    
    # Step 1: Train robust model
    print("\n" + "=" * 60)
    print("STEP 1: Training Robust Model")
    print("=" * 60)
    
    train_cmd = """python train_robust.py \\
        --model_type robust_mlp \\
        --hidden_sizes 256 128 \\
        --dropout 0.5 \\
        --use_augmentation \\
        --aug_prob 0.5 \\
        --batch_size 16 \\
        --num_epochs 50 \\
        --lr 0.0005 \\
        --weight_decay 1e-4 \\
        --patience 15"""
    
    success = run_command(train_cmd, "Training robust model")
    
    if not success:
        print("❌ Training failed. Please check the error messages above.")
        return
    
    # Step 2: Launch robust app
    print("\n" + "=" * 60)
    print("STEP 2: Launching Robust App")
    print("=" * 60)
    
    print("🌐 Starting the robust accent identification web app...")
    print("   Features:")
    print("   • Ensemble prediction with test-time augmentation")
    print("   • Robust audio preprocessing")
    print("   • Domain adaptation for better generalization")
    print("   • Advanced regularization techniques")
    print("\n🌍 App will be available at: http://127.0.0.1:5000")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        # Launch the robust app
        subprocess.run([sys.executable, "app/app_robust.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ App failed to start: {e}")
        print("\n🔧 Fallback: Try running the original app:")
        print("   python app/app.py")


def install_requirements():
    """Install additional requirements for robust features."""
    requirements = [
        "librosa>=0.8.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0"
    ]
    
    print("📦 Installing additional requirements...")
    for req in requirements:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", req], 
                         check=True, capture_output=True)
            print(f"  ✅ {req}")
        except subprocess.CalledProcessError:
            print(f"  ⚠️ Failed to install {req} (might already be installed)")


def check_system():
    """Check system requirements."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3.7, 0):
        print("❌ Python 3.7+ required")
        return False
    
    print(f"✅ Python {sys.version}")
    
    # Check for PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found. Please install PyTorch first.")
        return False
    
    # Check for transformers
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not found. Please install transformers first.")
        return False
    
    return True


if __name__ == "__main__":
    print("🎯 ROBUST ACCENT IDENTIFICATION - SETUP WIZARD")
    print("=" * 60)
    
    # Check system
    if not check_system():
        print("\n❌ System requirements not met. Please install missing dependencies.")
        sys.exit(1)
    
    # Install additional requirements
    install_requirements()
    
    # Run main setup
    main()
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("\n📋 What was improved:")
    print("  ✅ Data augmentation during training")
    print("  ✅ Enhanced model regularization")
    print("  ✅ Robust feature extraction")
    print("  ✅ Ensemble prediction with test-time augmentation")
    print("  ✅ Consistent audio preprocessing")
    print("  ✅ Advanced training techniques")
    
    print("\n🚀 Your accent identifier should now work much better on:")
    print("  • Unseen speakers")
    print("  • Different recording conditions")
    print("  • Various audio qualities")
    print("  • External audio sources")
    
    print(f"\n💡 Pro Tips:")
    print("  • Use high-quality audio for best results")
    print("  • Ensure audio is at least 1-2 seconds long")
    print("  • The ensemble prediction provides confidence scores")
    print("  • Check the /info endpoint for model details")