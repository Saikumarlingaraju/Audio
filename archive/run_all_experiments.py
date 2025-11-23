"""
Master script to run all advanced experiments and generate complete evaluation
"""
import sys
from pathlib import Path
import subprocess
import time

root = Path(__file__).parent
sys.path.insert(0, str(root))

def run_experiment(script_path, description):
    """Run a Python script and report results"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(root),
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False

def main():
    """Run all experiments in sequence"""
    print(f"\n{'#'*80}")
    print("# COMPREHENSIVE EVALUATION SUITE")
    print(f"{'#'*80}\n")
    print("This will run all advanced experiments:")
    print("1. Train all model architectures (MLP, CNN, BiLSTM, Transformer)")
    print("2. Create word-level dataset from sentences")
    print("3. Compare word-level vs sentence-level performance")
    print("4. Generate comprehensive evaluation analysis")
    print("5. Test robust prediction with ensemble inference\n")
    
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    experiments = [
        {
            'script': 'scripts/train_all_models.py',
            'description': 'Train All Model Architectures',
            'required': True
        },
        {
            'script': 'scripts/create_word_level_dataset.py',
            'description': 'Create Word-Level Dataset',
            'required': False
        },
        {
            'script': 'scripts/compare_word_sentence_level.py',
            'description': 'Compare Word vs Sentence Level',
            'required': False
        },
        {
            'script': 'notebooks/evaluation_analysis.py',
            'description': 'Comprehensive Evaluation Analysis',
            'required': True
        },
        {
            'script': 'src/utils/robust_prediction.py',
            'description': 'Test Robust Prediction',
            'required': False
        }
    ]
    
    results = {}
    total_start = time.time()
    
    for exp in experiments:
        script_path = root / exp['script']
        
        if not script_path.exists():
            print(f"\n⚠️  Script not found: {script_path}")
            if exp['required']:
                print("   This is a required experiment. Aborting.")
                break
            else:
                print("   Skipping optional experiment.")
                continue
        
        success = run_experiment(script_path, exp['description'])
        results[exp['description']] = success
        
        if not success and exp['required']:
            print(f"\n❌ Required experiment '{exp['description']}' failed. Aborting.")
            break
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print(f"\n{'#'*80}")
    print("# EVALUATION SUITE SUMMARY")
    print(f"{'#'*80}\n")
    print(f"Total time: {total_elapsed/60:.1f} minutes\n")
    
    print(f"{'Experiment':<45} {'Status'}")
    print("-" * 55)
    for exp_name, success in results.items():
        status = "✓ Success" if success else "❌ Failed"
        print(f"{exp_name:<45} {status}")
    
    # Success rate
    total = len(results)
    successes = sum(results.values())
    print(f"\n{successes}/{total} experiments completed successfully")
    
    # Output locations
    print(f"\n{'='*80}")
    print("Output Locations:")
    print(f"{'='*80}\n")
    print("Model Checkpoints:")
    print("  - experiments/indian_accents_mlp/")
    print("  - experiments/indian_accents_cnn/")
    print("  - experiments/indian_accents_bilstm/")
    print("  - experiments/indian_accents_transformer/")
    print("\nAnalysis & Visualizations:")
    print("  - experiments/analysis/model_comparison_plots.png")
    print("  - experiments/analysis/per_class_f1_heatmap.png")
    print("  - experiments/analysis/per_class_performance.csv")
    print("  - experiments/evaluation_report.md")
    print("  - experiments/word_vs_sentence_comparison.png")
    print("  - experiments/model_comparison.json")
    print("\nWord-Level Data:")
    print("  - data/word_level/word_level_metadata.csv")
    print("  - data/features/word_level/hubert_layer12_mean.pkl")
    
    print(f"\n{'#'*80}")
    print("# NEXT STEPS")
    print(f"{'#'*80}\n")
    print("1. Review evaluation_report.md for detailed analysis")
    print("2. Check visualization plots in experiments/analysis/")
    print("3. Test the web app with ensemble inference:")
    print("   python app/app_robust.py")
    print("4. Compare results and select best model for deployment")
    
    if all(results.values()):
        print("\n🎉 All experiments completed successfully!")
    else:
        print("\n⚠️  Some experiments failed. Check logs above for details.")

if __name__ == '__main__':
    main()
