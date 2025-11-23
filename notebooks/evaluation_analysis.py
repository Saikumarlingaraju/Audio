"""
Comprehensive evaluation analysis with visualizations and detailed metrics
"""
import sys
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_model_results(experiments_dir='experiments'):
    """Load results from all trained models"""
    exp_path = Path(experiments_dir)
    results = {}
    
    for model_dir in exp_path.iterdir():
        if model_dir.is_dir() and (model_dir / 'results.json').exists():
            with open(model_dir / 'results.json', 'r') as f:
                results[model_dir.name] = json.load(f)
    
    return results

def create_comparison_plots(results, output_dir='experiments/analysis'):
    """Create comprehensive comparison plots"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter only model comparison results (exclude level comparisons)
    model_results = {k: v for k, v in results.items() 
                    if k.startswith('indian_accents_') and 'level' not in k}
    
    if not model_results:
        print("⚠️  No model results found for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Test accuracy bar chart
    models = list(model_results.keys())
    model_labels = [m.replace('indian_accents_', '').upper() for m in models]
    test_accs = [model_results[m]['test_acc'] * 100 for m in models]
    val_accs = [model_results[m]['best_val_acc'] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, val_accs, width, label='Validation', alpha=0.8, color='#3498db')
    bars2 = axes[0, 0].bar(x + width/2, test_accs, width, label='Test', alpha=0.8, color='#e74c3c')
    
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Model Comparison: Validation vs Test Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_labels)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: F1-Score comparison
    test_f1s = [model_results[m].get('test_f1', 0) * 100 for m in models]
    
    bars = axes[0, 1].bar(model_labels, test_f1s, alpha=0.8, color='#2ecc71')
    axes[0, 1].set_ylabel('F1-Score (%)', fontsize=12)
    axes[0, 1].set_title('Model Comparison: Test F1-Scores', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Training curves
    for model_name, model_results_item in model_results.items():
        if 'history' in model_results_item:
            epochs = range(1, len(model_results_item['history']['val_acc']) + 1)
            label = model_name.replace('indian_accents_', '').upper()
            axes[1, 0].plot(epochs, [acc * 100 for acc in model_results_item['history']['val_acc']], 
                          label=label, marker='o', markersize=3, linewidth=2)
    
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1, 0].set_title('Training Curves Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Overfitting analysis (train vs test gap)
    train_accs = []
    for model_name in models:
        if 'history' in model_results[model_name]:
            train_accs.append(model_results[model_name]['history']['train_acc'][-1] * 100)
        else:
            train_accs.append(0)
    
    test_accs_plot = [model_results[m]['test_acc'] * 100 for m in models]
    gaps = [train - test for train, test in zip(train_accs, test_accs_plot)]
    
    colors = ['#e74c3c' if gap > 10 else '#2ecc71' for gap in gaps]
    bars = axes[1, 1].bar(model_labels, gaps, alpha=0.8, color=colors)
    axes[1, 1].axhline(y=10, color='orange', linestyle='--', label='10% threshold')
    axes[1, 1].set_ylabel('Train-Test Gap (%)', fontsize=12)
    axes[1, 1].set_title('Overfitting Analysis (Train - Test Accuracy)', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'model_comparison_plots.png', dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plots saved to {output_path / 'model_comparison_plots.png'}")
    
    plt.close()

def analyze_per_class_performance(experiments_dir='experiments', output_dir='experiments/analysis'):
    """Analyze per-class performance across models"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    exp_path = Path(experiments_dir)
    per_class_data = []
    
    for model_dir in exp_path.iterdir():
        if not model_dir.is_dir() or not (model_dir / 'results.json').exists():
            continue
        
        if 'level' in model_dir.name:
            continue
        
        with open(model_dir / 'results.json', 'r') as f:
            results = json.load(f)
        
        if 'classification_report' in results:
            report = results['classification_report']
            model_type = model_dir.name.replace('indian_accents_', '').upper()
            
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'f1-score' in metrics:
                    per_class_data.append({
                        'Model': model_type,
                        'Class': class_name,
                        'Precision': metrics['precision'] * 100,
                        'Recall': metrics['recall'] * 100,
                        'F1-Score': metrics['f1-score'] * 100,
                        'Support': metrics['support']
                    })
    
    if not per_class_data:
        print("⚠️  No per-class data available")
        return
    
    df = pd.DataFrame(per_class_data)
    
    # Create heatmap of F1-scores
    pivot_table = df.pivot(index='Class', columns='Model', values='F1-Score')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'F1-Score (%)'})
    plt.title('Per-Class F1-Score Comparison Across Models', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accent Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'per_class_f1_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Per-class heatmap saved to {output_path / 'per_class_f1_heatmap.png'}")
    plt.close()
    
    # Save per-class data
    df.to_csv(output_path / 'per_class_performance.csv', index=False)
    print(f"✓ Per-class data saved to {output_path / 'per_class_performance.csv'}")
    
    return df

def generate_evaluation_report(results, output_file='experiments/evaluation_report.md'):
    """Generate comprehensive markdown evaluation report"""
    report = []
    report.append("# Model Evaluation Report\n\n")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Filter model results
    model_results = {k: v for k, v in results.items() 
                    if k.startswith('indian_accents_') and 'level' not in k}
    
    if not model_results:
        report.append("⚠️  No model results available for evaluation.\n")
        with open(output_file, 'w') as f:
            f.writelines(report)
        return
    
    report.append("## Model Comparison Summary\n\n")
    
    # Create comparison table
    report.append("| Model | Val Accuracy | Test Accuracy | Test F1-Score | Train-Test Gap |\n")
    report.append("|-------|--------------|---------------|---------------|----------------|\n")
    
    for model_name, model_result in model_results.items():
        model_type = model_name.replace('indian_accents_', '').upper()
        val_acc = model_result['best_val_acc'] * 100
        test_acc = model_result['test_acc'] * 100
        test_f1 = model_result.get('test_f1', 0) * 100
        
        train_acc = 0
        if 'history' in model_result and len(model_result['history']['train_acc']) > 0:
            train_acc = model_result['history']['train_acc'][-1] * 100
        
        gap = train_acc - test_acc
        
        report.append(f"| {model_type} | {val_acc:.2f}% | {test_acc:.2f}% | {test_f1:.2f}% | {gap:.2f}% |\n")
    
    report.append("\n## Key Findings\n\n")
    
    # Find best model
    best_model = max(model_results.items(), key=lambda x: x[1]['test_acc'])
    best_model_name = best_model[0].replace('indian_accents_', '').upper()
    best_acc = best_model[1]['test_acc'] * 100
    
    report.append(f"- **Best Model:** {best_model_name} (Test Accuracy: {best_acc:.2f}%)\n")
    
    # Statistical analysis
    test_accs = [r['test_acc'] * 100 for r in model_results.values()]
    report.append(f"- **Average Test Accuracy:** {np.mean(test_accs):.2f}%\n")
    report.append(f"- **Standard Deviation:** {np.std(test_accs):.2f}%\n")
    report.append(f"- **Accuracy Range:** {np.min(test_accs):.2f}% - {np.max(test_accs):.2f}%\n")
    
    # Model rankings
    report.append("\n### Model Rankings (by Test Accuracy)\n\n")
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
    for i, (model_name, model_result) in enumerate(sorted_models, 1):
        model_type = model_name.replace('indian_accents_', '').upper()
        test_acc = model_result['test_acc'] * 100
        report.append(f"{i}. **{model_type}**: {test_acc:.2f}%\n")
    
    report.append("\n## Analysis\n\n")
    
    # Overfitting analysis
    report.append("### Overfitting Analysis\n\n")
    for model_name, model_result in model_results.items():
        if 'history' in model_result and len(model_result['history']['train_acc']) > 0:
            model_type = model_name.replace('indian_accents_', '').upper()
            train_acc = model_result['history']['train_acc'][-1] * 100
            test_acc = model_result['test_acc'] * 100
            gap = train_acc - test_acc
            
            if gap > 15:
                status = "⚠️ High overfitting"
            elif gap > 10:
                status = "⚠️ Moderate overfitting"
            else:
                status = "✓ Good generalization"
            
            report.append(f"- **{model_type}**: Train-Test gap = {gap:.2f}% {status}\n")
    
    report.append("\n## Recommendations\n\n")
    report.append("Based on the evaluation:\n\n")
    report.append(f"1. **Deploy {best_model_name}** for production as it achieves the highest test accuracy\n")
    report.append("2. Consider **ensemble methods** combining top 3 models for improved robustness\n")
    report.append("3. Apply **data augmentation** to reduce overfitting in models with large train-test gaps\n")
    report.append("4. Collect **more diverse training data** to improve generalization to external audio\n")
    report.append("5. Implement **ensemble inference** with test-time augmentation for external audio\n")
    
    report.append("\n## Visualizations\n\n")
    report.append("See generated plots:\n")
    report.append("- `experiments/analysis/model_comparison_plots.png`\n")
    report.append("- `experiments/analysis/per_class_f1_heatmap.png`\n")
    
    # Write report
    with open(output_file, 'w') as f:
        f.writelines(report)
    
    print(f"✓ Evaluation report saved to {output_file}")

def main():
    """Main evaluation analysis"""
    print(f"\n{'='*80}")
    print("Comprehensive Evaluation Analysis")
    print(f"{'='*80}\n")
    
    # Load all model results
    results = load_model_results()
    
    if not results:
        print("❌ No model results found. Please train models first using:")
        print("   python scripts/train_all_models.py")
        return
    
    print(f"Found {len(results)} experiment results:")
    for model_name in results.keys():
        print(f"  - {model_name}")
    
    # Create visualizations
    print("\nGenerating comparison plots...")
    create_comparison_plots(results)
    
    # Per-class analysis
    print("\nAnalyzing per-class performance...")
    analyze_per_class_performance()
    
    # Generate report
    print("\nGenerating evaluation report...")
    generate_evaluation_report(results)
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}\n")
    print("Check the following files:")
    print("  - experiments/analysis/model_comparison_plots.png")
    print("  - experiments/analysis/per_class_f1_heatmap.png")
    print("  - experiments/analysis/per_class_performance.csv")
    print("  - experiments/evaluation_report.md")

if __name__ == '__main__':
    main()
