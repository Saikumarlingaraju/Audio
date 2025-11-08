"""
Evaluation metrics for NLI classification.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MetricsCalculator:
    """Calculate and store classification metrics."""
    
    def __init__(self, class_names):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names (language names)
        """
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.all_predictions = []
        self.all_targets = []
        self.all_probs = []
    
    def update(self, predictions, targets, probs=None):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Predicted class indices
            targets: True class indices
            probs: Class probabilities (optional)
        """
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        if probs is not None and torch.is_tensor(probs):
            probs = probs.cpu().numpy()
        
        self.all_predictions.extend(predictions)
        self.all_targets.extend(targets)
        if probs is not None:
            self.all_probs.extend(probs)
    
    def compute(self):
        """
        Compute all metrics.
        
        Returns:
            dict: Dictionary of metrics
        """
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.class_names))
        )
        
        # Macro/micro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro'
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names)))
        
        metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'per_class': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            },
            'confusion_matrix': cm
        }
        
        return metrics
    
    def get_classification_report(self):
        """Get detailed classification report."""
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        return report
    
    def plot_confusion_matrix(self, save_path=None, normalize=False):
        """
        Plot confusion matrix.
        
        Args:
            save_path: Path to save figure
            normalize: Whether to normalize by row (true labels)
        """
        metrics = self.compute()
        cm = metrics['confusion_matrix']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), 
                 fontsize=14)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to: {save_path}")
        
        plt.close()
    
    def plot_per_class_metrics(self, save_path=None):
        """
        Plot per-class precision, recall, and F1 scores.
        
        Args:
            save_path: Path to save figure
        """
        metrics = self.compute()
        per_class = metrics['per_class']
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, per_class['precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, per_class['recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, per_class['f1'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Language', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Per-class metrics plot saved to: {save_path}")
        
        plt.close()


def print_metrics_summary(metrics):
    """Print formatted metrics summary."""
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    print(f"Overall Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy:       {metrics['balanced_accuracy']:.4f}")
    print(f"\nMacro Averages:")
    print(f"  Precision:             {metrics['precision_macro']:.4f}")
    print(f"  Recall:                {metrics['recall_macro']:.4f}")
    print(f"  F1-Score:              {metrics['f1_macro']:.4f}")
    print(f"\nMicro Averages:")
    print(f"  Precision:             {metrics['precision_micro']:.4f}")
    print(f"  Recall:                {metrics['recall_micro']:.4f}")
    print(f"  F1-Score:              {metrics['f1_micro']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Test metrics calculator
    class_names = ['Hindi', 'Tamil', 'Telugu', 'Malayalam', 'Kannada']
    calculator = MetricsCalculator(class_names)
    
    # Simulate predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 5, 100)
    y_pred = np.random.randint(0, 5, 100)
    
    calculator.update(y_pred, y_true)
    metrics = calculator.compute()
    print_metrics_summary(metrics)
    
    print(calculator.get_classification_report())
