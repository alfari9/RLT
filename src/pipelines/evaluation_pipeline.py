"""
Model Evaluation and Metrics Tracking
Comprehensive model evaluation with multiple metrics
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, mean_squared_error,
    mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates model performance with comprehensive metrics"""
    
    def __init__(self, task_type: str = 'classification'):
        self.task_type = task_type
        self.metrics = {}
        self.plots_dir = Path('evaluation_plots')
        self.plots_dir.mkdir(exist_ok=True)
    
    def evaluate_classification(self, y_true, y_pred, y_prob=None) -> Dict[str, float]:
        """Evaluate classification model"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC if probabilities are provided
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        self.metrics = metrics
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def evaluate_regression(self, y_true, y_pred) -> Dict[str, float]:
        """Evaluate regression model"""
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
        
        self.metrics = metrics
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, save_path: str = None):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path is None:
            save_path = self.plots_dir / f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
        
        return cm
    
    def plot_roc_curve(self, y_true, y_prob, save_path: str = None):
        """Plot ROC curve"""
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = self.plots_dir / f'roc_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {save_path}")
    
    def plot_feature_importance(self, model, feature_names: List[str], top_n: int = 20, save_path: str = None):
        """Plot feature importance"""
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importances')
            plt.gca().invert_yaxis()
            
            if save_path is None:
                save_path = self.plots_dir / f'feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            logger.warning("Model does not have feature_importances_ attribute")
    
    def generate_report(self, y_true, y_pred, labels=None) -> str:
        """Generate detailed classification report"""
        
        report = classification_report(y_true, y_pred, target_names=labels)
        logger.info(f"\nClassification Report:\n{report}")
        
        return report
    
    def save_metrics(self, filepath: str = None):
        """Save metrics to JSON file"""
        
        if filepath is None:
            filepath = f'metrics/metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")


class MetricsTracker:
    """Tracks metrics across multiple experiments"""
    
    def __init__(self, tracking_file: str = 'metrics/metrics_history.json'):
        self.tracking_file = tracking_file
        self.history = self.load_history()
    
    def load_history(self) -> List[Dict]:
        """Load metrics history"""
        if Path(self.tracking_file).exists():
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return []
    
    def add_experiment(self, experiment_name: str, metrics: Dict, metadata: Dict = None):
        """Add experiment results to history"""
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        self.history.append(entry)
        self.save_history()
        
        logger.info(f"Experiment '{experiment_name}' added to tracking history")
    
    def save_history(self):
        """Save metrics history to file"""
        Path(self.tracking_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.tracking_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_best_experiment(self, metric: str = 'accuracy') -> Dict:
        """Get experiment with best metric value"""
        
        if not self.history:
            return None
        
        best = max(self.history, key=lambda x: x['metrics'].get(metric, 0))
        logger.info(f"Best experiment by {metric}: {best['experiment_name']} = {best['metrics'].get(metric)}")
        
        return best
    
    def plot_metrics_over_time(self, metric: str = 'accuracy', save_path: str = None):
        """Plot metric evolution over experiments"""
        
        timestamps = [entry['timestamp'] for entry in self.history]
        values = [entry['metrics'].get(metric, 0) for entry in self.history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(values)), values, marker='o', linewidth=2)
        plt.xlabel('Experiment Number')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Over Experiments')
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = f'evaluation_plots/metrics_evolution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics evolution plot saved to {save_path}")


if __name__ == "__main__":
    logger.info("Model evaluation and metrics tracking initialized")
