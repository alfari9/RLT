"""
Model Training Pipeline with MLflow Integration
Trains ML models with automated logging and versioning
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and logs ML models with MLflow"""
    
    def __init__(self, experiment_name: str = "ml_experiments"):
        self.experiment_name = experiment_name
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
        # Initialize MLflow
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """Split and scale data"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Data prepared: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def train_model(self, model_name: str, model, X_train, X_test, y_train, y_test, params: Dict[str, Any] = None):
        """Train a single model with MLflow logging"""
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Train model
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Log metrics
            mlflow.log_metric("train_accuracy", train_score)
            mlflow.log_metric("test_accuracy", test_score)
            mlflow.log_metric("cv_accuracy_mean", cv_mean)
            mlflow.log_metric("cv_accuracy_std", cv_std)
            
            # Log model
            mlflow.sklearn.log_model(model, f"{model_name}_model")
            
            # Track best model
            if test_score > self.best_score:
                self.best_score = test_score
                self.best_model = model
                mlflow.set_tag("best_model", "true")
            
            logger.info(f"{model_name} - Train: {train_score:.4f}, Test: {test_score:.4f}, CV: {cv_mean:.4f}±{cv_std:.4f}")
            
            self.models[model_name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'cv_score': cv_mean
            }
            
            return model, test_score
    
    def train_multiple_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare performance"""
        
        models_config = {
            'RandomForest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                'params': {'n_estimators': 100, 'random_state': 42}
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'params': {'n_estimators': 100, 'random_state': 42}
            },
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {'max_iter': 1000, 'random_state': 42}
            }
        }
        
        results = {}
        
        for model_name, config in models_config.items():
            model, score = self.train_model(
                model_name,
                config['model'],
                X_train,
                X_test,
                y_train,
                y_test,
                config['params']
            )
            results[model_name] = score
        
        logger.info(f"\nBest model: {max(results, key=results.get)} with accuracy: {max(results.values()):.4f}")
        
        return results
    
    def save_best_model(self, filepath: str = "models/best_model.pkl"):
        """Save the best performing model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        logger.info(f"Best model saved to {filepath}")


class ModelRegistry:
    """Manages model versions and metadata"""
    
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = registry_path
        self.registry = self.load_registry()
    
    def load_registry(self) -> Dict:
        """Load model registry"""
        if Path(self.registry_path).exists():
            import json
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}
    
    def register_model(self, model_name: str, version: str, metrics: Dict, metadata: Dict = None):
        """Register a new model version"""
        import json
        
        if model_name not in self.registry:
            self.registry[model_name] = []
        
        model_entry = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        self.registry[model_name].append(model_entry)
        
        Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
        
        logger.info(f"Model registered: {model_name} v{version}")


if __name__ == "__main__":
    logger.info("Model training pipeline initialized")
