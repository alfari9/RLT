"""
Experiment Tracking with MLflow and Weights & Biases
Centralized experiment management and comparison
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging
from typing import Dict, Any, List
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Manages ML experiments with MLflow"""
    
    def __init__(self, experiment_name: str, tracking_uri: str = './mlruns'):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient(tracking_uri)
        
        logger.info(f"Experiment tracker initialized: {experiment_name}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """Start a new MLflow run"""
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, filepath: str):
        """Log artifact file"""
        mlflow.log_artifact(filepath)
        logger.info(f"Logged artifact: {filepath}")
    
    def log_model(self, model, model_name: str):
        """Log ML model"""
        mlflow.sklearn.log_model(model, model_name)
        logger.info(f"Logged model: {model_name}")
    
    def get_experiment_runs(self, max_results: int = 100) -> List[Dict]:
        """Get all runs from current experiment"""
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results
        )
        
        logger.info(f"Retrieved {len(runs)} runs")
        return runs
    
    def get_best_run(self, metric: str = 'test_accuracy', ascending: bool = False) -> Dict:
        """Get best run based on metric"""
        runs = self.get_experiment_runs()
        
        if not runs:
            logger.warning("No runs found")
            return None
        
        best_run = sorted(
            runs,
            key=lambda x: x.data.metrics.get(metric, 0),
            reverse=not ascending
        )[0]
        
        logger.info(f"Best run: {best_run.info.run_id} with {metric}={best_run.data.metrics.get(metric)}")
        
        return best_run
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs"""
        import pandas as pd
        
        comparison_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            data = {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                **run.data.metrics,
                **run.data.params
            }
            comparison_data.append(data)
        
        df = pd.DataFrame(comparison_data)
        logger.info(f"Compared {len(run_ids)} runs")
        
        return df


class ExperimentConfig:
    """Manages experiment configurations"""
    
    def __init__(self, config_dir: str = 'config/experiments'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, experiment_name: str, config: Dict[str, Any]):
        """Save experiment configuration"""
        config_file = self.config_dir / f'{experiment_name}.json'
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Experiment config saved: {config_file}")
    
    def load_config(self, experiment_name: str) -> Dict[str, Any]:
        """Load experiment configuration"""
        config_file = self.config_dir / f'{experiment_name}.json'
        
        if not config_file.exists():
            logger.warning(f"Config not found: {config_file}")
            return {}
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Experiment config loaded: {config_file}")
        return config
    
    def list_experiments(self) -> List[str]:
        """List all saved experiment configurations"""
        configs = [f.stem for f in self.config_dir.glob('*.json')]
        logger.info(f"Found {len(configs)} experiment configurations")
        return configs


class AutoLogging:
    """Automatic logging utilities"""
    
    @staticmethod
    def enable_autolog(framework: str = 'sklearn'):
        """Enable automatic logging for ML framework"""
        if framework == 'sklearn':
            mlflow.sklearn.autolog()
        
        logger.info(f"Auto-logging enabled for {framework}")
    
    @staticmethod
    def log_system_metrics():
        """Log system information"""
        import platform
        import psutil
        
        system_info = {
            'system': platform.system(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
        }
        
        mlflow.log_params(system_info)
        logger.info("System metrics logged")


if __name__ == "__main__":
    # Example usage
    tracker = ExperimentTracker('example_experiment')
    config_manager = ExperimentConfig()
    
    logger.info("Experiment tracking system initialized")
