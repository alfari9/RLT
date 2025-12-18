"""
Model Training Configuration
"""

# MLflow configuration
MLFLOW_CONFIG = {
    'tracking_uri': './mlruns',
    'experiment_name': 'ml_experiments',
    'artifact_location': './mlartifacts'
}

# Training configuration
TRAINING_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'n_jobs': -1
}

# Model hyperparameters
MODEL_PARAMS = {
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    },
    'LogisticRegression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
}

# Model registry settings
REGISTRY_CONFIG = {
    'registry_path': 'models/registry.json',
    'model_save_path': 'models/',
    'version_format': 'v{major}.{minor}.{patch}'
}
