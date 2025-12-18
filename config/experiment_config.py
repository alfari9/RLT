# MLflow Experiment Tracking Configuration

# Base experiment configuration
EXPERIMENT_BASE_CONFIG = {
    "experiment_name": "ml_model_experiments",
    "tracking_uri": "./mlruns",
    "artifact_location": "./mlartifacts",
    "tags": {
        "project": "MLOps-RLT",
        "team": "Data Science",
        "environment": "development"
    }
}

# Experiment templates for different model types
EXPERIMENT_TEMPLATES = {
    "classification": {
        "metrics_to_track": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ],
        "artifacts_to_log": [
            "confusion_matrix.png",
            "roc_curve.png",
            "feature_importance.png"
        ]
    },
    "regression": {
        "metrics_to_track": [
            "mse",
            "rmse",
            "mae",
            "r2_score"
        ],
        "artifacts_to_log": [
            "prediction_plot.png",
            "residuals_plot.png"
        ]
    }
}

# Auto-logging configuration
AUTOLOG_CONFIG = {
    "log_models": True,
    "log_input_examples": True,
    "log_model_signatures": True,
    "disable": False,
    "exclusive": False,
    "disable_for_unsupported_versions": False
}

# Run tagging standards
RUN_TAG_STANDARDS = {
    "model_type": ["RandomForest", "GradientBoosting", "LogisticRegression", "SVM", "NeuralNetwork"],
    "data_version": "v1.0",
    "preprocessing": ["standard", "minmax", "robust"],
    "feature_engineering": ["enabled", "disabled"]
}
