"""
DSO1: Feature Selection - Evaluation
- Compute Recall@50 (top features vs ground truth)
- Compute VI sparsity (proportion of low-importance features)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

def recall_at_k(vi, informative_features, k=50):
    """
    Recall@k: proportion of true informative features in top-k VI
    vi: pd.Series of feature importances (index=feature names)
    informative_features: list of ground-truth informative features
    """
    top_features = vi.head(k).index
    recall = len(set(top_features) & set(informative_features)) / len(informative_features)
    return recall

def sparsity(vi, threshold=1e-3):
    """
    VI sparsity: fraction of features with importance below threshold
    """
    return (vi < threshold).sum() / len(vi)

def run(prepared_data, model, informative_features=None):
    X_test = prepared_data["X_test"]
    y_test = prepared_data["y_test"]
    X_train = prepared_data["X_train"]
    y_train = prepared_data["y_train"]
    task = prepared_data["task"]

    # Predictions
    y_pred = model.predict(X_test)
    if task == "classification":
        metric = accuracy_score(y_test, y_pred)
    else:
        metric = r2_score(y_test, y_pred)

    # Feature importance
    vi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    # Compute Recall@50 if ground truth informative features are provided
    recall50 = None
    if informative_features is not None:
        recall50 = recall_at_k(vi, informative_features, k=50)

    # Compute VI sparsity
    vi_sparsity = sparsity(vi)

    results = {
        "metric": metric,
        "variable_importance": vi,
        "recall_at_50": recall50,
        "vi_sparsity": vi_sparsity
    }
    return results
