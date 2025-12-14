"""
DSO1: Feature Selection - Data Preparation
- Train/test split
- Standard scaling
- Add synthetic noise features
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def add_synthetic_features(X_train, X_test, n_synthetic=50, random_state=42):
    np.random.seed(random_state)
    synthetic_train = pd.DataFrame(
        np.random.randn(X_train.shape[0], n_synthetic),
        columns=[f"synthetic_{i}" for i in range(n_synthetic)],
        index=X_train.index
    )
    synthetic_test = pd.DataFrame(
        np.random.randn(X_test.shape[0], n_synthetic),
        columns=[f"synthetic_{i}" for i in range(n_synthetic)],
        index=X_test.index
    )
    X_train_aug = pd.concat([X_train, synthetic_train], axis=1)
    X_test_aug = pd.concat([X_test, synthetic_test], axis=1)
    return X_train_aug, X_test_aug

def run(dataset, test_size=0.3, random_state=42, n_synthetic=50):
    X = dataset["X"]
    y = dataset["y"]

    # Infer task
    task = dataset["task"]

    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Add synthetic noise features
    X_train_aug, X_test_aug = add_synthetic_features(X_train_scaled, X_test_scaled, n_synthetic=n_synthetic, random_state=random_state)

    return {
        "X_train": X_train_aug,
        "X_test": X_test_aug,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "feature_names": X_train_aug.columns.tolist(),
        "task": task
    }
