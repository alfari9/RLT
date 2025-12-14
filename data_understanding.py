"""
data_understanding.py
Unified dataset loader + profiler for RLT project.
Used by ALL DSOs.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# ======================================================
# Cache
# ======================================================
DATA_DIR = r"C:\Users\marie\OneDrive\Bureau\RLT\uci-datasets"

# ======================================================
# Dataset registry
# ======================================================
DATASETS_INFO = {
    "Breast Cancer": {"file": "breast_cancer_wisconsin_diagnostic.csv", "target": "target", "sep": ","},
    "Boston Housing": {"file": "boston_housing.csv", "target": -1, "sep": ","},
    "Parkinson": {"file": "parkinsons.csv", "target": "status", "sep": ","},
    "Sonar": {"file": "sonar.csv", "target": -1, "sep": ","},
    "White Wine": {"file": "white_wine_quality.csv", "target": "quality", "sep": ";"},
    "Red Wine": {"file": "red_wine_quality.csv", "target": "quality", "sep": ";"},
    "Parkinson Oxford": {"file": "parkinson_oxford.csv", "target": "total_UPDRS", "sep": ","},
    "Ozone": {"file": "ozone.csv", "target": -1, "sep": ","},
    "Concrete": {"file": "concrete_compressive_strength.csv", "target": "strength", "sep": ","},
    "Auto MPG": {"file": "auto_mpg.csv", "target": -1, "sep": ","}
}

# ======================================================
# Utilities
# ======================================================
def profile_data(df):
    profile = {
        "n_samples": len(df),
        "n_features": df.shape[1],
        "missing_per_column": df.isnull().sum()
    }
    return profile

# ======================================================
# Load CSV safely
# ======================================================
def load_csv(path, sep=","):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, sep=sep)

# ======================================================
# Dataset loader
# ======================================================
def load_dataset(name):
    if name not in DATASETS_INFO:
        raise ValueError(f"Unknown dataset: {name}")

    cfg = DATASETS_INFO[name]
    path = os.path.join(DATA_DIR, cfg["file"])
    sep = cfg.get("sep", ",")

    # Special case: Breast Cancer (sklearn)
    if name == "Breast Cancer":
        data = load_breast_cancer(as_frame=True)
        X = data.data
        y = data.target
        task = "classification"
    else:
        df = load_csv(path, sep=sep)

        # Drop non-numeric columns (dates, strings)
        df_numeric = df.select_dtypes(include=[np.number])

        # Determine target column
        target = cfg["target"]
        if target == -1:
            y = df_numeric.iloc[:, -1]
            X = df_numeric.iloc[:, :-1]
        else:
            if target not in df_numeric.columns:
                raise ValueError(f"Target column '{target}' not found in dataset '{name}'")
            y = df_numeric[target]
            X = df_numeric.drop(target, axis=1)

        # Task inference
        task = "classification" if pd.api.types.is_integer_dtype(y) and y.nunique() <= 20 else "regression"

    # Return dataset
    dataset = {
        "name": name,
        "X": X,
        "y": y,
        "task": task,
        "profile": profile_data(X)
    }

    # Print summary
    print(f"\n=== Dataset: {name} ===")
    print(f"Rows: {len(X)}, Columns: {X.shape[1]}")
    print(f"Task: {task}")
    print("\nColumn types:")
    print(X.dtypes.value_counts())
    print("\nMissing values per column:")
    print(X.isnull().sum())
    if task == "classification":
        print("\nTarget variable statistics:")
        print(y.value_counts())
    else:
        print("\nTarget variable statistics:")
        print(y.describe())

    return dataset
