"""
DSO3: High-dimensional prediction (p >> n)
Data preparation for both real datasets (10 UCI + synthetic noise) and simulated scenarios
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# Helpers
# -----------------------------
def add_synthetic_features(X, target_p=500, random_state=42):
    """Increase the total number of covariates to target_p by adding noise features"""
    np.random.seed(random_state)
    current_p = X.shape[1]
    n_additional = target_p - current_p
    if n_additional <= 0:
        return X
    synthetic = pd.DataFrame(
        np.random.randn(X.shape[0], n_additional),
        columns=[f"synthetic_{i}" for i in range(n_additional)]
    )
    return pd.concat([X.reset_index(drop=True), synthetic], axis=1)

def standardize(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_test_scaled

# -----------------------------
# Real dataset preparation
# -----------------------------
def prepare_real_dataset(dataset_dict, target_p=500, train_size=150, random_state=42):
    """Take standardized dataset and add synthetic features for high-dimensional setting"""
    X = pd.concat([dataset_dict["X_train"], dataset_dict["X_test"]], ignore_index=True)
    y = np.concatenate([dataset_dict["y_train"], dataset_dict["y_test"]])
    
    # Randomly sample training data
    np.random.seed(random_state)
    idx = np.random.choice(len(X), train_size, replace=False)
    X_train = X.iloc[idx]
    y_train = y[idx]
    
    # Use remaining for test
    mask = np.ones(len(X), dtype=bool)
    mask[idx] = False
    X_test = X.iloc[mask]
    y_test = y[mask]
    
    # Add synthetic features
    X_train_aug = add_synthetic_features(X_train, target_p=target_p, random_state=random_state)
    X_test_aug = add_synthetic_features(X_test, target_p=target_p, random_state=random_state+1)
    
    # Standardize
    X_train_scaled, X_test_scaled = standardize(X_train_aug, X_test_aug)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# -----------------------------
# Simulation scenarios
# -----------------------------
def generate_scenario(scenario=1, n=100, p=200, random_state=42):
    """Generate simulated dataset according to Zhu et al. 2015 scenarios"""
    np.random.seed(random_state)
    
    if scenario == 1:
        # Classification, independent uniform
        X = np.random.rand(n, p)
        mu = 1 / (1 + np.exp(-10*(X[:,0]-1) - 20*np.abs(X[:,1]-0.5)))
        y = np.random.binomial(1, mu)
    elif scenario == 2:
        # Non-linear regression, independent uniform
        X = np.random.rand(n, p)
        y = 100*(X[:,0]-0.5)**2 * np.maximum(X[:,1]-0.25,0) + np.random.randn(n)
    elif scenario == 3:
        # Checkerboard-like model, correlated covariates
        Sigma = 0.9 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
        X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
        y = 2*X[:,49]*X[:,99] + 2*X[:,149]*X[:,199] + np.random.randn(n)
    elif scenario == 4:
        # Linear model, correlated covariates
        Sigma = 0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p))) + 0.2*np.eye(p)
        X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
        y = 2*X[:,49] + 2*X[:,99] + 4*X[:,149] + np.random.randn(n)
    else:
        raise ValueError(f"Unknown scenario {scenario}")
    
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    return X, y
def run(dataset_dict):
    """
    Entry point for main.py
    Returns a dict with keys: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = prepare_real_dataset(dataset_dict)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
