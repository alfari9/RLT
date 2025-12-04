"""
Data Understanding Module for RLT Project (CRISP-DM Phase 2)

Provides a unified interface to load and inspect datasets from:
- Predefined datasets
- External URLs (CSV, XLS, XLSX)
Returns standardized features (X), target (y), and metadata.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings("ignore")

# Predefined datasets mapping (name -> loader)
DATASETS_INFO = {
    "Breast Cancer": ("Classification", "sklearn", None),
    "Boston Housing": ("Regression", "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", 13),
    "Parkinson": ("Classification", "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data", "status"),
    "Sonar": ("Classification", "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data", 60),
    "White Wine": ("Regression", "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", "quality"),
    "Red Wine": ("Regression", "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", "quality"),
    "Parkinson Oxford": ("Regression", "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data", "total_UPDRS"),
    "Ozone": ("Classification","https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/onehr.data",-1),
    "Concrete": ("Regression", "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls", -1),
    "Auto MPG": ("Regression", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", 0)
}


def load_dataset(name_or_url, is_url=False):
    """
    Load a dataset from a predefined name or a URL.

    Parameters:
    ----------
    name_or_url : str
        Dataset name (predefined) or a URL.
    is_url : bool
        If True, treats name_or_url as URL.

    Returns:
    -------
    dict:
        {
            'dataset_name': str,
            'task_type': str ('Classification' or 'Regression'),
            'X': pd.DataFrame,
            'y': pd.Series,
            'target_name': str,
            'feature_names': list
        }
    """
    # -------------------------
    # 1. Load dataset from URL
    # -------------------------
    if is_url:
        url = name_or_url
        try:
            if url.endswith(".csv") or url.endswith(".CSV"):
                # Auto-detect delimiter
                import io, requests, csv
                r = requests.get(url)
                content = r.content.decode('utf-8')
                # Detect delimiter from first line
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(content.splitlines()[0])
                df = pd.read_csv(io.StringIO(content), delimiter=dialect.delimiter)
            elif url.endswith(".xls") or url.endswith(".xlsx"):
                df = pd.read_excel(url)
            else:
                df = pd.read_csv(url, delim_whitespace=True, header=None)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from URL '{url}': {str(e)}")

        # Assume last column is target
        X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
        y = df.iloc[:, -1]
        target_name = df.columns[-1]
        task_type = "Regression"  # default, user can customize
        return {
            "dataset_name": url,
            "task_type": task_type,
            "X": X,
            "y": y,
            "target_name": target_name,
            "feature_names": X.columns.tolist()
        }

    # -------------------------
    # 2. Predefined datasets
    # -------------------------
    if name_or_url not in DATASETS_INFO:
        raise FileNotFoundError(f"No dataset registered with name '{name_or_url}'")
    task_type, loader, target_info = DATASETS_INFO[name_or_url]

    if loader == "sklearn":
        # Breast Cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        target_name = 'target'
    else:
        url = loader.strip()
        if "winequality" in url:
            df = pd.read_csv(url, sep=';')
        elif "housing.data" in url:
            col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                         'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
            df = pd.read_csv(url, delim_whitespace=True, names=col_names)
        elif "auto-mpg.data" in url:
            df = pd.read_csv(url, delim_whitespace=True, na_values='?')
            col_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                         'acceleration', 'model_year', 'origin', 'car_name']
            df.columns = col_names
            df = df.drop(columns=['car_name'])
            df = df.apply(pd.to_numeric, errors='coerce')
        elif "parkinsons_updrs.data" in url or "parkinsons.data" in url:
            df = pd.read_csv(url)
        elif "sonar.all-data" in url:
            df = pd.read_csv(url, header=None)
            df.columns = [f"V{i}" for i in range(1, 61)] + ['Class']
        elif "ozone_level_detection.data" in url:
            df = pd.read_csv(url, skiprows=4, header=None, delim_whitespace=True)
        elif "Concrete_Data" in url:
            col_names = [
                'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
                'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age',
                'Concrete_Compressive_Strength'
            ]
            df = pd.read_excel(url, names=col_names)
        else:
            try:
                df = pd.read_csv(url)
            except:
                df = pd.read_csv(url, delim_whitespace=True, header=None)

        # Extract target
        if isinstance(target_info, str):
            y = df[target_info].copy()
            X = df.drop(columns=[target_info])
            target_name = target_info
        elif isinstance(target_info, int):
            target_col = df.columns[target_info]
            y = df[target_col].copy()
            X = df.drop(columns=[target_col])
            target_name = target_col
        else:
            y = df.iloc[:, -1].copy()
            X = df.iloc[:, :-1]
            target_name = df.columns[-1]

        X = X.apply(pd.to_numeric, errors='coerce')

    # Final cleanup
    X = X.dropna(axis=1, how='all').reset_index(drop=True)
    y = y.reset_index(drop=True)

    return {
        "dataset_name": name_or_url,
        "task_type": task_type,
        "X": X,
        "y": y,
        "target_name": target_name,
        "feature_names": X.columns.tolist()
    }
