import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

def remove_outliers_iqr(X: pd.DataFrame, y: pd.Series, factor: float = 1.5):
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X < (Q1 - factor * IQR)) | (X > (Q3 + factor * IQR))).any(axis=1)
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)

def knn_impute(X: pd.DataFrame, n_neighbors: int = 5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

def prepare_data(data: dict, test_size: float = 0.2, random_state: int = 42):
    """
    Full data preparation pipeline:
    1. Outlier removal (IQR)
    2. KNN imputation
    3. Train/test split
    """
    X, y = data["X"], data["y"]

    # 1. Remove outliers
    X_clean, y_clean = remove_outliers_iqr(X, y)

    # 2. Impute missing values
    X_clean = knn_impute(X_clean)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state
    )

    preprocessor_desc = "IQR outlier removal + KNN imputation (no scaling)"
    return X_train, X_test, y_train, y_test, preprocessor_desc
