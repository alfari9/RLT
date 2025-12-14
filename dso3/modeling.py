"""
DSO3 modeling: RLT + baseline models (RF, Lasso)
"""

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, accuracy_score

def fit_rlt(X_train, y_train, task='regression', n_estimators=500, max_features=5):
    """Placeholder RLT: RandomForest with linear combination splits"""
    if task == 'regression':
        model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=42)
    model.fit(X_train, y_train)
    return model

def fit_baselines(X_train, y_train, task='regression'):
    """Fit other methods"""
    models = {}
    if task == 'regression':
        models['RF-all'] = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=42).fit(X_train, y_train)
        models['Lasso'] = Lasso(alpha=0.01).fit(X_train, y_train)
    else:
        models['RF-all'] = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=42).fit(X_train, y_train)
    return models
def run(prepared_data):
    """
    Entry point for main.py
    Fits RLT + baseline models and returns a dict of fitted models
    """
    X_train = prepared_data["X_train"]
    y_train = prepared_data["y_train"]
    task = "regression" if prepared_data.get("type", "regression") == "Regression" else "classification"
    
    models = {}
    # Fit RLT
    models["RLT"] = fit_rlt(X_train, y_train, task=task)
    
    # Fit baselines
    baselines = fit_baselines(X_train, y_train, task=task)
    models.update(baselines)
    
    return models
