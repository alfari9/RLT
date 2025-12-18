"""
DSO2 Evaluation
Compute variable importance and metrics for synthetic interaction datasets
"""

import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

def run(prepared_data, models_dict):
    """
    prepared_data: dict with X_test, y_test
    models_dict: dict of trained models (RLT variants)
    
    Returns:
        results: dict {variant_name: {'vi': VI dict, 'metric': score}}
    """
    X_test = prepared_data.get("X_test")
    y_test = prepared_data.get("y_test")
    
    task = prepared_data.get("task", "regression")
    results = {}

    for variant_name, model_output in models_dict.items():
        # model_output can be tuple (model, vi) or just vi dict
        if isinstance(model_output, tuple):
            model, vi = model_output
        else:
            model = None
            vi = model_output

        # Convert VI to dict if it's a Series
        if isinstance(vi, pd.Series):
            vi_dict = vi.to_dict()
        else:
            vi_dict = vi

        # Compute metric if model and test data exist
        metric = None
        if model is not None and X_test is not None and y_test is not None:
            try:
                y_pred = model.predict(X_test)
                if task == "classification":
                    metric = accuracy_score(y_test, y_pred)
                else:
                    metric = r2_score(y_test, y_pred)
            except Exception as e:
                print(f"Warning: Could not compute metric for {variant_name}: {e}")

        results[variant_name] = {
            "vi": vi_dict,
            "metric": metric
        }

    return results
