from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

def run(prepared_data, models):
    """
    Evaluate all fitted models on the prepared test data.
    Returns a dict of results with metric + predictions.
    """
    X_test = prepared_data["X_test"]
    y_test = prepared_data["y_test"]
    task = "regression" if prepared_data.get("type", "Regression") == "Regression" else "classification"

    results = {}
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            if task == "regression":
                metric = mean_squared_error(y_test, y_pred)
            else:
                metric = accuracy_score(y_test, y_pred)
            results[name] = {"metric": metric, "y_pred": y_pred}
        except Exception as e:
            results[name] = {"error": str(e)}
    return results
# dso3/evaluation.py



def evaluate_metric(model, X_test, y_test):
    """
    Return a numeric evaluation metric:
    - Regression: Mean Squared Error
    - Classification: Accuracy
    """
    y_pred = model.predict(X_test)
    if len(np.unique(y_test)) > 2 or y_test.dtype.kind in 'fc':  # regression
        return mean_squared_error(y_test, y_pred)
    else:  # classification
        return accuracy_score(y_test, y_pred)
