from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def train_and_evaluate(data, X_train, X_test, y_train, y_test):
    task = data["task_type"]
    results = {}

    if task == "Classification":
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000)
        }
    else:  # Regression
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Lasso": Lasso(alpha=0.1),
            "LinearRegression": LinearRegression()
        }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            try:
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
            except:
                auc = np.nan
            results[name] = {"accuracy": acc, "f1_score": f1, "roc_auc": auc}

        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {"MSE": mse, "MAE": mae, "R2": r2}

    return results
