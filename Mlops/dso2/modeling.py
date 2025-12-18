from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd

def run(prepared_data, variants=["RLT1","RLT2","RLT5","RLT-naive"]):
    """
    Fit RLT variants on prepared data.
    Returns dict of models and dict of feature importances
    """
    X_train = prepared_data["X_train"]
    y_train = prepared_data["y_train"]
    task = prepared_data.get("task", "regression")

    models = {}
    vis = {}

    for v in variants:
        if v == "RLT-naive":
            max_features = "sqrt"
        elif v == "RLT1":
            max_features = 1
        elif v == "RLT2":
            max_features = 2
        elif v == "RLT5":
            max_features = 5
        else:
            max_features = "sqrt"

        if task == "classification":
            model = RandomForestClassifier(
                n_estimators=500,
                max_features=max_features,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=500,
                max_features=max_features,
                random_state=42
            )

        model.fit(X_train, y_train)

        # Feature importance
        vi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        models[v] = (model, vi)
        vis[v] = vi

    return models  # returning models as dict of (model, vi)
