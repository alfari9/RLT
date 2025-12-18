"""
DSO1: Feature Selection - Modeling
Random Forests detect informative features
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def run(prepared_data):
    X_train = prepared_data["X_train"]
    y_train = prepared_data["y_train"]
    task = prepared_data["task"]

    # Choose model
    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=500,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=500,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

    # Fit model
    model.fit(X_train, y_train)
    prepared_data["model"] = model
    return model
