# dso3_simulation_fixed.py

import numpy as np
from dso3.data_preparation import generate_scenario
from dso3.modeling import fit_rlt, fit_baselines
from dso3.evaluation import evaluate_metric  # your function returning numeric metric
from sklearn.model_selection import train_test_split

# -----------------------------
# Simulation parameters
# -----------------------------
scenarios = [1, 2, 3, 4]
p_values = [200, 500, 1000]
n_reps = 200
n_train = 100  # Scenario-dependent, can adjust
random_state = 42

# Store results
results_all = {s: {p: {} for p in p_values} for s in scenarios}

# -----------------------------
# Simulation loop
# -----------------------------
for scenario in scenarios:
    for p in p_values:
        np.random.seed(random_state)
        # Initialize storage for metrics
        scenario_metrics = {"RLT": [], "RF-all": [], "Lasso": []}  # extend as needed

        for rep in range(n_reps):
            # Generate simulated data
            X, y = generate_scenario(scenario=scenario, n=n_train, p=p, random_state=random_state + rep)

            # Split into train/test (e.g., 70% train)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state + rep)

            # Fit RLT
            rlt_model = fit_rlt(X_train, y_train, task='regression' if scenario>1 else 'classification')
            rlt_metric = evaluate_metric(rlt_model, X_test, y_test)
            scenario_metrics["RLT"].append(rlt_metric)

            # Fit baselines
            baselines = fit_baselines(X_train, y_train, task='regression' if scenario>1 else 'classification')
            for name, model in baselines.items():
                metric_val = evaluate_metric(model, X_test, y_test)
                scenario_metrics[name].append(metric_val)

        # Compute mean performance for each model
        results_all[scenario][p] = {m: np.mean(vals) for m, vals in scenario_metrics.items()}

# -----------------------------
# Print example results
# -----------------------------
for scenario in scenarios:
    print(f"Scenario {scenario}:")
    for p in p_values:
        print(f" p={p}: {results_all[scenario][p]}")
