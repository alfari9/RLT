"""
main.py
Unified CLI for all DSOs
Usage: python main.py "Dataset Name" -dso dso1|dso2|dso3
"""

import sys
import importlib
import pandas as pd
from data_understanding import load_dataset
from sklearn.metrics import accuracy_score, r2_score

# -----------------------------
# CLI argument parsing
# -----------------------------
if len(sys.argv) < 4 or sys.argv[2] != "-dso":
    print('Usage: python main.py "Dataset Name" -dso dso1|dso2|dso3')
    sys.exit(1)

dataset_name = sys.argv[1]
dso_name = sys.argv[3]

# -----------------------------
# Load dataset and print summary
# -----------------------------
print(f"[1/4] Loading dataset '{dataset_name}'...")
dataset = load_dataset(dataset_name)

# Dataset dict can contain either 'X_train'/'X_test' (DSO1) or 'X'/'y' (DSO2)
if "X_train" in dataset:
    X = pd.concat([dataset["X_train"], dataset["X_test"]], axis=0)
    y = pd.concat([pd.Series(dataset["y_train"]), pd.Series(dataset["y_test"])], axis=0)
else:
    X = dataset["X"]
    y = dataset["y"]

task = dataset["task"]

print(f"  Task: {task}")
print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
print("  Missing values per column:")
print(X.isnull().sum())

# -----------------------------
# Import DSO modules
# -----------------------------
try:
    prep_module = importlib.import_module(f"{dso_name}.data_preparation")
    model_module = importlib.import_module(f"{dso_name}.modeling")
    eval_module = importlib.import_module(f"{dso_name}.evaluation")
except ModuleNotFoundError:
    print(f"DSO '{dso_name}' not found.")
    sys.exit(1)

# -----------------------------
# Data Preparation
# -----------------------------
print(f"[2/4] Running Data Preparation for {dso_name}...")
prepared = prep_module.run(dataset)
prepared["task"] = task

# -----------------------------
# Modeling
# -----------------------------
print(f"[3/4] Running Modeling for {dso_name}...")
model_output = model_module.run(prepared)

# -----------------------------
# Evaluation
# -----------------------------
print(f"[4/4] Running Evaluation for {dso_name}...")

def safe_series(vi_dict):
    """Convert VI dict to numeric Series, ignore non-numeric"""
    return pd.Series({k: v for k, v in vi_dict.items() if isinstance(v, (int, float))}).sort_values(ascending=False)

if dso_name == "dso2":
    # DSO2 returns tuple: (models_dict, vi_dict)
    if isinstance(model_output, tuple):
        models_dict, vi_dict = model_output
    else:
        vi_dict = model_output
        models_dict = None

    results = eval_module.run(prepared, vi_dict)

    print("\n=== Evaluation Results ===")
    for variant, info in results.items():
        # info can be numeric VI dict or nested dict
        if isinstance(info, dict) and "vi" in info:
            vi_series = safe_series(info["vi"])
            metric = info.get("metric", None)
        else:
            vi_series = safe_series(info)
            metric = None

        print(f"\nVariant: {variant}")
        if metric is not None:
            metric_name = "R²" if task == "regression" else "Accuracy"
            print(f"Metric ({metric_name}): {metric}")
        print("Top 10 variables by importance:")
        print(vi_series.head(10))

else:
    # DSO1: single model, evaluation needs informative_features
    informative_features = [f for f in prepared.get("feature_names", X.columns) if not f.startswith("synthetic_")]
    results = eval_module.run(prepared, model_output, informative_features=informative_features)

    print("\n=== Evaluation Results ===")
    metric_name = "Accuracy" if task == "classification" else "R²"
    print(f"{metric_name}: {results['metric']:.4f}")
    print(f"Recall@50: {results['recall_at_50']:.4f}")
    print(f"VI Sparsity: {results['vi_sparsity']:.4f}")

    vi = results["variable_importance"]
    real_vars = [f for f in vi.index if not f.startswith("synthetic_")]
    synthetic_vars = [f for f in vi.index if f.startswith("synthetic_")]

    print("\nTop 10 Informative (Real) Features:")
    print(vi[real_vars].head(10))

    print("\nTop 10 Noise (Synthetic) Features:")
    print(vi[synthetic_vars].head(10))

print("\nPipeline completed successfully!")
