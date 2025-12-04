# main.py
# CRISP-DM Phase 2 & 3 & 4: Data Understanding + Data Preparation + Modeling

import os
import argparse
import pandas as pd
from data_understanding import load_dataset
from data_preparation import prepare_data  # type: ignore
from modeling import train_and_evaluate

# Ensure reports folder exists
os.makedirs("reports", exist_ok=True)
eda_report_path = "reports/eda_summary.txt"
model_report_path = "reports/modeling_summary.txt"

def main():
    parser = argparse.ArgumentParser(description="CRISP-DM Data Pipeline")
    parser.add_argument("--dataset", type=str, help="Name of the predefined dataset to load")
    parser.add_argument("--path", type=str, help="Local CSV/XLS/XLSX dataset path")
    parser.add_argument("--url", type=str, help="URL to CSV/XLS/XLSX dataset")
    args = parser.parse_args()

    # ------------------------------
    # Step 1: Load dataset
    # ------------------------------
    if args.dataset:
        print(f"\n🔍 Loading dataset: {args.dataset}")
        data = load_dataset(args.dataset)
        dataset_name = args.dataset
    elif args.path or args.url:
        file_source = args.path if args.path else args.url
        print(f"\n🔍 Loading dataset from {'URL' if args.url else 'path'}: {file_source}")
        try:
            if file_source.endswith(".csv"):
                df = pd.read_csv(file_source)
            elif file_source.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file_source)
            else:
                df = pd.read_csv(file_source)  # fallback
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return

        # Automatically guess target: last column
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        data = {
            "dataset_name": file_source,
            "task_type": "Classification" if y.nunique() < 20 else "Regression",
            "X": X,
            "y": y,
            "target_name": df.columns[-1],
            "feature_names": X.columns.tolist()
        }
        dataset_name = file_source
    else:
        print("Please provide --dataset, --path, or --url")
        return

    # ------------------------------
    # Step 2: Data Understanding
    # ------------------------------
    X, y = data["X"], data["y"]
    task = data["task_type"]
    n_samples, n_features = X.shape
    missing_X = X.isnull().sum().sum()
    missing_y = y.isnull().sum()

    summary = f"""
Dataset: {dataset_name}
- Task: {task}
- Shape: {n_samples} samples × {n_features} features
- Missing in X: {missing_X} ({100 * missing_X / (n_samples * n_features):.2f}%)
- Missing in y: {missing_y}
"""
    if task == "Classification":
        class_dist = y.value_counts()
        summary += f"- Class distribution:\n{class_dist.to_string()}\n"
    else:
        y_stats = y.describe()
        summary += f"- Target stats:\n{y_stats.to_string()}\n"

    print(summary)
    with open(eda_report_path, "w") as report:
        report.write("CRISP-DM Phase 2 & 3: Data Understanding + Preparation Summary\n")
        report.write("=" * 60 + "\n")
        report.write(summary + "\n")

    # ------------------------------
    # Step 3: Data Preparation
    # ------------------------------
    print("🔧 Preparing data for modeling...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(data)

    print(f"- Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"- Preprocessor: {preprocessor}")
    print("\n✅ Data preparation complete.")

    # Log prep to report
    with open(eda_report_path, "a") as report:
        report.write("-" * 60 + "\n")
        report.write(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")
        report.write(f"Preprocessor: {preprocessor}\n")
        report.write("=" * 60 + "\n")

    # ------------------------------
    # Step 4: Modeling
    # ------------------------------
    print("\n🎯 Training models and evaluating performance...")
    results = train_and_evaluate(data, X_train, X_test, y_train, y_test)

    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.4f}")

    # Log modeling results
    with open(model_report_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write("="*50 + "\n")
        for model_name, metrics in results.items():
            f.write(f"\nModel: {model_name}\n")
            for metric, value in metrics.items():
                f.write(f"- {metric}: {value:.4f}\n")
        f.write("\n Modeling complete.\n")


if __name__ == "__main__":
    main()
