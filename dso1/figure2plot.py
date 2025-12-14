import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
from dso1 import modeling, evaluation, data_preparation
from data_understanding import load_dataset

# -----------------------------
# Load dataset
# -----------------------------
dataset_name = "Concrete"  # replace with any dataset you have
dataset = load_dataset(dataset_name)

# -----------------------------
# Prepare data (DSO1)
# -----------------------------
prepared = data_preparation.run(dataset)
X_train, X_test = prepared["X_train"], prepared["X_test"]
y_train, y_test = prepared["y_train"], prepared["y_test"]

# -----------------------------
# Random Forest - single run
# -----------------------------
rf_model = modeling.run(prepared)
rf_results = evaluation.run(prepared, rf_model)
vi_rf_single = rf_results["variable_importance"]

# -----------------------------
# RLT - single run
# -----------------------------
# Assuming your RLT model is implemented in dso1/modeling.py and called with a flag
# For simplicity, let's simulate RLT as a modified RF in your current pipeline
rlt_model = modeling.run(prepared)  # replace with real RLT run
rlt_results = evaluation.run(prepared, rlt_model)
vi_rlt_single = rlt_results["variable_importance"]

# -----------------------------
# Optional: multiple runs for averaging
# -----------------------------
n_runs = 10
vi_rf_avg = np.zeros_like(vi_rf_single)
vi_rlt_avg = np.zeros_like(vi_rlt_single)

for _ in range(n_runs):
    # Random Forest
    rf_model = modeling.run(prepared)
    vi_rf_avg += evaluation.run(prepared, rf_model)["variable_importance"].values
    
    # RLT
    rlt_model = modeling.run(prepared)  # replace with actual RLT
    vi_rlt_avg += evaluation.run(prepared, rlt_model)["variable_importance"].values

vi_rf_avg /= n_runs
vi_rlt_avg /= n_runs

# -----------------------------
# Figure 2 plot
# -----------------------------
P = X_train.shape[1]
strong_vars = [50, 100, 150, 200]  # adjust if known, else highlight top VI features
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(f"Figure 2: Variable Importance Comparison ({dataset_name})", fontsize=16)

def plot_vi(ax, vi, title):
    if isinstance(vi, pd.Series):
        vi = vi.values
    colors = ['black' if (i+1) in strong_vars else 'gray' for i in range(len(vi))]
    ax.bar(np.arange(1, len(vi)+1), vi, color=colors)
    ax.set_xlabel("Variable Index")
    ax.set_ylabel("Variable Importance")
    ax.set_title(title)
    ax.set_xticks(np.arange(0, len(vi)+1, 20))

plot_vi(axes[0,0], vi_rf_single, "Random Forest - Single Run")
plot_vi(axes[0,1], vi_rlt_single, "RLT - Single Run")
plot_vi(axes[1,0], vi_rf_avg, "Random Forest - Average")
plot_vi(axes[1,1], vi_rlt_avg, "RLT - Average")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"figure2_vi_{dataset_name.replace(' ', '_')}.png", dpi=300)
plt.show()
