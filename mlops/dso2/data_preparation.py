import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def run(dataset, n_strong=4, n_features=50, add_synthetic=True):
    """
    Prepare dataset for DSO2 modeling (synthetic XOR or real datasets)
    Returns dict with X_train, X_test, y_train, y_test, feature_names
    """
    rng = np.random.RandomState(42)

    # Check if dataset is synthetic (already has X_train/X_test)
    if "X_train" in dataset and "X_test" in dataset:
        X_train = dataset["X_train"]
        X_test = dataset["X_test"]
        y_train = dataset["y_train"]
        y_test = dataset["y_test"]
        feature_names = X_train.columns.tolist()
    else:
        # Real dataset: split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            dataset["X"], dataset["y"], test_size=0.3, random_state=42
        )
        feature_names = X_train.columns.tolist()

        if add_synthetic:
            n_samples_train = X_train.shape[0]
            n_samples_test = X_test.shape[0]

            # Generate synthetic XOR features
            X_strong_train = rng.randint(0, 2, size=(n_samples_train, n_strong))
            X_strong_test = rng.randint(0, 2, size=(n_samples_test, n_strong))

            X_noise_train = rng.randn(n_samples_train, n_features - n_strong)
            X_noise_test = rng.randn(n_samples_test, n_features - n_strong)

            X_synth_train = np.hstack([X_strong_train, X_noise_train])
            X_synth_test = np.hstack([X_strong_test, X_noise_test])

            synth_columns = [f"synthetic_{i}" for i in range(n_features)]
            X_synth_train = pd.DataFrame(X_synth_train, columns=synth_columns, index=X_train.index)
            X_synth_test = pd.DataFrame(X_synth_test, columns=synth_columns, index=X_test.index)

            # Append synthetic features
            X_train = pd.concat([X_train, X_synth_train], axis=1)
            X_test = pd.concat([X_test, X_synth_test], axis=1)
            feature_names += synth_columns

            # Target for synthetic XOR (optional, can be ignored for real datasets)
            y_train_synth = np.logical_xor.reduce(X_strong_train, axis=1).astype(int)
            y_test_synth = np.logical_xor.reduce(X_strong_test, axis=1).astype(int)
            # For real datasets, keep original y
            # If you want to replace y with synthetic XOR, uncomment:
            # y_train = y_train_synth
            # y_test = y_test_synth

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names
    }
