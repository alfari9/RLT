"""
RLT (Reinforcement Learning Trees) Implementation
Extracted from notebook for deployment
"""
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


class Node:
    def __init__(self, depth=0):
        self.depth = depth
        self.is_leaf = False
        self.value = None  # Prediction value for leaf nodes
        self.impurity = None # Impurity of the node
        self.split_var = None  # Index of the feature to split on (for univariate splits)
        self.split_val = None  # Value to split on
        self.linear_coef = None # Coefficients for linear combination splits
        self.variables_used = [] # Indices of features used in this node (for VI and linear splits)
        self.left = None  # Left child node
        self.right = None # Right child node


class RLTTree:
    def __init__(self, max_depth=5, min_samples_split=10, task="classification",
             max_vars=5, linear_split=False, vi_threshold=0.5,
             muting_rate=0.0, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task = task
        self.max_vars = max_vars
        self.linear_split = linear_split
        self.vi_threshold = vi_threshold
        self.muting_rate = muting_rate
        self.random_state = np.random.RandomState(random_state)
        self.root = None

    def fit(self, X, y):
        # X must be a DataFrame for feature name access
        self.feature_names = X.columns.tolist()
        self.root = self.build_node(X, y, depth=0)

    def predict(self, X):
        preds = []
        for _, x in X.iterrows():
            preds.append(self._predict_row(x, self.root))
        return np.array(preds)

    def explain_prediction(self, x):
        path_info = []
        self._predict_row(x, self.root, path_info=path_info)
        return path_info

    def _predict_row(self, x, node, path_info=None):
        if node.is_leaf:
            return node.value

        # Feature contribution tracking
        split_feature = None
        split_value = None
        decision = None
        next_node = None

        if node.linear_coef is not None:
            # Linear split
            used_features = [self.feature_names[i] for i in node.variables_used]
            proj = np.dot(x[used_features], node.linear_coef)
            split_feature = f"Linear Comb ({', '.join(used_features)})"
            split_value = node.split_val
            if proj <= node.split_val:
                decision = f"Projection ({proj:.2f}) <= {node.split_val:.2f} (Left)"
                next_node = node.left
            else:
                decision = f"Projection ({proj:.2f}) > {node.split_val:.2f} (Right)"
                next_node = node.right
        else:
            # Univariate split
            split_feature = self.feature_names[node.split_var]
            split_value = node.split_val
            if x[split_feature] <= node.split_val:
                decision = f"{split_feature} ({x[split_feature]:.2f}) <= {node.split_val:.2f} (Left)"
                next_node = node.left
            else:
                decision = f"{split_feature} ({x[split_feature]:.2f}) > {node.split_val:.2f} (Right)"
                next_node = node.right

        if path_info is not None:
            path_info.append({
                'depth': node.depth,
                'feature': split_feature,
                'value': x[split_feature] if node.linear_coef is None else proj,
                'split_value': split_value,
                'decision': decision,
                'impurity_reduction': node.impurity - next_node.impurity if next_node else node.impurity
            })

        return self._predict_row(x, next_node, path_info)

    def build_node(self, X, y, depth):
        node = Node(depth=depth)
        node.impurity = self.compute_impurity(y)
        node.value = self.leaf_value(y)

        if depth >= self.max_depth or len(y) < self.min_samples_split or node.impurity == 0:
            node.is_leaf = True
            return node

        # 1. Embedded VI (ExtraTrees at node)
        vi = self.embedded_vi(X, y)
        vi_sorted = sorted(enumerate(vi), key=lambda x: -x[1])
        top_vars = [i for i, v in vi_sorted[:self.max_vars] if v >= (self.vi_threshold * max(vi))]
        if not top_vars:
            top_vars = [i for i, v in vi_sorted[:self.max_vars]]
        node.variables_used = top_vars

        # 2. Variable muting (focus only on informative vars)
        X_sub = X.iloc[:, top_vars]

        # 3. Reinforcement Learning Bandit split selection (simplified UCB/epsilon-greedy is implicit in best gain search)
        best_gain = -np.inf
        best_split = None
        best_left = None
        best_right = None
        best_linear = None

        # Univariate splits
        for var_idx in top_vars:
            col = X.iloc[:, var_idx]
            # Simplified candidate selection
            val_candidates = np.linspace(col.min(), col.max(), num=10)
            for v in val_candidates:
                left_idx = col <= v
                right_idx = col > v
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                gain = self.compute_impurity(y) - (
                    left_idx.sum() / len(y) * self.compute_impurity(y[left_idx])
                    + right_idx.sum() / len(y) * self.compute_impurity(y[right_idx]))

                if gain > best_gain:
                    best_gain = gain
                    best_split = (var_idx, v)
                    best_left = (X[left_idx], y[left_idx])
                    best_right = (X[right_idx], y[right_idx])
                    best_linear = None

        # 4. Linear combination split (optional)
        if self.linear_split and len(top_vars) >= 2:
            # Generate random coefficients
            coefs = self.random_state.randn(len(top_vars))
            # Normalize coefficients for stability
            coefs = coefs / np.linalg.norm(coefs)

            proj = np.dot(X_sub.values, coefs)
            val_candidates = np.linspace(proj.min(), proj.max(), num=10)

            for v in val_candidates:
                left_idx = proj <= v
                right_idx = proj > v
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                gain = self.compute_impurity(y) - (
                    left_idx.sum() / len(y) * self.compute_impurity(y[left_idx])
                    + right_idx.sum() / len(y) * self.compute_impurity(y[right_idx]))

                if gain > best_gain:
                    best_gain = gain
                    best_split = (None, v)
                    best_left = (X[left_idx], y[left_idx])
                    best_right = (X[right_idx], y[right_idx])
                    best_linear = coefs

        # Si pas de split, faire une feuille
        if best_gain <= 0 or best_left is None or best_right is None:
            node.is_leaf = True
            return node

        # Enregistrer le split choisi
        node.split_var, node.split_val = best_split
        node.linear_coef = best_linear

        # Pour les splits linéaires, variables_used contient les indices des caractéristiques utilisées
        if node.linear_coef is not None:
            node.variables_used = top_vars
        else:
            # Pour les splits univariés, variables_used contient l'indice de la caractéristique utilisée
            node.variables_used = [node.split_var]

        # Construire récursivement les enfants
        node.left = self.build_node(*best_left, depth=depth+1)
        node.right = self.build_node(*best_right, depth=depth+1)
        return node

    def embedded_vi(self, X, y):
        if self.task == "classification":
            model = ExtraTreesClassifier(n_estimators=10, max_depth=3, random_state=42)
        else:
            model = ExtraTreesRegressor(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)
        return model.feature_importances_

    def compute_impurity(self, y):
        if self.task == "classification":
            # Gini impurity
            if len(y) == 0: return 0
            probs = np.bincount(y.astype(int)) / len(y)
            return 1 - np.sum(probs ** 2)
        else:
            # Variance reduction (MSE)
            return np.var(y)

    def leaf_value(self, y):
        if self.task == "classification":
            # Majority vote for binary classification (0/1)
            if len(y) == 0: return 0
            return 1 if np.mean(y) > 0.5 else 0
        else:
            return np.mean(y)


class RLTForest:
    def __init__(self, n_trees=50, **kwargs):
        self.n_trees = n_trees
        self.kwargs = kwargs
        self.trees = []
        self.feature_names = []

    def fit(self, X, y):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        # Ensure y is a Series
        if not isinstance(y, pd.Series):
            y_df = pd.Series(y, index=X_df.index, copy=True)
        else:
            y_df = y.copy()
            
        self.feature_names = X_df.columns.tolist()
        self.trees = []
        
        for i in range(self.n_trees):
            sample_idx = X_df.sample(frac=0.7, replace=True, random_state=i).index
            X_sample = X_df.loc[sample_idx]
            y_sample = y_df.loc[sample_idx]
            tree = RLTTree(**self.kwargs, random_state=i)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
            
        preds = np.array([tree.predict(X_df) for tree in self.trees])
        
        if self.kwargs.get("task") == "classification":
            # Use mode for classification
            mode_result = [Counter(preds[:, i].astype(int)).most_common(1)[0][0] for i in range(preds.shape[1])]
            return np.array(mode_result)
        else:
            return np.mean(preds, axis=0)

    def get_feature_importance(self):
        # Simplified: average of embedded VI from trees
        vi_sum = np.zeros(len(self.feature_names))
        for tree in self.trees:
            # This is a simplification
            vi_sum += tree.embedded_vi(tree.root.X, tree.root.y)
        return vi_sum / self.n_trees

    def explain_instance(self, x_instance, n_trees=3):
        if not isinstance(x_instance, pd.Series):
            X_df = pd.DataFrame([x_instance], columns=self.feature_names)
        else:
            X_df = pd.DataFrame([x_instance], columns=self.feature_names)
            
        explanations = []
        for i in range(min(n_trees, self.n_trees)):
            tree = self.trees[i]
            path_info = tree.explain_prediction(X_df.iloc[0])
            explanations.append(path_info)
        return explanations
