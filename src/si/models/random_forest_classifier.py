import numpy as np
from collections import Counter
from typing import List, Tuple, Union

from si.base.model import Model
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier   # your previous exercise
from si.metrics.accuracy import accuracy


class RandomForestClassifier(Model):
    """
    Random Forest Classifier implemented as an ensemble of Decision Trees.
    Features for each tree are selected randomly → reduces correlation,
    improves generalization and avoids overfitting.

    Parameters
    ----------
    n_estimators : int
        Number of decision trees in the forest
    max_features : int or None
        Number of features per tree (if None → sqrt(n_features))
    min_samples_split : int
        Minimum samples required to split a node in each tree
    max_depth : int or None
        Max depth of each tree (None means unlimited)
    mode : str
        Impurity mode ('gini' or 'entropy')
    seed : int
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_features: Union[int, None] = None,
        min_samples_split: int = 2,
        max_depth: Union[int, None] = None,
        mode: str = "gini",
        seed: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        # trained forest → list of (selected_features, decision_tree)
        self.trees: List[Tuple[np.ndarray, DecisionTreeClassifier]] = []


    def _fit(self, dataset: Dataset):
        """Train `n_estimators` decision trees using bootstrap sampling"""
        np.random.seed(self.seed)

        n_samples, n_features = dataset.X.shape

        # If no feature limit: sqrt rule
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            # Bootstrap samples WITH replacement
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)

            # Random feature subset WITHOUT replacement
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)

            X_bootstrap = dataset.X[sample_indices][:, feature_indices]
            y_bootstrap = dataset.y[sample_indices]

            # Train Decision Tree
            tree = DecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(Dataset(X_bootstrap, y_bootstrap)) #boo

            # Store features + tree
            self.trees.append((feature_indices, tree))

        return self


    def _predict(self, dataset: Dataset) -> np.ndarray:
        """Predict using majority vote across all trees"""

        predictions = []

        for features, tree in self.trees:
            preds = tree.predict(Dataset(dataset.X[:, features], dataset.y))
            predictions.append(preds)

        # Convert (n_estimators, n_samples) → (n_samples, n_estimators)
        predictions = np.array(predictions).T

        # majority vote
        return np.array([Counter(row).most_common(1)[0][0] for row in predictions])


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """Compute accuracy score"""
        return accuracy(dataset.y, predictions)
