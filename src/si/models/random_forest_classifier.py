from typing import Literal

import numpy as np

from SIB_machine_learning.src.si.data.dataset import Dataset
from SIB_machine_learning.src.si.metrics.accuracy import accuracy
from SIB_machine_learning.src.si.models.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_estimators, max_features=None, min_sample_split=2, max_depth=None,
                 mode: Literal['gini', 'entropy'] = 'gini', seed=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        self.trees = []

    def _set_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)

    def _get_bootstrap_dataset(self, dataset):
        n_samples, n_features = dataset.shape()
        indices = np.random.choice(n_samples, n_samples, replace=True)
        features = np.random.choice(n_features, self.max_features, replace=False)
        return Dataset(X=dataset.X[indices][:, features], y=dataset.y[indices])

    def fit(self, dataset):
        self._set_seed()
        n_features = dataset.shape()[1]
        self.max_features = int(np.sqrt(n_features)) if self.max_features is None else self.max_features

        for _ in range(self.n_estimators):
            bootstrap_dataset = self._get_bootstrap_dataset(dataset)
            tree = DecisionTreeClassifier(min_sample_split=self.min_sample_split,
                                          max_depth=self.max_depth,
                                          mode=self.mode)
            tree.fit(bootstrap_dataset)
            self.trees.append((bootstrap_dataset.features, tree))

        return self

    def predict(self, dataset):
        predictions = []
        for _, tree in self.trees:
            subset_X = dataset.X[:, tree.feature_idx]
            tree_predictions = tree.predict(Dataset(X=subset_X))
            predictions.append(tree_predictions)

        predictions = np.array(predictions).T
        return np.array([np.argmax(np.bincount(sample)) for sample in predictions])

    def score(self, dataset):
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)
