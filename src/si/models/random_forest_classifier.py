from typing import Literal

import numpy as np

from SIB_machine_learning.src.si.data.dataset import Dataset
from SIB_machine_learning.src.si.metrics.accuracy import accuracy
from SIB_machine_learning.src.si.models.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:

    def __init__(self, n_estimators: int, min_sample_split: int,
                 mode: Literal['gini', 'entropy'], seed: int, max_features: int = None, max_depth: int = None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def fit(self, dataset: Dataset):
        # Sets the random seed
        np.random.seed(self.seed)

        # Defines self.max_features to be int(np.sqrt(n_features)) if None
        if self.max_features is None:
            self.max_features = int(np.sqrt(dataset.X.shape[1]))

        # Loop over the number of estimators to create individual trees
        for i in range(self.n_estimators):
            # Pick n_samples random samples from the dataset with replacement
            random_indices = np.random.choice(dataset.X.shape[0], dataset.X.shape[0], replace=True)

            # Pick self.max_features random features without replacement from the original dataset
            random_features = np.random.choice(dataset.X.shape[1], self.max_features, replace=False)

            # Create and train a decision tree with the bootstrap dataset
            bootsrap_dataset = Dataset(dataset.X[random_indices][:, random_features], dataset.y[random_indices])
            tree = DecisionTreeClassifier(self.min_sample_split, self.max_depth, self.mode)
            tree.fit(bootsrap_dataset)

            # Append a tuple containing the features used and the trained tree
            self.trees.append((random_features, tree))

        # Return itself (self)
        return self

    def predict(self, dataset: Dataset):
        # Get predictions for each tree using the respective set of features
        predictions = np.array([tree.predict(Dataset(dataset.X[:, features])) for features, tree in self.trees])
        # Get the most common predicted class for each sample
        predictions = np.array([np.argmax(np.bincount(prediction)) for prediction in predictions.T])
        # Return predictions
        return predictions

    def score(self, dataset: Dataset):
        # Get predictions using the predict method
        predictions = self.predict(dataset)
        # Computes the accuracy between predicted and real values
        accuracy_score = accuracy(dataset.y, predictions)
        # Return accuracy
        return accuracy_score


if __name__ == '__main__':
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)

    X_test = np.random.rand(50, 10)
    y_test = np.random.randint(0, 2, 50)

    random_forest = RandomForestClassifier(n_estimators=10, max_features=None, min_sample_split=2, max_depth=None,
                                           mode='gini', seed=42)

    random_forest.fit(Dataset(X_train, y_train))

    accuracy_test = random_forest.score(Dataset(X_test, y_test))

    print("Accuracy on Test Set:", accuracy_test)
