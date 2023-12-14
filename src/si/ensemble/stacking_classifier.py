from typing import List

import numpy as np

from SIB_machine_learning.src.si.data.dataset import Dataset
from SIB_machine_learning.src.si.metrics.accuracy import accuracy
from SIB_machine_learning.src.si.models.decision_tree_classifier import DecisionTreeClassifier
from SIB_machine_learning.src.si.models.random_forest_classifier import RandomForestClassifier


class StackingClassifier:
    def __init__(self, models: List, final_model):
        self.models = models
        self.final_model = final_model
        self.final_model_trained = False

    def fit(self, dataset: Dataset):
        for model in self.models:
            model.fit(dataset)

        predictions = np.array([model.predict(dataset) for model in self.models]).T

        self.final_model.fit(Dataset(X=predictions, y=dataset.y))
        self.final_model_trained = True

        return self

    def predict(self, dataset: Dataset):
        if not self.final_model_trained:
            raise ValueError("Final model has not been trained. Please run fit() first.")

        initial_predictions = np.array([model.predict(dataset) for model in self.models]).T

        final_predictions = self.final_model.predict(Dataset(X=initial_predictions))

        return final_predictions

    def score(self, dataset: Dataset):
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)


if __name__ == '__main__':
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 1, 0, 1])

    X_test = np.array([[1.5, 2.5], [3.5, 4.5]])
    y_test = np.array([0, 1])

    decision_tree1 = DecisionTreeClassifier(min_sample_split=2, max_depth=3, mode='gini')
    decision_tree2 = DecisionTreeClassifier(min_sample_split=2, max_depth=3, mode='gini')

    final_model = RandomForestClassifier(n_estimators=3, max_features=None, min_sample_split=2, max_depth=None,
                                         mode='gini', seed=42)

    stacking_classifier = StackingClassifier(models=[decision_tree1, decision_tree2], final_model=final_model)

    stacking_classifier.fit(Dataset(X_train, y_train))

    predictions = stacking_classifier.predict(Dataset(X_test))

    accuracy_score = stacking_classifier.score(Dataset(X_test, y_test))

    print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy_score}")