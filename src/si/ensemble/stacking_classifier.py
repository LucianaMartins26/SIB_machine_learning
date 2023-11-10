from typing import List

import numpy as np

from SIB_machine_learning.src.si.data.dataset import Dataset
from SIB_machine_learning.src.si.metrics.accuracy import accuracy


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
