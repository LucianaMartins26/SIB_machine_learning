import numpy as np

from SIB_machine_learning.src.si.data.dataset import Dataset
from SIB_machine_learning.src.si.metrics.accuracy import accuracy
from SIB_machine_learning.src.si.model_selection.split import train_test_split
from SIB_machine_learning.src.si.models.knn_classifier import KNNClassifier
from SIB_machine_learning.src.si.models.logistic_regression import LogisticRegression


class VotingClassifier:
    """
    Ensemble classifier that uses the majority vote to predict the class labels.

    Parameters
    ----------
    models : array-like, shape = [n_models]
        Different models for the ensemble.

    Attributes
    ----------
    """

    def __init__(self, models):
        """
        Initialize the ensemble classifier.

        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.

        """
        # parameters
        self.models = models

    def fit(self, dataset: Dataset) -> 'VotingClassifier':
        """
        Fit the models according to the given training data.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : VotingClassifier
            The fitted model.
        """
        for model in self.models:
            model.fit(dataset)

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        y : array-like, shape = [n_samples]
            The predicted class labels.
        """

        # helper function
        def _get_majority_vote(pred: np.ndarray) -> int:
            """
            It returns the majority vote of the given predictions

            Parameters
            ----------
            pred: np.ndarray
                The predictions to get the majority vote of

            Returns
            -------
            majority_vote: int
                The majority vote of the given predictions
            """
            # get the most common label
            labels, counts = np.unique(pred, return_counts=True)
            return labels[np.argmax(counts)]

        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, self.predict(dataset))


if __name__ == '__main__':
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    voting = VotingClassifier([knn, lg])

    voting.fit(dataset_train)

    score = voting.score(dataset_test)
    print(f"Score: {score}")

    print(voting.predict(dataset_test))
