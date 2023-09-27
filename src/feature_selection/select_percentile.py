from typing import Callable

import numpy as np

from SIB_machine_learning.src.si.data.dataset import Dataset
from SIB_machine_learning.src.statistics.f_classification import f_classification


class SelectPercentile:
    """
    Selects features with the highest F value up to the specified percentile.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    percentile: int, default=10
        Percentile to use to select from total features

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """

    def __init__(self, score: Callable = f_classification, percentile: int = 10):
        """
        Selects features with the highest F value up to the specified percentile.

        Parameters
        ----------
        score: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: int, default=10
            Percentile to use to select from total features
        """

        self.score = score
        self.percentile = percentile
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset):
        """
        It fits SelectPercentile to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """

        self.F, self.p = self.score(dataset)
        return self

    def transform(self, dataset: Dataset):
        """
        It transforms the dataset by selecting features with the highest F value up to the specified percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the k*percentile higher scoring features.
        """

        wanted_features = (len(dataset.features))*(self.percentile/100)
        sorted_num = np.argsort(self.F)[-int(wanted_features):]
        X = dataset.X[:, sorted_num]
        features = np.array(dataset.features)[sorted_num]
        return Dataset(X=X, y=dataset.y, features=features, label=dataset.label)

    def fit_transform(self, dataset: Dataset):
        """
        It fits SelectPercentile and transforms the dataset by selecting features
        with the highest F value up to the specified percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the k highest scoring features.
        """

        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':

    data = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    select_percentile = SelectPercentile(percentile=50)
    select_percentile.fit(data)
    dataset = select_percentile.transform(data)
    print(dataset.features)
