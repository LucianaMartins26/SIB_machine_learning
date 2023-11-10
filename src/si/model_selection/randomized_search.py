from typing import Callable, Tuple, Dict, Any
import numpy as np

from SIB_machine_learning.src.si.data.dataset import Dataset
from SIB_machine_learning.src.si.model_selection.cross_validation import k_fold_cross_validation


def randomized_search_cv(model,
                         dataset: Dataset,
                         hyperparameter_grid: Dict[str, Tuple],
                         scoring: Callable = None,
                         cv: int = 5,
                         n_iter: int = 10) -> Dict[str, Any]:
    """
    Performs a randomized search cross validation on a model.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    hyperparameter_grid: Dict[str, Tuple]
        The hyperparameter grid to use.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross-validation folds.
    n_iter: int
        Number of hyperparameter random combinations to test.

    Returns
    -------
    results: Dict[str, Any]
        The results of the randomized search cross-validation. Includes the scores, hyperparameters,
        best hyperparameters, and best score.
    """
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    results = {'hyperparameters': [], 'scores': []}

    for _ in range(n_iter):
        hyperparameters = {param: np.random.choice(values) for param, values in hyperparameter_grid.items()}

        for parameter, value in hyperparameters.items():
            setattr(model, parameter, value)

        scores = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        results['hyperparameters'].append(hyperparameters)
        results['scores'].append(np.mean(scores))

    best_index = np.argmax(results['scores'])
    results['best_hyperparameters'] = results['hyperparameters'][best_index]
    results['best_score'] = results['scores'][best_index]

    return results
