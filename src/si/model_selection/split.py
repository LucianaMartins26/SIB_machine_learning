from SIB_machine_learning.src.si.data.dataset import Dataset
import numpy as np


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42):
    np.random.seed(random_state)

    n_samples = dataset.shape()[0]

    n_test = int(n_samples * test_size)

    permutations = np.random.permutation(n_samples)

    test_indices = permutations[:n_test]
    train_indices = permutations[n_test:]

    train = Dataset(X=dataset.X[train_indices], y=dataset.y[train_indices], features=dataset.features,
                    label=dataset.label)
    test = Dataset(X=dataset.X[test_indices], y=dataset.y[test_indices], features=dataset.features, label=dataset.label)

    return train, test


def stratified_train_test_split(dataset: Dataset, test_size=0.2, random_state=42):
    unique_labels, label_counts = np.unique(dataset.y, return_counts=True)

    train_indices = []
    test_indices = []

    np.random.seed(random_state)

    label_index_mapping = {label: i for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        label_index = label_index_mapping[label]
        num_test_samples = int(label_counts[label_index] * test_size)

        indices = np.where(dataset.y == label)[0]

        np.random.shuffle(indices)

        test_indices.extend(indices[:num_test_samples])

        train_indices.extend(indices[num_test_samples:])

    X_train = dataset.X[train_indices]
    y_train = dataset.y[train_indices]
    X_test = dataset.X[test_indices]
    y_test = dataset.y[test_indices]

    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)

    return train_dataset, test_dataset
