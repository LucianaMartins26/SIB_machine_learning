from typing import Tuple, Iterator, Callable

import numpy as np

from SIB_machine_learning.src.si.neural_networks.losses import LossFunction
from SIB_machine_learning.src.si.neural_networks.optimizers import Optimizer, SGD

from SIB_machine_learning.src.si.data.dataset import Dataset
from SIB_machine_learning.src.si.neural_networks.layers import Layer
from SIB_machine_learning.src.si.neural_networks.losses import MeanSquaredError
from SIB_machine_learning.src.si.metrics.accuracy import accuracy


class NeuralNetwork:

    def __init__(self, epochs: int = 1000, batch_size: int = 32,
                 optimizer: Optimizer = SGD(), learning_rate: float = 0.1,
                 verbose: bool = True, loss: LossFunction = MeanSquaredError(),
                 metric: Callable[[np.ndarray, np.ndarray], float] = accuracy,
                 **kwargs) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.loss = loss
        self.metric = metric
        self.kwargs = kwargs

        self.layers = []
        self.history = {}

    def add(self, layer: Layer) -> None:
        # set the input shape of the layer based on the output layer of the previous
        # layer (except for the first one for which input shape must be provided)
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())
        # initialize the layer with the optimizer (if the layer needs one)
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)
        # append the layer to self.layers
        self.layers.append(layer)

    def _get_mini_batches(self, X: np.ndarray, y: np.ndarray = None,
                          shuffle: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate mini-batches for the given data.

        Parameters
        ----------
        X: numpy.ndarray
            The feature matrix.
        y: numpy.ndarray
            The label vector.
        shuffle: bool
            Whether to shuffle the data or not.

        Returns
        -------
        Iterator[Tuple[numpy.ndarray, numpy.ndarray]]
            The mini-batches.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        assert self.batch_size <= n_samples, "Batch size cannot be greater than the number of samples"
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples - self.batch_size + 1, self.batch_size):
            if y is not None:
                yield X[indices[start:start + self.batch_size]], y[indices[start:start + self.batch_size]]
            else:
                yield X[indices[start:start + self.batch_size]], None

    def _forward_propagation(self, X: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        X: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def _backward_propagation(self, output_error: float) -> float:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def fit(self, dataset: Dataset) -> 'NeuralNetwork':
        X = dataset.X
        y = dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        for epoch in range(1, self.epochs + 1):
            # store mini-batch data for epoch loss and quality metrics calculation
            output_x_ = []
            y_ = []
            for X_batch, y_batch in self._get_mini_batches(X, y):
                # Forward propagation
                output = self._forward_propagation(X_batch, training=True)
                # Backward propagation
                error = self.loss.derivative(y_batch, output)
                self._backward_propagation(error)

                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)

            # compute loss
            loss = self.loss.loss(y_all, output_x_all)

            if self.metric is not None:
                metric = self.metric(y_all, output_x_all)
                metric_s = f"{self.metric.__name__}: {metric / len(y_all):.4f}"
            else:
                metric_s = "NA"
                metric = 'NA'

            # save loss and metric for each epoch
            self.history[epoch] = {'loss': loss, 'metric': metric}

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {np.mean(loss):.4f} - {metric_s}")

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the labels for the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict.

        Returns
        -------
        numpy.ndarray
            The predicted labels.
        """
        return self._forward_propagation(dataset.X, training=False)

    def score(self, dataset: Dataset) -> float:
        """
        Compute the score of the neural network on the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to score.

        Returns
        -------
        float
            The score of the neural network.
        """
        if self.metric is not None:
            return self.metric(dataset.y, self.predict(dataset))
        else:
            raise ValueError("No metric specified for the neural network.")


if __name__ == '__main__':
    from SIB_machine_learning.src.si.data.dataset import Dataset
    from SIB_machine_learning.src.si.metrics.accuracy import accuracy
    from SIB_machine_learning.src.si.neural_networks.optimizers import SGD
    from SIB_machine_learning.src.si.neural_networks.losses import BinaryCrossEntropy
    from SIB_machine_learning.src.si.neural_networks.layers import DenseLayer
    from SIB_machine_learning.src.si.neural_networks.activation_layer import SigmoidActivation, ReLUActivation

    np.random.seed(42)
    X_train = np.random.rand(1000, 32)
    y_train = np.random.randint(0, 2, size=(1000, 1))

    # Create a random test dataset
    X_test = np.random.rand(200, 32)
    y_test = np.random.randint(0, 2, size=(200, 1))

    # Create a dataset
    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)

    # Build the neural network
    model = NeuralNetwork(epochs=100, batch_size=16, optimizer=SGD(learning_rate=0.01),
                          loss=BinaryCrossEntropy(), metric=accuracy)

    # Add layers to the model
    model.add(DenseLayer(n_units=16, input_shape=(32,)))
    model.add(ReLUActivation())
    model.add(DenseLayer(n_units=8))
    model.add(ReLUActivation())
    model.add(DenseLayer(n_units=4))
    model.add(ReLUActivation())
    model.add(DenseLayer(n_units=1))
    model.add(SigmoidActivation())

    # Train the neural network
    model.fit(train_dataset)

    train_accuracy = model.score(train_dataset)
    print(f"Training Accuracy: {train_accuracy:.4f}")

    # Evaluate on the test set
    test_accuracy = model.score(test_dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")