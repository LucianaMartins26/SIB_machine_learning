from abc import abstractmethod
import numpy as np


class LossFunction:

    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MeanSquaredError(LossFunction):

    def loss(self, y_true, y_pred):
        return np.mean(np.sum((y_true - y_pred) ** 2))

    def derivative(self, y_true, y_pred):
        return np.mean(2 * (y_pred - y_true))


class BinaryCrossEntropy(LossFunction):

    def loss(self, y_true, y_pred):
        return - np.mean(np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1))

    def derivative(self, y_true, y_pred):
        return (-y_true / y_pred) + ((1 - y_true) / (1 - y_pred))


class CategoricalCrossEntropy(LossFunction):

    def loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-15), axis=1)  # adicionamos 1e-15 para evitar log(0)

    def derivative(self, y_true, y_pred):
        return -y_true / (y_pred + 1e-15)  # adicionamos 1e-15 para evitar divisão por 0


if __name__ == '__main__':

    y_true = np.array([[1, 0], [0, 1], [1, 0]])
    y_pred = np.array([[0.9, 0.1], [0.3, 0.7], [0.8, 0.2]])

    mse_loss = MeanSquaredError()
    mse_result = mse_loss.loss(y_true, y_pred)
    mse_derivative = mse_loss.derivative(y_true, y_pred)

    print(f"Mean Squared Error Loss: {mse_result}")
    print(f"Mean Squared Error Derivative: {mse_derivative}")

    bce_loss = BinaryCrossEntropy()
    bce_result = bce_loss.loss(y_true, y_pred)
    bce_derivative = bce_loss.derivative(y_true, y_pred)

    print(f"Binary Cross Entropy Loss: {bce_result}")
    print(f"Binary Cross Entropy Derivative: {bce_derivative}")

    cce_loss = CategoricalCrossEntropy()
    cce_result = cce_loss.loss(y_true, y_pred)
    cce_derivative = cce_loss.derivative(y_true, y_pred)

    print(f"Categorical Cross Entropy Loss: {cce_result}")
    print(f"Categorical Cross Entropy Derivative: {cce_derivative}")
