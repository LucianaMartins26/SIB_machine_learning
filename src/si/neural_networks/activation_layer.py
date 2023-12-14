import numpy as np

from abc import abstractmethod

from SIB_machine_learning.src.si.neural_networks.layers import Layer


class ActivationLayer(Layer):

    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        if training is True:
            self.input = input
            self.output = self.activation_function(input)

            return self.output
        else:
            return input

    def backward_propagation(self, output_error: float):
        return self.derivate(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def output_shape(self) -> tuple:
        return self._input_shape

    @abstractmethod
    def derivate(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> int:
        return 0


class SigmoidActivation(ActivationLayer):

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))

    def derivate(self, input: np.ndarray) -> np.ndarray:
        return self.activation_function(input) * (1 - self.activation_function(input))


class ReLUActivation(ActivationLayer):

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(0, input)

    def derivate(self, input: np.ndarray) -> np.ndarray:
        return np.where(input > 0, 1, 0)


class SoftmaxActivation(ActivationLayer):

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        return np.exp(input) / np.sum(np.exp(input), axis=0)

    def derivate(self, input: np.ndarray) -> np.ndarray:
        return input * (1 - input)


class TanhActivation(ActivationLayer):

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        return np.tanh(input)

    def derivate(self, input: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(input) ** 2


if __name__ == '__main__':
    input_data = np.array([[1.0, -2.0, 3.0]])

    sigmoid_activation = SigmoidActivation()
    sigmoid_output = sigmoid_activation.forward_propagation(input_data, training=True)
    sigmoid_derivative = sigmoid_activation.derivate(input_data)

    print("Sigmoid Activation Output:")
    print(sigmoid_output)
    print("Sigmoid Activation Derivative:")
    print(sigmoid_derivative)

    relu_activation = ReLUActivation()
    relu_output = relu_activation.forward_propagation(input_data, training=True)
    relu_derivative = relu_activation.derivate(input_data)

    print("\nReLU Activation Output:")
    print(relu_output)
    print("ReLU Activation Derivative:")
    print(relu_derivative)

    softmax_activation = SoftmaxActivation()
    softmax_output = softmax_activation.forward_propagation(input_data, training=True)
    softmax_derivative = softmax_activation.derivate(input_data)

    print("\nSoftmax Activation Output:")
    print(softmax_output)
    print("Softmax Activation Derivative:")
    print(softmax_derivative)

    tanh_activation = TanhActivation()
    tanh_output = tanh_activation.forward_propagation(input_data, training=True)
    tanh_derivative = tanh_activation.derivate(input_data)

    print("\nTanh Activation Output:")
    print(tanh_output)
    print("Tanh Activation Derivative:")
    print(tanh_derivative)
