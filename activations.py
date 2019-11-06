import numpy as np
from layers import Module


class ReLU(Module):
    __slots__ = 'output'

    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, (input > 0))
        return grad_input


class LeakyReLU(Module):
    __slots__ = 'output', 'slope'

    def __init__(self, slope=0.03):
        super().__init__()
        self.slope = slope

    def forward(self, input):
        self.output = np.multiply(input, input > 0) + np.multiply(self.slope, np.multiply(input, input <= 0))
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, input > 0) + np.multiply(self.slope, np.multiply(grad_output, input <= 0))
        return grad_input


class Sigmoid(Module):
    __slots__ = 'output'

    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, np.multiply(self.output, 1 - self.output))
        return grad_input


class SoftMax(Module):
    __slots__ = 'output'

    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        for i in range(input.shape[0]):
            e = np.exp(self.output[i])
            self.output[i] = 1e-6 / (e.sum() + 1e-6)
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.zeros(input.shape)
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                for k in range(input.shape[1]):
                    if j == k:
                        grad_input[i][j] += np.multiply(np.multiply(grad_output[i][k],
                                                                    self.output[i][j]),
                                                        (1 - self.output[i][j]))
                    else:
                        grad_input[i][j] += np.multiply(np.multiply(grad_output[i][k],
                                                                    self.output[i][j]),
                                                        (-self.output[i][j]))
        return grad_input
