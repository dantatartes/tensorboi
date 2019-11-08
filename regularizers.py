from layers import Module
import numpy as np


class Dropout(Module):
    __slots__ = 'p', 'mask', '_train', 'grad_input'

    def __init__(self, p=0.5):
        super().__init__()

        self.p = p
        self.mask = None

    def forward(self, input):
        if self._train:
            self.mask = np.random.binomial(1, self.p, input.shape[1]) / self.p
            self.output = np.array([np.multiply(input[i], self.mask) for i in range(input.shape[0])])
        else:
            self.output = self.p * input
        return self.output

    def backward(self, input, grad_output):
        if self._train:
            self.grad_input = np.multiply(grad_output, self.mask)
        else:
            self.grad_input = self.p * grad_output
        return self.grad_input


class BatchNorm(Module):
    __slots__ = 'gamma', 'mu', 'sigma'

    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, input):
        if self._train:
            self.mu = np.mean(input, axis=1, keepdims=True)
            self.sigma = np.var(input, axis=1, keepdims=True)
            input_norm = (input - self.mu) / np.sqrt(self.sigma + 1e-9)
            self.output = self.gamma * input_norm
        else:
            self.output = input
        return self.output

    def backward(self, input, grad_output):
        if self._train:
            n, d = input.shape
            input_mu = input - self.mu
            std_inv = 1. / np.sqrt(self.sigma + 1e-8)

            grad_input_norm = grad_output * self.gamma
            grad_sigma = np.sum(grad_input_norm * input_mu, axis=0) * -.5 * std_inv ** 3
            grad_mu = np.sum(grad_input_norm * -std_inv, axis=0) + grad_sigma * np.mean(-2. * input_mu, axis=0)

            grad_input = (grad_input_norm * std_inv) + (grad_sigma * 2 * input_mu / n) + (grad_mu / n)
        else:
            grad_input = grad_output

        return grad_input

