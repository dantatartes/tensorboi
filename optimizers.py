def SGD(params, gradients, lr=1e-3):
    for weights, gradient in zip(params, gradients):
        weights -= lr * gradient
