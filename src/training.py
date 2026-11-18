"""Explicit squared error gradients and mini-batch SGD with momentum."""

import numpy as np


def squared_errors(outputs, targets):
    """Half the sum of squared output errors, separately for each sample."""
    return 0.5 * np.sum(np.square(outputs - targets), axis=1)


def gradients(network, inputs, targets, multiply=np.matmul):
    layers = network.activations(inputs, multiply)
    outputs = layers[-1]
    if outputs.shape != targets.shape or len(outputs) == 0:
        raise ValueError("Targets must match a nonempty batch of network outputs.")
    delta = (outputs - targets) * outputs * (1 - outputs) / len(outputs)
    weight_grads = [None] * len(network.weights)
    bias_grads = [None] * len(network.biases)
    for index in range(len(network.weights) - 1, -1, -1):
        previous = layers[index]
        weight_grads[index] = multiply(previous.T, delta)
        bias_grads[index] = delta.sum(axis=0)
        if index:
            delta = multiply(delta, network.weights[index].T) * previous * (1 - previous)
    return weight_grads, bias_grads


class Momentum:
    def __init__(self, network, learning_rate, momentum, multiply=np.matmul):
        if learning_rate <= 0 or not 0 <= momentum < 1:
            raise ValueError("Learning rate must be positive and momentum in [0, 1).")
        self.network = network
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.multiply = multiply
        self.parameters = network.weights + network.biases
        self.velocities = [np.zeros_like(parameter) for parameter in self.parameters]

    def step(self, inputs, targets):
        weights, biases = gradients(self.network, inputs, targets, self.multiply)
        for parameter, velocity, gradient in zip(
            self.parameters, self.velocities, weights + biases, strict=True
        ):
            velocity *= self.momentum
            velocity -= self.learning_rate * gradient
            parameter += velocity


def train(network, inputs, targets, epochs, batch_size, learning_rate, momentum,
          seed, evaluate, multiply=np.matmul):
    if epochs < 1 or batch_size < 1 or len(inputs) == 0 or len(inputs) != len(targets):
        raise ValueError("Expected positive epochs and batch size, and aligned training data.")
    rng = np.random.default_rng(seed)
    optimizer = Momentum(network, learning_rate, momentum, multiply)
    history = [{"epoch": 0, **evaluate(network)}]
    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(inputs))
        for offset in range(0, len(indices), batch_size):
            batch = indices[offset:offset + batch_size]
            optimizer.step(inputs[batch], targets[batch])
        if epoch % 10 == 0 or epoch == epochs:
            history.append({"epoch": epoch, **evaluate(network)})
    return history
