import unittest

import numpy as np

from src.network import Network
from src.training import Momentum, gradients, squared_errors, train


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.network = Network([2, 3, 2], seed=3, dtype=np.float64)
        self.inputs = np.array([[0.2, 0.8], [0.7, 0.1]])
        self.targets = np.array([[1.0, 0.0], [0.0, 1.0]])

    def loss(self, network):
        return squared_errors(network.predict(self.inputs), self.targets).mean()

    def test_gradients_match_finite_differences(self):
        weights, biases = gradients(self.network, self.inputs, self.targets)
        eps = 1e-5
        parameters = self.network.weights + self.network.biases
        for parameter, gradient in zip(parameters, weights + biases, strict=True):
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + eps
                above = self.loss(self.network)
                parameter[index] = original - eps
                below = self.loss(self.network)
                parameter[index] = original
                self.assertAlmostEqual(gradient[index], (above - below) / (2 * eps), places=8)

    def test_second_step_includes_previous_velocity(self):
        optimizer = Momentum(self.network, learning_rate=0.4, momentum=0.7)
        w = self.network.weights[0]
        start = w.copy()
        optimizer.step(self.inputs, self.targets)
        first = w.copy()
        second_gradient = gradients(self.network, self.inputs, self.targets)[0][0]
        optimizer.step(self.inputs, self.targets)
        np.testing.assert_allclose(w, first + 0.7 * (first - start) - 0.4 * second_gradient)

    def test_training_reduces_loss_and_records_epoch_zero(self):
        history = train(self.network, self.inputs, self.targets, 110, 2, 1.0, 0.4, 3,
                        lambda model: {"loss": self.loss(model)})
        self.assertEqual([row["epoch"] for row in history], list(range(0, 111, 10)))
        self.assertLess(history[-1]["loss"], history[0]["loss"] / 4)
