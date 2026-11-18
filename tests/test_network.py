from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.network import Network


class NetworkTests(unittest.TestCase):
    def test_forward_matches_hand_calculation(self):
        network = Network([1, 1, 1], dtype=np.float64)
        network.weights[0][:] = 2
        network.biases[0][:] = -2
        network.weights[1][:] = 4
        network.biases[1][:] = -1
        np.testing.assert_allclose(network.predict([[1]]), [[1 / (1 + np.exp(-1))]])

    def test_batch_shape_and_round_trip(self):
        network = Network([4, 3, 2, 5])
        inputs = np.ones((7, 4))
        self.assertEqual(network.predict(inputs).shape, (7, 5))
        with self.assertRaises(ValueError):
            network.predict([[0, 0]])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            network.save(path)
            np.testing.assert_array_equal(network.predict(inputs), Network.load(path).predict(inputs))
