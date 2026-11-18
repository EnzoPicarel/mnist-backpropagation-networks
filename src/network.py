"""Feed-forward networks with an explicit sigmoid layer at each stage."""

import numpy as np


def sigmoid(values):
    """Evaluate the logistic function without overflowing for large inputs."""
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


class Network:
    def __init__(self, widths, seed=42, dtype=np.float32):
        self.widths = tuple(widths)
        if len(self.widths) < 2 or min(self.widths) < 1:
            raise ValueError("Provide at least two positive layer sizes.")
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(self.widths, self.widths[1:]):
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-bound, bound, (fan_in, fan_out)).astype(dtype))
            self.biases.append(np.zeros(fan_out, dtype=dtype))

    def activations(self, inputs, multiply=np.matmul):
        """Return the batch input and successive layer outputs."""
        values = np.asarray(inputs, dtype=self.weights[0].dtype)
        if values.ndim != 2 or values.shape[1] != self.widths[0]:
            raise ValueError(f"Expected inputs of shape (batch, {self.widths[0]}).")
        layers = [values]
        for weights, bias in zip(self.weights, self.biases, strict=True):
            values = sigmoid(multiply(values, weights) + bias)
            layers.append(values)
        return layers

    def predict(self, inputs, multiply=np.matmul):
        return self.activations(inputs, multiply)[-1]

    def save(self, path):
        arrays = {}
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            arrays[f"weights_{index}"] = weights
            arrays[f"bias_{index}"] = bias
        np.savez_compressed(path, widths=np.array(self.widths), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            network = cls(arrays["widths"], dtype=arrays["weights_0"].dtype)
            for index in range(len(network.weights)):
                network.weights[index][:] = arrays[f"weights_{index}"]
                network.biases[index][:] = arrays[f"bias_{index}"]
        return network
