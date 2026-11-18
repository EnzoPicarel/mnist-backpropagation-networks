"""Compare NumPy and ROCm for identical FP32 forward and training workloads."""

import argparse
from statistics import median
from time import perf_counter

import numpy as np

from src.classifier import targets
from src.data import load_split
from src.network import Network
from src.rocm import Rocblas
from src.training import Momentum


def trial(inputs, expected, width, multiply):
    model = Network([784, width, expected.shape[1]], seed=42)
    optimizer = Momentum(model, 0.12, 0.9, multiply)
    start = perf_counter()
    for offset in range(0, len(inputs), 64):
        batch = slice(offset, offset + 64)
        optimizer.step(inputs[batch], expected[batch])
    outputs = model.predict(inputs, multiply)
    return perf_counter() - start, outputs, model.weights + model.biases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/mnist")
    parser.add_argument("--hidden", type=int, default=160)
    args = parser.parse_args()
    data = load_split(args.data, "train", 400)
    with Rocblas() as gpu:
        for name, expected in (("classifier", targets(data.digits)), ("autoencoder", data.images)):
            trial(data.images[:64], expected[:64], args.hidden, gpu)
            cpu_times, gpu_times = [], []
            for _ in range(3):
                cpu_time, cpu, cpu_parameters = trial(data.images, expected, args.hidden, np.matmul)
                gpu_time, device, gpu_parameters = trial(data.images, expected, args.hidden, gpu)
                np.testing.assert_allclose(device, cpu, rtol=1e-4, atol=2e-5)
                for cpu_value, gpu_value in zip(cpu_parameters, gpu_parameters, strict=True):
                    np.testing.assert_allclose(gpu_value, cpu_value, rtol=1e-4, atol=2e-5)
                cpu_times.append(cpu_time)
                gpu_times.append(gpu_time)
            print(f"{name}: CPU {median(cpu_times):.3f}s, ROCm {median(gpu_times):.3f}s "
                  f"({median(cpu_times) / median(gpu_times):.2f}x), FP32 parity passed")


if __name__ == "__main__":
    main()
