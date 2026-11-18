"""Load the frozen MNIST split and carve validation from training only."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Split:
    images: np.ndarray
    digits: np.ndarray
    indices: np.ndarray

    def subset(self, rows):
        return Split(self.images[rows], self.digits[rows], self.indices[rows])


def load_split(directory, name, per_digit):
    directory = Path(directory)
    images = np.loadtxt(directory / f"{name}_images.txt", dtype=np.float32, ndmin=2)
    digits = np.loadtxt(directory / f"{name}_labels.txt", dtype=np.int64, ndmin=1)
    indices = np.loadtxt(directory / f"{name}_indices.txt", dtype=np.int64, ndmin=1)
    rows = 10 * per_digit
    if images.shape != (rows, 784) or digits.shape != (rows,) or indices.shape != (rows,):
        raise ValueError(f"Unexpected {name} dataset shape.")
    if not ((images >= 0) & (images <= 1)).all():
        raise ValueError(f"{name} pixels must be finite and in [0, 1].")
    if not ((digits >= 0) & (digits <= 9)).all() or np.any(np.bincount(digits) != per_digit):
        raise ValueError(f"{name} must contain {per_digit} samples of every digit.")
    if not ((indices >= 0) & (indices < 5000)).all() or len(np.unique(indices)) != rows:
        raise ValueError(f"{name} source indices must be unique and in [0, 4999].")
    return Split(images, digits, indices)


def validation_split(training, seed):
    rng = np.random.default_rng(seed)
    keep, validate = [], []
    for digit in range(10):
        rows = rng.permutation(np.flatnonzero(training.digits == digit))
        validate.extend(rows[:40])
        keep.extend(rows[40:])
    return training.subset(sorted(keep)), training.subset(sorted(validate))
