"""Create a deterministic split of a balanced MNIST text dataset."""

import argparse
from collections import Counter
from pathlib import Path
import random

import numpy as np

from src.data import load_split


def prepare(images_path, labels_path, destination):
    images = images_path.read_text(encoding="utf-8").splitlines()
    labels = list(map(int, labels_path.read_text(encoding="utf-8").split()))
    if len(images) != 5000 or Counter(labels) != {digit: 500 for digit in range(10)}:
        raise ValueError("Expected 5,000 images with 500 labels for each digit.")
    pixels = np.loadtxt(images, dtype=np.float32, ndmin=2)
    if pixels.shape != (5000, 784) or not ((pixels >= 0) & (pixels <= 1)).all():
        raise ValueError("Expected 784 finite pixels in [0, 1] per image.")

    groups = [[] for _ in range(10)]
    for index, digit in enumerate(labels):
        groups[digit].append(index)
    rng = random.Random(42)
    for group in groups:
        rng.shuffle(group)
    train = [index for group in groups for index in group[:400]]
    test = [index for group in groups for index in group[400:]]
    rng.shuffle(train)

    files = {}
    for split, indices in (("train", train), ("test", test)):
        for name, values in (
            ("indices", indices),
            ("images", [images[index] for index in indices]),
            ("labels", [labels[index] for index in indices]),
        ):
            path = destination / f"{split}_{name}.txt"
            contents = ("\n".join(map(str, values)) + "\n").encode()
            if path.exists() and path.read_bytes() != contents:
                raise ValueError(f"Refusing to replace a different file: {path}")
            files[path] = contents

    destination.mkdir(parents=True, exist_ok=True)
    for path, contents in files.items():
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(contents)
    load_split(destination, "train", 400)
    load_split(destination, "test", 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/mnist"))
    args = parser.parse_args()
    prepare(args.images, args.labels, args.output)


if __name__ == "__main__":
    main()
