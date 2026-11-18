"""Train the MNIST autoencoder with the classifier's hidden width."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import load_split, validation_split
from src.experiment import BATCH_SIZE, SEED, destination, run, tune, write_csv, write_json
from src.training import squared_errors


def reconstruction_errors(model, data):
    return squared_errors(model.predict(data.images), data.images)


def summarize_digits(digits, errors):
    results = []
    for digit in range(10):
        values = errors[digits == digit].astype(np.float64)
        mean = values.sum() / len(values)
        variance = np.square(values - mean).sum() / (len(values) - 1)
        results.append({"digit": digit, "samples": len(values),
                        "mean": float(mean), "sample_std": float(np.sqrt(variance))})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/mnist")
    parser.add_argument("--classifier", required=True, type=Path)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    classifier = json.loads((args.classifier / "config.json").read_text())
    if len(classifier["hidden"]) != 1:
        parser.error("The autoencoder requires a single classifier hidden layer.")
    hidden = classifier["hidden"]
    out = destination(args.output)
    training = load_split(args.data, "train", 400)
    fitting, validation = validation_split(training, SEED)

    candidates = [
        {"hidden": hidden, "epochs": 160, "learning_rate": rate, "momentum": momentum}
        for rate, momentum in [(0.05, 0.9), (0.15, 0.9), (0.25, 0.8)]
    ]
    config, records = tune(
        fitting.images, fitting.images, candidates,
        lambda model: {"validation_mre": float(reconstruction_errors(model, validation).mean())},
        lambda row: (row["validation_mre"], row["epoch"]),
    )
    write_json(out / "validation.json", records)
    test = load_split(args.data, "test", 100)
    if np.intersect1d(training.indices, test.indices).size:
        raise ValueError("Training and test indices overlap.")

    model, history = run(
        training.images, training.images, config,
        lambda net: {"train_mre": float(reconstruction_errors(net, training).mean())},
    )
    model.save(out / "model.npz")
    write_json(out / "config.json", {**config, "batch_size": BATCH_SIZE, "seed": SEED})
    write_csv(out / "history.csv", history)
    means = {}
    for name, split in (("train", training), ("test", test)):
        outputs = model.predict(split.images)
        errors = squared_errors(outputs, split.images)
        means[name] = float(errors.mean())
        np.savez_compressed(out / f"{name}_outputs.npz", indices=split.indices,
                            digits=split.digits, outputs=outputs, errors=errors)
        if name == "test":
            write_csv(out / "test_by_digit.csv", summarize_digits(split.digits, errors))
    write_json(out / "metrics.json", means)
    print(means, flush=True)


if __name__ == "__main__":
    main()
