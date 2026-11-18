"""Train and evaluate the ten-output MNIST classifier."""

import argparse

import numpy as np

from src.data import load_split, validation_split
from src.experiment import BATCH_SIZE, SEED, destination, run, tune, write_csv, write_json
from src.training import squared_errors


OUTPUT_PRIOR = 0.1
DEFAULT_CANDIDATES = [
    {"hidden": [width], "epochs": 200, "learning_rate": rate, "momentum": 0.9,
     "output_prior": OUTPUT_PRIOR}
    for width, rate in [(120, 0.08), (160, 0.14), (200, 0.20)]
]



def targets(digits):
    return np.eye(10, dtype=np.float32)[digits]


def metrics(model, data):
    outputs = model.predict(data.images)
    return {
        "error": float(np.mean(outputs.argmax(axis=1) != data.digits)),
        "loss": float(np.mean(squared_errors(outputs, targets(data.digits)))),
    }


def export(model, split, directory, name):
    outputs = model.predict(split.images)
    predictions = outputs.argmax(axis=1)
    errors = predictions != split.digits
    losses = squared_errors(outputs, targets(split.digits))
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (split.digits, predictions), 1)
    np.savetxt(directory / f"{name}_confusion.csv", confusion, fmt="%d", delimiter=",")
    np.savez_compressed(directory / f"{name}_outputs.npz", indices=split.indices,
                        digits=split.digits, outputs=outputs, predictions=predictions,
                        incorrect=errors, losses=losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/mnist")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden", nargs="+", type=int, help="Use this architecture without tuning.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.14)
    parser.add_argument("--momentum", type=float, default=0.9)
    args = parser.parse_args()
    out = destination(args.output)
    training = load_split(args.data, "train", 400)
    if args.hidden is None:
        fitting, validation = validation_split(training, SEED)
        config, records = tune(
            fitting.images, targets(fitting.digits), DEFAULT_CANDIDATES,
            lambda model: {f"validation_{k}": v for k, v in metrics(model, validation).items()},
            lambda row: (row["validation_error"], row["validation_loss"], row["epoch"]),
        )
        write_json(out / "validation.json", records)
    else:
        config = {"hidden": args.hidden, "epochs": args.epochs,
                  "learning_rate": args.learning_rate, "momentum": args.momentum, "output_prior": OUTPUT_PRIOR}

    test = load_split(args.data, "test", 100)
    if np.intersect1d(training.indices, test.indices).size:
        raise ValueError("Training and test indices overlap.")

    def evaluate(model):
        return {f"{name}_{k}": v for name, split in (("train", training), ("test", test))
                for k, v in metrics(model, split).items()}

    model, history = run(training.images, targets(training.digits), config, evaluate)
    model.save(out / "model.npz")
    write_json(out / "config.json", {**config, "batch_size": BATCH_SIZE, "seed": SEED})
    write_csv(out / "history.csv", history)
    export(model, training, out, "train")
    export(model, test, out, "test")
    print(history[-1], flush=True)


if __name__ == "__main__":
    main()
