"""Utilities for reproducible validation and final training."""

import csv
import json
from pathlib import Path

import numpy as np

from src.network import Network
from src.training import train

SEED = 42
BATCH_SIZE = 64


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(inputs, targets, config, evaluate, multiply=np.matmul):
    model = Network([inputs.shape[1], *config["hidden"], targets.shape[1]], seed=SEED)
    if "output_prior" in config:
        prior = config["output_prior"]
        model.biases[-1].fill(np.log(prior / (1 - prior)))
    history = train(model, inputs, targets, config["epochs"], BATCH_SIZE,
                    config["learning_rate"], config["momentum"], SEED, evaluate, multiply)
    return model, history


def tune(inputs, targets, candidates, evaluate, rank):
    records = []
    for candidate in candidates:
        _, history = run(inputs, targets, candidate, evaluate)
        best = min((row for row in history if row["epoch"] > 0), key=rank)
        config = {**candidate, "epochs": best["epoch"]}
        records.append({"config": config, "best": best, "history": history})
        print(f"{config}: {best}", flush=True)
    return min(records, key=lambda record: rank(record["best"]))["config"], records


def destination(value):
    path = Path(value)
    path.mkdir(parents=True, exist_ok=False)
    return path
