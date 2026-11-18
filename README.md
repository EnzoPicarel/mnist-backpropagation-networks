# MNIST Backpropagation Networks

A handwritten digit classifier and an autoencoder share a configurable NumPy
network with explicit backpropagation and mini-batch descent with momentum.

Use Python 3.11 or newer. Place the normalized MNIST split in `data/mnist/`
and verify its row order and contents with `sha256sum -c data/mnist/SHA256SUMS`.

```sh
python -m pip install -r requirements.txt
python -m src.classifier --output results/classifier
python -m src.autoencoder --classifier results/classifier --output results/autoencoder
python -m unittest discover -s tests
```

Local run directories contain model checkpoints, learning curves as CSV data,
and outputs indexed by source row.
