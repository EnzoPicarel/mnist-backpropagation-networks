<div align="center">
  <h3>MNIST Backpropagation Networks</h3>

  <p>
    Handwritten digit <strong>classification</strong> and <strong>reconstruction</strong>
    with a shared neural network written from scratch in NumPy.
    <br />
    <a href="#-getting-started"><strong>Quick Start »</strong></a>
  </p>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=flat&amp;logo=python&amp;logoColor=white">
    <img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?style=flat&amp;logo=numpy&amp;logoColor=white">
    <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-087F8C?style=flat">
    <img alt="ROCm" src="https://img.shields.io/badge/ROCm-ED1C24?style=flat&amp;logo=amd&amp;logoColor=white">
  </p>
</div>

## 🔍 About The Project

Two neural networks learn from the same 5,000 handwritten digits. The classifier
maps each image to one of ten digit classes. The autoencoder reconstructs the
original pixels through a hidden layer of the same width. Both use explicit
backpropagation and mini-batch gradient descent with momentum.

## 🛠 Built With

- **Python 3.11+ and NumPy** for network layers, gradients, and training.
- **Matplotlib** for learning curves, confusion matrices, and reconstruction figures.
- **AMD ROCm / rocBLAS** for the optional GPU benchmark.

## 📐 Architecture

### Technical Highlights

- **Shared simulator.** Configurable sigmoid layers support both the classifier
  (`784 → 200 → 10`) and the autoencoder (`784 → 200 → 784`).
- **Fixed split.** The same 4,000 training and 1,000 test images are used for both
  tasks. A stratified holdout of 40 training images per digit guides model
  selection. Final training uses all 4,000 training images.
- **Reproducible outputs.** Each run saves its configuration, learning history,
  weights, biases, and per-image outputs and errors keyed by original row index.

### File Organization

```text
├── src/
│   ├── network.py, training.py     # Forward pass, backpropagation, and momentum
│   ├── classifier.py, autoencoder.py
│   ├── prepare_data.py, data.py    # Restore and load the frozen split
│   └── benchmark.py, rocm.py      # CPU/GPU matrix benchmark
├── tests/                        # Forward pass, gradients, and training checks
├── data/mnist/                    # Tracked split indices and checksums
└── requirements.txt
```

## 🚀 Getting Started

Use Python 3.11 or newer and a local, class-balanced subset of 5,000 MNIST
images. Provide two text files in matching order: one row of 784 normalized
pixels per image and one digit label per row. The training commands run on CPU.

```sh
git clone https://github.com/EnzoPicarel/mnist-backpropagation-networks.git
cd mnist-backpropagation-networks
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.prepare_data /path/to/images.txt /path/to/labels.txt
sha256sum -c data/mnist/SHA256SUMS
```

The script creates a stratified split using seed 42. The checksums identify
the exact sample and row order used for the results below. Use a separate
output directory for a different sample, then pass it to both training
commands with `--data`.

## ⚡ Execution

```sh
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python -m src.classifier --output results/classifier
python -m src.autoencoder --classifier results/classifier --output results/autoencoder
```

Use a fresh output directory for each run. Each run stores its checkpoint,
configuration, metrics, and per-image outputs under `results/`.

## 📊 Results

| Task | Metric | Train | Test |
| --- | --- | ---: | ---: |
| Digit classification | Accuracy | 98.65% | 93.10% |
| Image reconstruction | Mean reconstruction error | 0.5544 | 1.0353 |

Reconstruction error is half the sum of squared pixel differences per image,
averaged over the split. These results describe the fixed 1,000-image test
split, with 100 examples per digit.

### CPU and ROCm Benchmark

Median time for one epoch plus a forward pass over 4,000 images, with FP32,
200 hidden units, and three trials on the same machine:

| Task | CPU, two threads | ROCm, Radeon 8060S |
| --- | ---: | ---: |
| Classifier | 0.068 s | 0.426 s |
| Autoencoder | 0.138 s | 0.529 s |

Outputs and updated parameters agreed within the benchmark tolerances. ROCm
timings include host/device copies, and CPU was faster for both measured tasks.
On the configured WSL system, run:

```sh
HSA_ENABLE_DXG_DETECTION=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python -m src.benchmark --hidden 200
```

## 🧪 Tests

```sh
python -m unittest discover -s tests
```

The tests check a hand-calculated forward pass, checkpoint round trips,
finite-difference gradients, momentum, and loss reduction.

## 👥 Author

- **Enzo Picarel**
