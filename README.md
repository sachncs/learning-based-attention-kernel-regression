# LAKER: Learning-based Attention Kernel Regression

Production-ready PyTorch implementation of the LAKER algorithm from
*[Accelerating Regularized Attention Kernel Regression for Spectrum Cartography](https://arxiv.org/html/2604.25138v1)* (Tao & Tan, 2026).

> **Disclaimer:** This repository is an independent implementation of the LAKER algorithm. I am not an author of the original paper.

LAKER solves large-scale regularized attention kernel regression problems of the form

```
min_alpha ||G alpha - y||_2^2 + lambda alpha^T G alpha
```

where `G = exp(E E^T)` is an exponential attention kernel induced by learned
embeddings `E`. The key innovation is a **learned data-dependent preconditioner**
obtained via a shrinkage-regularized Convex-Concave Procedure (CCCP), which
reduces the condition number of the system by up to three orders of magnitude
and enables near size-independent Preconditioned Conjugate Gradient (PCG)
convergence.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.23.0

## Features

- **Scalable to 100k+ samples**: Matrix-free attention kernel with chunked
dot-products and optional explicit mode for small problems.
- **GPU acceleration**: First-class PyTorch backend with CPU/CUDA/MPS support.
- **Efficient preconditioner**: Factored representation exploits fixed random-probe
structure to achieve `O(N_r^3)` CCCP iterations independent of problem size `n`.
- **Production API**: `sklearn`-compatible `LAKERRegressor` with standard
`fit`/`predict` interface.
- **Exact replicability**: Faithful implementation of Algorithm 1 from the paper
with numerically stable shrinkage and trace normalization.
- **Modular design**: Swap embeddings, kernels, solvers, and preconditioners
independently.

## Quick Start

```python
import torch
from laker import LAKERRegressor

# Sparse measurements in a 2D spatial domain
n = 1000
x_train = torch.rand(n, 2) * 100.0  # locations in [0, 100]^2
y_train = torch.randn(n)

# Fit the model
model = LAKERRegressor(
    embedding_dim=10,
    lambda_reg=1e-2,
    gamma=1e-1,
    num_probes=None,      # adaptive heuristic based on n
    cccp_max_iter=100,
    pcg_tol=1e-10,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
model.fit(x_train, y_train)

# Reconstruct radio map on a dense grid
x_test = torch.rand(2000, 2) * 100.0
y_pred = model.predict(x_test)
```

### Command-line interface

LAKER also ships with a small CLI for batch workflows:

```bash
# Fit a model from saved tensors
laker fit --locations x_train.pt --measurements y_train.pt --output model.pt

# Predict on new locations
laker predict --model model.pt --locations x_test.pt --output y_pred.pt
```

### Save and load

```python
model.save("laker_model.pt")
loaded = LAKERRegressor.load("laker_model.pt")
```

## Installation

```bash
pip install -e ".[dev]"
pytest tests/
```

## Citation

```bibtex
@article{tao2026laker,
  title={Accelerating Regularized Attention Kernel Regression for Spectrum Cartography},
  author={Tao, Liping and Tan, Chee Wei},
  journal={arXiv preprint arXiv:2604.25138},
  year={2026}
}
```
