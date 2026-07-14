<p align="center">
  <h1 align="center">LAKER</h1>
  <p align="center">Learning-based Attention Kernel Regression for scalable spectrum cartography.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/laker/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/laker/ci.yml?branch=master" alt="CI"></a>
    <a href="https://pypi.org/project/laker/"><img src="https://img.shields.io/pypi/v/laker" alt="PyPI"></a>
    <a href="https://github.com/sachncs/laker/stargazers"><img src="https://img.shields.io/github/stars/sachncs/laker" alt="Stars"></a>
  </p>
</p>

**LAKER** is a PyTorch implementation of the algorithm from
Tao & Tan (2026),
[*Accelerating Regularized Attention Kernel Regression for Spectrum Cartography*](https://arxiv.org/abs/2604.25138).
It solves regularised attention kernel regression using a **learned
data-dependent preconditioner** that reduces the system condition number by up
to three orders of magnitude.

> **Disclaimer:** This repository is an independent implementation of the
> LAKER algorithm. The author is not one of the paper's authors.

---

## Features

- **Scalable to 100k+ samples** — Matrix-free attention kernel with adaptive
  1-D/2-D tiling and optional explicit mode for small problems.
- **Low-rank kernel approximations** — Nyström, random Fourier features (RFF),
  sparse k-NN, SKI, spectral shaping, and two-scale kernels reduce matvec cost
  from `O(n^2)` to `O(n*r)`.
- **Learned preconditioner** — Factored CCCP preconditioner with `O(N_r^3)`
  iterations independent of problem size; adaptive strategy selection.
- **Predictive variance** — Exact variance via batched PCG; closed-form for RFF
  via the Woodbury identity.
- **Mixed-precision training** — Compute embeddings in `float16`/`bfloat16`
  while keeping the solver in `float32`/`float64`.
- **Automatic hyperparameter search** — Validation-based grid search and
  Bayesian optimization with a lightweight GP surrogate.
- **Streaming / online learning** — `partial_fit` with warm-start and optional
  preconditioner rebuild; regularization paths and continuation schedules.
- **Learned embeddings** — End-to-end optimization of `PositionEmbedding` MLP
  weights via backprop through the kernel operator.
- **Multi-GPU distributed matvec** — Shards embeddings across CUDA devices
  and gathers results automatically.
- **Bilevel hyperparameter learning** — Implicit differentiation through the
  PCG fixed-point for joint optimization of `lambda_reg` and embeddings.
- **Uncertainty-aware training** — NLL + calibration penalty objective for
  well-calibrated predictive variances.
- **Residual corrector** — Tiny MLP captures local misspecification without
  destabilising the core solver.
- **sklearn-compatible API** — `fit`/`predict`/`score` with `GridSearchCV`
  and `Pipeline` support.

---

## Installation

### From PyPI

```bash
pip install laker
```

### From source

```bash
git clone https://github.com/sachncs/laker.git
cd laker
pip install -e .
```

### With dev dependencies

```bash
pip install -e ".[dev]"
```

### Optional: Visualization

```bash
pip install -e ".[viz]"
```

**Requirements**: Python >= 3.9, PyTorch >= 2.0, NumPy >= 1.23

---

## Quick Start

### CLI

```bash
laker fit --locations x_train.pt --measurements y_train.pt --output model.pt
laker predict --model model.pt --locations x_test.pt --output y_pred.pt
```

### Python API

```python
import torch
from laker import LAKERRegressor

n = 1000
x_train = torch.rand(n, 2) * 100.0
y_train = torch.randn(n)

model = LAKERRegressor(
    embedding_dim=10,
    lambda_reg=1e-2,
    gamma=1e-1,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
model.fit(x_train, y_train)

x_test = torch.rand(2000, 2) * 100.0
y_pred = model.predict(x_test)
print(f"R^2 score: {model.score(x_test, y_test):.4f}")
```

---

## Configuration

### Core Parameters

| Parameter | Env Variable | Default | Description |
|-----------|--------------|---------|-------------|
| `embedding_dim` | — | 10 | Dimension of the embedding space |
| `lambda_reg` | — | 1e-2 | Ridge regularization weight |
| `gamma` | — | 0.1 | Kernel bandwidth for CCCP preconditioner |
| `num_probes` | — | `None` | Random probe vectors for preconditioner |
| `pcg_tol` | — | 1e-6 | PCG relative residual tolerance |
| `pcg_max_iter` | — | 1000 | Maximum PCG iterations |
| `cccp_max_iter` | — | 200 | Maximum CCCP iterations |
| `device` | — | `None` | PyTorch device (`"cpu"`, `"cuda"`, `"mps"`) |
| `dtype` | — | `float32` | Floating-point dtype for the solver |

### Kernel Approximation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_approx` | `None` | `None` (exact), `"nystrom"`, `"rff"`, `"knn"`, `"ski"`, `"spectral"`, `"twoscale"` |
| `num_landmarks` | `None` | Landmarks for Nyström / two-scale kernels |
| `num_features` | `None` | Random Fourier features for RFF kernel |
| `k_neighbors` | `None` | Nearest neighbours for sparse k-NN kernel |
| `grid_size` | `None` | Grid resolution for SKI kernel |
| `twoscale_alpha` | 0.5 | Blending weight for two-scale kernel |
| `landmark_method` | `"greedy"` | `"greedy"` or `"leverage"` landmark selection |
| `spectral_knots` | 5 | Spline knots for spectral kernel |

### Preconditioner

| Parameter | Default | Description |
|-----------|---------|-------------|
| `preconditioner` | `"cccp"` | `"cccp"` or `"adaptive"` strategy |
| `epsilon` | 1e-8 | Numerical stability constant |
| `base_rho` | 0.05 | Base spectral norm bound for CCCP |

### Training (Learned Embeddings / Bilevel)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate for embedding optimization |
| `epochs` | 50 | Training epochs |
| `rebuild_freq` | 10 | Preconditioner rebuild frequency |
| `patience` | 5 | Early stopping patience |
| `beta` | 0.1 | Calibration penalty weight (uncertainty-aware) |

See [docs/](docs/) for detailed configuration options.

---

## API

| Symbol | Type | Description |
|--------|------|-------------|
| `LAKERRegressor` | class | sklearn-compatible estimator (`fit`/`predict`/`score`) |
| `PositionEmbedding` | class | Embedding module (random Fourier features + MLP) |
| `SearchService` | class | Grid and Bayesian hyperparameter search |
| `StreamingService` | class | `partial_fit` with warm-start and rebuild |
| `ResidualCorrector` | class | Small MLP for local misspecification |
| `DistributedMatvec` | class | Multi-GPU sharded matrix-free matvec |
| `BilevelLearner` | class | Implicit-differentiation bilevel optimiser |
| `save` / `load` | function | Model serialisation to / from disk |

---

## Examples

The package ships with end-to-end worked examples under [`examples/`](examples/):

```bash
# Reconstruct a synthetic radio field
python examples/radio_field.py --n 2000 --embedding-dim 10

# Run the bilevel lambda / embedding optimization
python examples/bilevel.py --epochs 50 --lr 1e-3

# Distributed multi-GPU matvec sanity check
python examples/distributed_matvec.py --devices 0,1
```

A full reproduction of the paper's Table 5 fit lives in
[`benchmarks/reproducible.py`](benchmarks/reproducible.py).

---

## Project Structure

```
laker/
├── laker/                     # Main package
│   ├── __init__.py            # Public API and version
│   ├── __main__.py            # CLI interface (laker fit, laker predict)
│   ├── models.py              # LAKERRegressor (sklearn-compatible API)
│   ├── core.py                # Core pipeline: embeddings, kernels, solvers
│   ├── kernels.py             # All kernel operators (exact, Nyström, RFF, etc.)
│   ├── preconditioner.py      # CCCP and adaptive preconditioners
│   ├── solvers.py             # PCG solver and baselines
│   ├── training.py            # Embedding training, residual correctors
│   ├── embeddings.py          # PositionEmbedding (random Fourier features + MLP)
│   ├── search.py              # Grid search and Bayesian optimization
│   ├── streaming.py           # Partial fit, regularization path, continuation
│   ├── distributed_kernels.py # Multi-GPU distributed matvec
│   ├── data.py                # Synthetic radio field generators
│   ├── visualize.py           # Radio map and convergence plotting
│   ├── benchmark.py           # Benchmarking utilities
│   ├── persistence.py         # Save/load models
│   ├── utils.py               # Numerical stability helpers
│   ├── backend.py             # Device/dtype management
│   ├── bilevel.py             # Bilevel hyperparameter learning
│   ├── implicit_diff.py       # Adjoint method for hypergradients
│   ├── correctors.py          # ResidualCorrector MLP
│   └── executor.py            # Abstract Executor base class
├── tests/                     # Test suite (19 files)
├── examples/                  # Worked examples
├── benchmarks/                # Benchmark suite
├── docs/                      # Documentation
├── pyproject.toml             # Build & tool config
├── CHANGELOG.md               # Release history
└── CONTRIBUTING.md            # Contribution guidelines
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=laker

# Format
black laker/ tests/

# Sort imports
isort laker/ tests/

# Lint
flake8 laker/ tests/

# Type check
mypy laker/
```

### Running Benchmarks

```bash
python -m benchmarks.reproducible    # Full reproducible benchmark suite
python -m benchmarks.baseline        # Baseline vs optimised comparison
python -m benchmarks.approximations  # Approximation speed comparison
python -m benchmarks.run             # Legacy quick benchmarks
```

### Code Style

- Line length: 100
- Formatting: black
- Type hints: required on all public signatures
- Docstrings: Google-style with Args/Returns/Raises/Examples sections
- No semi-private naming (`_foo`) — all identifiers are public

### Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add residual-aware anchor selection
fix: handle edge case in drift computation
docs: add comprehensive docstrings across all modules
refactor: convert semi-private attributes to public API
test: add parity tests for cached vs streamed memory
chore: update ruff config
```

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=laker
```

---

## Build

```bash
python -m build
```

---

## Release

See [docs/release.md](docs/release.md) — version is bumped in `pyproject.toml`,
the changelog updated, a `vX.Y.Z` tag is cut, and the PyPI publishing workflow
publishes the source and wheel distributions.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.9+ |
| Numerical | [PyTorch](https://pytorch.org/) >= 2.0, [NumPy](https://numpy.org/) >= 1.23 |
| Machine Learning | [scikit-learn](https://scikit-learn.org/) (optional, for `GridSearchCV`) |
| Lint/Format | [black](https://github.com/psf/black), [flake8](https://flake8.pycqa.org/), [isort](https://pycqa.github.io/isort/) |
| Type Check | [mypy](https://mypy-lang.org/) |
| Testing | [pytest](https://docs.pytest.org/) + pytest-cov |
| Build | [setuptools](https://setuptools.pypa.io/) |

---

## Performance

All numbers below were measured on an Apple M3 (Darwin) with PyTorch 2.11.0,
fixed seed `42`, and 50 measurement trials for matvec. The **baseline** is the
pre-optimisation code run under identical conditions (`float64`, `pcg_tol=1e-10`).
The **optimised** column is the new code with the same settings, isolating
algorithmic changes from the float32 switch.

### Speedup vs Baseline (float64)

| Metric | Baseline | Optimised | Speedup |
|---|---|---|---|
| Kernel matvec n=5000 | 30.40 ms | 25.96 ms | **1.17x** |
| Preconditioner build n=5000 | 103.84 ms | 84.78 ms | **1.22x** |
| Full fit n=1000 | 346.98 ms | 322.38 ms | **1.08x** |

### Memory Reduction

For `n = 100,000` and `chunk_size = 8192`, peak block memory drops from
**3.2 GB** (original 1-D chunking) to **256 MB** (2-D tiling), a **12x memory
reduction**.

### Approximation Matvec Comparison (n=2000)

| Method | Mean (ms) | Speedup vs Exact |
|---|---|---|
| exact | 6.82 | 1.0x |
| nystrom | 0.04 | **170x** |
| rff | 0.09 | **76x** |
| knn | 4.31 | 1.6x |
| ski | 60.41 | 0.11x |

### How It Works

**Adaptive Tiling** — For `n <= chunk_size` or when a single chunk against the
full input fits in a 64 MB budget, we use 1-D chunking (fastest path).
Otherwise we tile over both the output and reduction dimensions, keeping peak
memory bounded.

**Factored Preconditioner** — The learned covariance `Sigma` is maintained as
`a*I + Q*C*Q^T` where `Q` is an orthonormal basis for the random-probe span.
This reduces each CCCP iteration to `O(N_r^3)` instead of `O(n^3)`.

**PCG Solver** — The solver uses standard preconditioned conjugate gradient
with explicit breakdown detection (`p^T A p <= 0`). Optional residual
replacement (`restart_freq`) can suppress round-off drift in very long float64
runs, but it is **disabled by default** because it causes catastrophic
cancellation in float32.

---

## Limitations

1. **PCG does not always converge within max_iter.** On very ill-conditioned
   problems or with `float32`, the solver may hit the iteration cap. Using
   `dtype=torch.float64` and `pcg_tol=1e-10` usually fixes this at a ~2x runtime
   cost.

2. **Default float32 trades accuracy for speed.** The float32 path is suitable
   for most ML workloads but can struggle when `lambda_reg` is very small
   (`< 1e-4`) or when the kernel matrix has entries near the float32 dynamic
   range.

3. **Low-rank approximations are rough for exponential kernels.** The Nyström
   and RFF approximations reduce matvec cost but can have high relative error on
   the fast-growing exponential kernel. They are best used for very large `n`
   where exact evaluation is infeasible, or when speed dominates accuracy.

4. **SKI grid grows exponentially with embedding_dim.** Because SKI builds a
   product grid in the embedding space, the grid size scales as `gpd^d`. For
   `embedding_dim > 10`, the grid becomes impractical; use Nyström or RFF instead.

5. **Custom embeddings must be importable for save/load.** If you pass a custom
   `embedding_module` to `LAKERRegressor`, the module and class must be importable
   when calling `LAKERRegressor.load()`.

---

## Roadmap

- [ ] Warm-start preconditioner for incremental datasets
- [ ] Distributed model parallelism (all-reduce over partial contributions)
- [ ] Improved landmark-selection heuristics for exponential kernels
- [ ] Batch prediction for multiple independent query sets
- [ ] Sparse tensor backends (`torch.sparse_csr`, `scipy.sparse`)
- [ ] GPU-accelerated Nyström landmark selection
- [ ] ONNX export for deployed models

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Pull request process
- Coding standards
- Test expectations

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

## Security

Report vulnerabilities to **sachncs@gmail.com** — see [SECURITY.md](SECURITY.md).

---

## Citation

If you use LAKER in your research, please cite:

```bibtex
@article{tao2026laker,
  title={Accelerating Regularized Attention Kernel Regression for Spectrum Cartography},
  author={Tao, Liping and Tan, Chee Wei},
  journal={arXiv preprint arXiv:2604.25138},
  year={2026}
}
```

## License

[MIT](LICENSE) © 2026 Sachin
