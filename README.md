<p align="center">
  <h1 align="center">LAKER</h1>
  <p align="center">Learning-based Attention Kernel Regression for scalable spectrum cartography.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/laker/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/laker/ci.yml?branch=master" alt="CI"></a>
    <a href="https://pypi.org/project/laker/"><img src="https://img.shields.io/pypi/v/laker" alt="PyPI"></a>
    <a href="https://github.com/sachncs/laker/stargazers"><img src="https://img.shields.io/github/stars/sachncs/laker" alt="Stars"></a>
  </p>
</p>

**LAKER** is a PyTorch implementation of the algorithm from
Tao & Tan (2026),
[*Accelerating Regularized Attention Kernel Regression for Spectrum Cartography*](https://arxiv.org/abs/2604.25138).
It solves regularised attention kernel regression problems of the form

$$\min_\alpha \|G \alpha - y\|_2^2 + \lambda \, \alpha^\top G \alpha$$

where $G = \exp(E E^\top)$ is an exponential attention kernel induced by
learned embeddings $E$. The dominant cost is solving the linear system
$(G + \lambda I) \alpha = y$, which LAKER accelerates with a learned
data-dependent preconditioner built by a shrinkage-regularised
Convex-Concave Procedure (CCCP). The preconditioner reduces the system
condition number by up to three orders of magnitude and yields near
size-independent Preconditioned Conjugate Gradient (PCG) convergence.

---

## Features

- **Scalable to 100k+ samples** — Matrix-free attention kernel with
  adaptive 1-D/2-D tiling and an optional `exact` mode for small problems.
- **Low-rank kernel approximations** — Nyström, random Fourier features
  (RFF), sparse k-NN, structured kernel interpolation (SKI), spectral
  shaping, and a two-scale hybrid reduce matvec cost from `O(n^2)` to
  `O(n * r)`.
- **Learned preconditioner** — Factored CCCP preconditioner with `O(N_r^3)`
  per-iteration cost independent of problem size, plus an adaptive
  strategy selector (Jacobi / CCCP / aggressive CCCP).
- **Predictive variance** — Exact posterior variance via batched PCG;
  closed-form for RFF via the Woodbury identity.
- **Mixed precision** — Compute embeddings in `float16` / `bfloat16`
  while keeping the solver in `float32` / `float64`.
- **Hyperparameter search** — Validation-based grid search and Bayesian
  optimisation with a lightweight GP surrogate.
- **Streaming / online learning** — `update` with warm-start and optional
  preconditioner rebuild; regularisation paths and continuation schedules.
- **Learned embeddings** — End-to-end optimisation of the `Position` /
  `Visual` encoders via backprop through the kernel operator.
- **Multi-GPU distributed matvec** — Shards embeddings across CUDA
  devices and gathers results automatically.
- **Bilevel hyperparameter learning** — Implicit differentiation through
  the PCG fixed point for joint optimisation of `lam` and embeddings.
- **Uncertainty-aware training** — NLL + calibration penalty objective
  for well-calibrated predictive variances.
- **Residual corrector** — Tiny MLP that captures local misspecification
  without destabilising the core solver.
- **sklearn-compatible API** — `fit` / `predict` / `score` with
  `get_params` / `set_params` and `__sklearn_clone__` for use with
  scikit-learn meta-estimators.
- **Reproducible real-world + paper experiments** — `examples.scalable`
  drives a full kernel sweep on the 50,000-map UCF-50K corpus and
  validates the winner on the complete masked 256×256 grid over all
  50,000 maps (resumable, parallel); `examples.paper` reproduces the
  LAKER paper's Section V numerical experiment on the paper's
  synthetic scene in ~2 minutes.

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

### Optional: visualization

```bash
pip install -e ".[viz]"
```

**Requirements**: Python 3.10 – 3.13, PyTorch ≥ 2.0, NumPy ≥ 1.23.

---

## Quick Start

### CLI

```bash
laker fit --locations x_train.pt --measurements y_train.pt --output model.pt
laker predict --model model.pt --locations x_test.pt --output y_pred.pt
```

The full flag list is available via `laker fit --help`.

### Python API

```python
import torch
from laker import Laker

n = 1000
x_train = torch.rand(n, 2) * 100.0
y_train = torch.randn(n)

model = Laker(
    embed_dim=10,
    lam=1e-2,
    gamma=1e-1,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
model.fit(x_train, y_train)

x_test = torch.rand(2000, 2) * 100.0
y_pred = model.predict(x_test)
```

`Laker` exposes the full sklearn API: `fit`, `predict`, `score`,
`variance`, `condition`, and the workflows `search`, `bayes`, `update`,
`path`, `continuation`, `learn`, `correct`, `bilevel`, `calibrate`,
`tune`. Fitted state is on `m.coef`, `m.embed`, `m.kernel`, `m.prec`,
`m.encoder`, `m.inputs`, `m.targets`, `m.iters`.

---

## Configuration

### Core parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | `10` | Dimension of the embedding space |
| `lam` | `1e-2` | Ridge weight `λ` |
| `gamma` | `0.1` | Kernel bandwidth for the CCCP preconditioner |
| `num` | `None` | Random-probe count for preconditioner construction |
| `eps` | `1e-8` | Numerical stability constant |
| `base` | `0.05` | Base spectral norm bound for CCCP |
| `pcg_tol` | `1e-6` | PCG relative residual tolerance |
| `pcg_max` | `1000` | Maximum PCG iterations |
| `cccp_max` | `200` | Maximum CCCP iterations |
| `cccp_tol` | `1e-6` | CCCP convergence tolerance |
| `chunk` | `None` | Tile size for chunked kernel evaluation |
| `device` | `None` | PyTorch device (`"cpu"`, `"cuda"`, `"mps"`) |
| `dtype` | `None` | Floating-point dtype for the solver |
| `embed_dtype` | `None` | Dtype for embedding computation (defaults to `dtype`) |
| `verbose` | `True` | Whether to log diagnostics |
| `warm` | `False` | Carry fitted state across `fit` calls |

### Kernel approximation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_type` | `"exact"` | `"exact"`, `"nystrom"`, `"fourier"`, `"neighbors"`, `"grid"`, `"spectrum"`, `"hybrid"` |
| `landmarks` | `None` | Nyström landmark count |
| `features` | `None` | Random Fourier features for the Fourier kernel |
| `neighbors` | `None` | k-NN sparsity count for the sparse kernel |
| `grid_size` | `None` | SKI grid resolution |
| `blend` | `0.5` | Hybrid kernel blend weight in `[0, 1]` |
| `selection` | `"greedy"` | `"greedy"` or `"leverage"` landmark selection |
| `pilot` | `1000` | Leverage-score pilot size |
| `knots` | `5` | Spline knots for the spectrum kernel |
| `distributed` | `False` | Use multi-device distributed kernel |

### Preconditioner

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prec_kind` | `"cccp"` | `"cccp"` or `"adaptive"` strategy |

### Custom embedding

Pass any `torch.nn.Module` as `encoder` to replace the default positional
embedding. The module must accept a `(n, d)` tensor and return a
`(n, embed_dim)` tensor.

See [docs/guides/](docs/guides/) for detailed configuration.

---

## Public API

The single public entry point is `laker.Laker`. The rest of the package
exposes the building blocks under single-word names:

| Module | Public classes |
|--------|----------------|
| `laker.backend` | `Backend` |
| `laker.check` | `Check` |
| `laker.corrector` | `Corrector` |
| `laker.data` | `Data` |
| `laker.embed` | `Embed`, `Position`, `Visual` |
| `laker.math` | `Math`, `GP`, `pdf_np`, `cdf_np` |
| `laker.core` | `Core` |
| `laker.kernel` | `Exact`, `Nystrom`, `Fourier`, `Neighbors`, `Grid`, `Hybrid`, `Spectrum`, `Shaper` |
| `laker.solve` | `PCG`, `Descent`, `Jacobi`, `Report` |
| `laker.prec` | `CCCP`, `Adaptive` |
| `laker.distributed` | `Distributed` |
| `laker.search` | `Search` |
| `laker.bilevel` | `Bilevel` |
| `laker.implicit` | `hypergradient` |
| `laker.train` | `Trainer` |
| `laker.stream` | `Stream` |
| `laker.store` | `Store` |
| `laker.plot` | `Plot` |
| `laker.bench` | `Bench`, `BaseBench`, `SolveBench` |
| `laker.cli` | `CLI` |

Per-module documentation lives under [docs/api/](docs/api/). Algorithm
notes live under [docs/algorithms/](docs/algorithms/). The full
documentation site is published at
<https://sachncs.github.io/laker/>.

---

## Examples

End-to-end worked examples under [`examples/`](examples/):

```bash
python examples/simple.py     # minimal fit / predict
python examples/learn.py      # end-to-end with learned embeddings
python examples/scale.py      # scaling with sample size
python examples/flow.py       # streaming updates
python examples/tune.py       # hyperparameter tuning
python examples/map.py        # radio-map visualisation (requires [viz])
python -m examples.scalable   # real-world UCF-50K full-sweep experiment
python -m examples.paper      # reproduce the paper's Section V numerical experiment
```

The real-world example (`scalable`) downloads ~10 GB of ray-traced
spectrum cartography maps and runs a full kernel sweep with
reproducibility artifacts (event log, manifests, resumable
full-corpus validation); `paper` reproduces the LAKER paper's
Section V numerical experiment on the paper's synthetic scene.

Headline result on the full UCF-50K corpus (50,000 maps, complete
masked 256×256 grid, run `20260801T114917Z`): the winner
`nystrom_m100, λ=1e-2` reaches masked RMSE **10.38 ± 0.91 dB**
(median 10.30), about **50 % below the per-scene mean baseline**
(20.73 dB) and matching the exact dense solve within ~5 %. A
stationary cross-map prior (per-pixel mean over the 40,000 training
maps) measures **20.73 ± 1.50 dB** on the same 50,000 maps — i.e.
the per-scene conditioning, not cross-map learning, is what does
the work. For a fuller positioning against learned CNN baselines
see [docs/examples/scalable.md](docs/examples/scalable.md).

The benchmark suite reproduces the paper's headline numbers:

```bash
python -m benchmarks.reproducible   # full reproducible benchmark
python -m benchmarks.baseline       # pre-optimisation vs current
python -m benchmarks.approximations # kernel approximation speed
```

---

## Project Structure

```
laker/
├── laker/                     # Main package
│   ├── __init__.py            # Public API: `Laker`
│   ├── cli.py                 # CLI entry point
│   ├── model.py               # `Laker` estimator (sklearn-compatible API)
│   ├── core.py                # emb → kernel → prec → solve → predict
│   ├── backend.py             # device / dtype / compile / seed
│   ├── check.py               # input validation and tensor coercion
│   ├── data.py                # synthetic radio-field generation
│   ├── embed.py               # `Position`, `Visual` encoders
│   ├── math.py                # `Math` helpers, `GP` Bayesian surrogate
│   ├── kernel.py              # `Exact`, `Nystrom`, `Fourier`, ...,
│   │                          # `Neighbors`, `Grid`, `Hybrid`, `Spectrum`
│   ├── solve.py               # `PCG`, `Descent`, `Jacobi`
│   ├── prec.py                # `CCCP`, `Adaptive`
│   ├── distributed.py         # multi-GPU wrapper
│   ├── search.py              # grid + Bayesian search
│   ├── train.py               # `Trainer` (learn, correct, calibrate)
│   ├── bilevel.py             # implicit-diff hyperparameter learning
│   ├── implicit.py            # `hypergradient` adjoint
│   ├── corrector.py           # residual MLP
│   ├── stream.py              # `Stream` update + path + continuation
│   ├── store.py               # save / load
│   ├── plot.py                # radio-map + convergence plots
│   ├── bench.py               # benchmark harness
│   └── executor.py            # async execution helpers
├── tests/                     # Test suite (23 files, 310 tests)
├── examples/                  # Worked examples
├── benchmarks/                # Benchmark suite
├── docs/                      # API + algorithm + guide docs
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
pytest tests/

# Run tests with coverage
pytest tests/ --cov=laker

# Lint
ruff check laker/ tests/ examples/ benchmarks/

# Format
ruff format laker/ tests/ examples/ benchmarks/

# Type check
mypy laker/
```

### Code style

- Line length: 100
- Linter / formatter: `ruff`
- Type hints throughout; `mypy` runs in CI
- Google-style docstrings with `Args` / `Returns` / `Raises` / `Examples`
- Single-word public names — no leading or trailing underscores on
  identifiers anywhere in the repo

### Commit conventions

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

## Choosing a kernel

The trade-off between the low-rank and exact kernels is summarised
below. The thresholds come from the UCF-50K sweep under
`outputs/scalable/`.

| Regime (n) | Default `kernel_type` | Why |
|------------|----------------------|-----|
| `n ≤ 5_000` | `exact` | Memory ≈ 100 MB at float32; the exact path is fastest and most accurate. |
| `5_000 < n ≤ 50_000` | `nystrom` with `landmarks ≈ 0.1 * n` | Low-rank matvec; matches `exact` to within 5 % relative error on UCF-50K. |
| `n > 50_000` | `fourier` with `features ≈ 2_000` | Cheaper than Nyström at very large n; some accuracy loss on fast-growing exponential kernels. |
| `embed_dim ≤ 4` only | `grid` (SKI) | Product grid is only practical in low dimensions. |

Cross-reference: the headline `10.38 ± 0.91 dB` UCF-50K number in the
[Headline result](#headline-result) section was produced with
`kernel_type="nystrom"` and `landmarks=100` against the pinned
snapshot recorded in `data/ucf50k/MANIFEST.json`.

---

## Limitations

1. **PCG may not converge within `pcg_max`.** On very ill-conditioned
   problems or with `float32`, the solver may hit the iteration cap.
   Switching to `dtype=torch.float64` and `pcg_tol=1e-10` usually fixes
   this at a ~2× runtime cost.

2. **`float32` trades accuracy for speed.** The default path is suitable
   for most ML workloads but can struggle when `lam < 1e-4` or when the
   kernel matrix has entries near the `float32` dynamic range.

3. **Low-rank approximations are rough for exponential kernels.** Nyström
   and RFF reduce matvec cost but can have high relative error on the
   fast-growing exponential kernel. They are best used for very large
   `n` where exact evaluation is infeasible, or when speed dominates
   accuracy. See *Choosing a kernel* above for the regime that each
   kernel handles well.

4. **SKI grid grows exponentially with `embed_dim`.** Because SKI builds
   a product grid in the embedding space, the grid size scales as
   `per_dim ** embed_dim`. For `embed_dim > 10`, the grid becomes
   impractical; use Nyström or RFF instead.

5. **Custom encoders must be importable for save / load.** If you pass a
   custom `encoder` to `Laker`, the module and class must be importable
   when calling `Laker.load()`.

---

## Citation

If you use LAKER in your research, please cite:

```bibtex
@article{tao2026laker,
  title  = {Accelerating Regularized Attention Kernel Regression for Spectrum Cartography},
  author = {Tao, Liping and Tan, Chee Wei},
  year   = {2026},
  journal= {arXiv preprint arXiv:2604.25138},
  url    = {https://arxiv.org/abs/2604.25138}
}
```

---

## License

[MIT](LICENSE) © 2026 LAKER Contributors
