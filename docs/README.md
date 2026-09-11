# LAKER Documentation

Welcome to the LAKER (Learning-based Attention Kernel Regression) developer
and research documentation. This is the authoritative guide for
understanding, extending, and contributing to the codebase.

## What is LAKER?

LAKER is a PyTorch implementation of the algorithm from
Tao & Tan (2026), *"Accelerating Regularized Attention Kernel Regression
for Spectrum Cartography"* ([arXiv:2604.25138](https://arxiv.org/abs/2604.25138)).
It solves the regularised attention-kernel regression problem

```
min_α  ‖G α − y‖²  +  λ α^T G α
```

where `G = exp(E E^T)` is the exponential attention kernel induced by
learned embeddings `E`. The dominant cost is solving
`(G + λ I) α = y`, which LAKER accelerates with a learned data-dependent
preconditioner built by a shrinkage-regularised Convex-Concave Procedure
(CCCP).

The preconditioner reduces the system condition number by up to three
orders of magnitude and yields near size-independent Preconditioned
Conjugate Gradient (PCG) convergence.

## Quick start

```python
from laker import Laker
import torch

torch.manual_seed(0)
n, d = 200, 2
x = torch.rand(n, d, dtype=torch.float64) * 100
y = torch.sin(x[:, 0] / 50) + torch.cos(x[:, 1] / 50)

model = Laker(embed_dim=8, lam=1e-2, dtype=torch.float64)
model.fit(x, y)
pred = model.predict(x)
print(f"R^2 = {model.score(x, y):.3f}")
```

## Table of contents

### [Guides](guides/)
Step-by-step guides for common tasks.

- [Getting started](guides/getting_started.md) — install, basic fit/predict, save/load.
- [Choosing a kernel](guides/choosing_kernel.md) — `exact` / `nystrom` / `fourier` / `neighbors` / `grid` / `spectrum` / `hybrid`.
- [Hyperparameter search](guides/hyperparameter_search.md) — `search` and `bayes`.
- [Streaming updates](guides/streaming.md) — `update`, `path`, `continuation`.
- [End-to-end training](guides/training.md) — `learn`, `correct`, `calibrate`, `bilevel`.
- [Persistence and reproducibility](guides/persistence.md) — `save`, `load`, seed handling.

### [Algorithms](algorithms/)
Mathematical background for the internals.

- [Attention kernel](algorithms/attention_kernel.md) — `G = exp(E E^T)` and its spectral structure.
- [CCCP preconditioner](algorithms/cccp.md) — shrinkage-regularised Convex-Concave Procedure.
- [PCG solver](algorithms/pcg.md) — preconditioned conjugate gradient, breakdown, batching.
- [Low-rank approximations](algorithms/low_rank.md) — Nyström, RFF, sparse k-NN, SKI, hybrid, spectrum.
- [Implicit differentiation](algorithms/implicit.md) — adjoint method for hypergradients.
- [Bilevel learning](algorithms/bilevel.md) — joint optimisation of lambda and embeddings.

### [API reference](api/)
Public-symbol reference, one page per module.

| Module | Class / functions |
|--------|-------------------|
| [`laker`](api/laker.md) | `Laker` |
| [`laker.backend`](api/backend.md) | `Backend` |
| [`laker.check`](api/check.md) | `Check` |
| [`laker.cli`](api/cli.md) | `CLI` |
| [`laker.core`](api/core.md) | `Core` |
| [`laker.corrector`](api/corrector.md) | `Corrector` |
| [`laker.data`](api/data.md) | `Data` |
| [`laker.distributed`](api/distributed.md) | `Distributed` |
| [`laker.embed`](api/embed.md) | `Embed`, `Position`, `Visual` |
| [`laker.executor`](api/executor.md) | `Executor` |
| [`laker.implicit`](api/implicit.md) | `hypergradient` |
| [`laker.kernel`](api/kernel.md) | `Exact`, `Nystrom`, `Fourier`, `Neighbors`, `Grid`, `Hybrid`, `Spectrum`, `Shaper`, `exp_safe`, `exact_matvec`, `weights` |
| [`laker.math`](api/math.md) | `Math`, `GP`, `pdf_np`, `cdf_np` |
| [`laker.plot`](api/plot.md) | `Plot` |
| [`laker.prec`](api/prec.md) | `CCCP`, `Adaptive`, `apply_core` |
| [`laker.search`](api/search.md) | `Search` |
| [`laker.bilevel`](api/bilevel.md) | `Bilevel` |
| [`laker.solve`](api/solve.md) | `PCG`, `Descent`, `Jacobi`, `Report` |
| [`laker.store`](api/store.md) | `Store` |
| [`laker.stream`](api/stream.md) | `Stream` |
| [`laker.train`](api/train.md) | `Trainer` |
| [`laker.bench`](api/bench.md) | `Bench`, `SolveBench`, `BaseBench`, `bench`, `bench_all` |

### [Examples](examples/)
Runnable examples.

- [`flow.py`](../../examples/flow.py) — streaming updates.
- [`learn.py`](../../examples/learn.py) — end-to-end fit + save / load.
- [`map.py`](../../examples/map.py) — radio map reconstruction.
- [`scale.py`](../../examples/scale.py) — large-scale fit and predict.
- [`simple.py`](../../examples/simple.py) — minimal sin / cos fit.
- [`tune.py`](../../examples/tune.py) — hyperparameter search.
- [`scalable.py`](examples/scalable.md) — full reproducible sweep on the
  real-world UCF-50K corpus (downloads ~10 GB).
- [`paper.py`](examples/paper.md) — reproduces the LAKER paper's
  Section V numerical experiment on the paper's synthetic scene.
- [`cross_map_cache.py`](../../examples/cross_map_cache.py) /
  [`cross_map_score.py`](../../examples/cross_map_score.py) — build the
  per-pixel train-mean cache and score the stationary cross-map prior
  against the full UCF-50K corpus (the SOTA-positioning baseline).

### External
- [GitHub repository](https://github.com/sachncs/laker)
- [Paper on arXiv](https://arxiv.org/abs/2604.25138)
- [Issue tracker](https://github.com/sachncs/laker/issues)

## Project layout

```
laker/                  # 22 modules, single-word naming
├── __init__.py         # exports Laker, __version__
├── backend.py          # device, dtype, compile, seed configuration
├── bench.py            # benchmarking harness
├── bilevel.py          # bilevel learning
├── check.py            # input validation helpers
├── cli.py              # command-line interface
├── core.py             # pipeline composition (embed → kernel → prec → solve → predict)
├── corrector.py        # residual corrector MLP
├── data.py             # synthetic radio-field generation
├── distributed.py      # multi-GPU wrapper
├── embed.py            # Position / Visual encoders
├── executor.py         # abstract Executor pattern
├── implicit.py         # adjoint-method hypergradient
├── kernel.py           # 7 kernel operators
├── math.py             # numerical helpers + GP surrogate
├── model.py            # Laker (the public estimator)
├── plot.py             # radio-map and convergence plots
├── prec.py             # CCCP / Adaptive preconditioners
├── search.py           # grid + Bayesian hyperparameter search
├── solve.py            # PCG / Descent / Jacobi / Report
├── store.py            # save / load
├── stream.py           # streaming / path / continuation
└── train.py            # embedding / corrector / uncertainty training

tests/                  # 310 tests, real behavioral assertions
examples/               # 11 runnable scripts
benchmarks/            # 5 benchmark scripts
docs/                  # this documentation
```

## Conventions

- **Naming.** Public symbols are single-word, no underscores. Examples:
  `Laker`, `coef`, `update`.
- **State.** A fitted `Laker` exposes fitted state as plain names:
  `coef`, `embed`, `kernel`, `prec`, `encoder`, `inputs`, `targets`,
  `iters`.
- **Dtype.** The `dtype` parameter is required when the input is a
  numpy array; with a `torch.Tensor` the model uses the tensor's dtype.
- **Device.** All tensors must live on `Backend.device`. The model does
  not move tensors between devices.

## License

MIT. See [LICENSE](../../LICENSE).
