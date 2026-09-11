# Getting started

This guide walks through installing LAKER, running a first fit/predict, and saving/loading the fitted model.

## Install

From PyPI:

```bash
pip install laker
```

From source:

```bash
git clone https://github.com/sachncs/laker
cd laker
pip install -e ".[dev,viz]"
```

The package depends only on `numpy>=1.23` and `torch>=2.0`. The
optional `viz` extra installs `matplotlib` for plotting; `dev` adds
`pytest`, `pytest-cov`, `hypothesis`, `ruff`, and `mypy`.

## A first fit

```python
from laker import Laker
import torch

torch.manual_seed(0)
n, d = 200, 2
x = torch.rand(n, d, dtype=torch.float64) * 100
y = torch.sin(x[:, 0] / 50) + torch.cos(x[:, 1] / 50)

model = Laker(embed_dim=8, lam=1e-2, dtype=torch.float64, verbose=False)
model.fit(x, y)
pred = model.predict(x)
print(f"R^2 = {model.score(x, y):.3f}")
```

The constructor takes the single-word canonical hyperparameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `embed_dim` | 10 | Output dimension of the `Position` encoder |
| `lam` | 1e-2 | Ridge weight `λ` |
| `gamma` | 0.1 | CCCP shrinkage parameter |
| `kernel` | "exact" | `exact` / `nystrom` / `fourier` / `neighbors` / `grid` / `spectrum` / `hybrid` |
| `landmarks` | None | Nyström landmark count (auto if None) |
| `features` | None | RFF feature count |
| `neighbors` | None | k-NN sparsity count |
| `grid_size` | None | SKI grid resolution |
| `blend` | 0.5 | Hybrid Nyström / Neighbors blend weight |
| `selection` | "greedy" | `greedy` or `leverage` landmark selection |
| `pilot` | 1000 | Leverage-score pilot size |
| `knots` | 5 | Spectral-kernel spline knots |
| `distributed` | False | Use multi-GPU distributed kernel |
| `num` | None | Random-probe count for the preconditioner |
| `eps` | 1e-8 | Numerical stability |
| `base` | 0.05 | Base spectral norm bound for CCCP |
| `prec_kind` | "cccp" | `cccp` or `adaptive` |
| `cccp_max` | 200 | CCCP max iterations |
| `cccp_tol` | 1e-6 | CCCP tolerance |
| `pcg_tol` | 1e-6 | PCG tolerance |
| `pcg_max` | 1000 | PCG max iterations |
| `chunk` | None | Chunk size for tiled matvec |
| `encoder` | None | Optional pre-built embedding module |
| `embed_dtype` | None | Dtype for embedding computation |
| `device` | None | Target torch device |
| `dtype` | None | Floating-point dtype |
| `verbose` | True | Log progress |
| `warm` | False | Carry fitted state across `fit` calls |

## Save and load

```python
model.save("model.pt")
restored = Laker.load("model.pt")
pred = restored.predict(x)
```

`Laker.save` writes a single `.pt` file containing the version header,
hyperparameters, fitted tensors (`coef_`, `embed_`), the kernel operator,
the preconditioner, and (optionally) the encoder and corrector state
dicts. `Laker.load` reconstructs the full model including the kernel
operator (selected by the `kernel` hyperparameter).

## Inspect fitted state

```python
print(model.coef_)         # solution vector (n,)
print(model.embed_)        # training embeddings (n, embed_dim)
print(model.kernel_)       # kernel operator
print(model.prec_)         # preconditioner
print(model.encoder_)      # encoder module
print(model.iters_)        # PCG iteration count from last fit
```

## Scoring

`Laker.score(x, y)` returns the coefficient of determination `R²` on
`(x, y)`. By sklearn convention, `1.0` is a perfect fit, `0.0` matches
the mean predictor, and negative values are worse than the mean.

## Run a script

The CLI wraps the same flow as a command:

```bash
python -m laker.cli fit --locations x.npy --measurements y.npy --output model.pt --dtype float64
python -m laker.cli predict --model model.pt --locations x.npy --output preds.pt
```

`.npy` and `.pt` files are both accepted.

## Next steps

- See [Choosing a kernel](choosing_kernel.md) to pick a kernel
  approximation.
- See [End-to-end training](training.md) for `learn` / `correct` /
  `calibrate` / `bilevel`.
- See [Streaming updates](streaming.md) for `update` / `path` /
  `continuation`.
- See [Hyperparameter search](hyperparameter_search.md) for
  `search` and `bayes`.
