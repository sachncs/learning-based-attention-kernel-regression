# LAKER Package

Learning-based Attention Kernel Regression for large-scale spectrum cartography.

See the project root `README.md` and `docs/` for the canonical
documentation, examples, and API reference. The current module layout
(after the public-API rename) is documented in
[`laker/__init__.py`](../laker/__init__.py).

## Quick start

```python
import torch
from laker import Laker

torch.manual_seed(0)
locations = torch.rand(200, 2, dtype=torch.float64) * 100.0
measurements = torch.sin(locations[:, 0] / 50.0)

model = Laker(embed_dim=10, lam=1e-2, dtype=torch.float64, verbose=False)
model.fit(locations, measurements)

query = torch.rand(1000, 2, dtype=torch.float64) * 100.0
predictions = model.predict(query)
```

## Kernel approximations

The `kernel_type` argument selects the operator:

- `"exact"` — exact attention kernel (default).
- `"nystrom"` — Nyström low-rank approximation (`landmarks=k`).
- `"fourier"` — Random Fourier features (`features=k`).
- `"neighbors"` — sparse k-NN approximation (`neighbors=k`).
- `"grid"` — Structured Kernel Interpolation (`grid_size=k`).
- `"spectrum"` — Spectral shaping (`knots=k`).
- `"hybrid"` — Two-scale combined approximation.

Example:

```python
model = Laker(
    kernel_type="nystrom",
    landmarks=100,
    dtype=torch.float64,
    verbose=False,
)
model.fit(locations, measurements)
```

## Public API

The single top-level export is `Laker`. Secondary symbols live behind
their modules:

| Module | Symbols |
|--------|---------|
| `laker.backend` | `Backend` |
| `laker.check` | `Check` |
| `laker.core` | `Core` |
| `laker.corrector` | `Corrector` |
| `laker.data` | `Data` |
| `laker.distributed` | `Distributed` |
| `laker.embed` | `Embed`, `Position`, `Visual` |
| `laker.executor` | `Executor` |
| `laker.implicit` | `hypergradient` |
| `laker.kernel` | `Exact`, `Nystrom`, `Fourier`, `Neighbors`, `Grid`, `Hybrid`, `Spectrum`, `Shaper`, `exp_safe` |
| `laker.math` | `Math`, `GP`, `pdf_np`, `cdf_np` |
| `laker.plot` | `Plot` |
| `laker.prec` | `CCCP`, `Adaptive`, `apply_core` |
| `laker.search` | `Search` |
| `laker.bilevel` | `Bilevel` |
| `laker.solve` | `PCG`, `Descent`, `Jacobi`, `Report` |
| `laker.store` | `Store` |
| `laker.stream` | `Stream` |
| `laker.train` | `Trainer` |
| `laker.bench` | `Bench`, `SolveBench`, `BaseBench`, `bench`, `bench_all` |
| `laker.cli` | `CLI` |

## Design principles

- **One primary class per module**: every module exports a single
  public class; helpers exist as `@staticmethod`.
- **Module-qualified names**: secondary classes live behind their
  parent module (`laker.kernel.Nystrom`, `laker.solve.PCG`).
- **No legacy aliases**: clean-break renames between releases;
  see `CONTRIBUTING.md`.
- **No semi-private naming**: no leading-underscore modules in the
  public surface.
- **Logging over prints**: all diagnostic output goes through
  `logging`.
