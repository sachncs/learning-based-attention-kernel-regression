# `laker` — top-level package

The single public type is [`Laker`](laker.md). The rest of the package
is exposed through module-qualified names.

## Exports

| Symbol | Source module |
|--------|---------------|
| `Laker` | `laker.model` |
| `__version__` | this package |

```python
from laker import Laker
print(Laker.__module__)  # "laker.model"
```

## Public surface (single-word naming)

| Module | Public symbols |
|--------|----------------|
| `laker.backend` | `Backend` |
| `laker.check` | `Check` |
| `laker.cli` | `CLI` |
| `laker.core` | `Core` |
| `laker.corrector` | `Corrector` |
| `laker.data` | `Data` |
| `laker.distributed` | `Distributed` |
| `laker.embed` | `Embed`, `Position`, `Visual` |
| `laker.executor` | `Executor` |
| `laker.implicit` | `hypergradient` |
| `laker.kernel` | `Exact`, `Nystrom`, `Fourier`, `Neighbors`, `Grid`, `Hybrid`, `Spectrum`, `Shaper`, `exp_safe`, `exact_matvec`, `weights` |
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

## Versioning

`__version__` follows [Semantic Versioning](https://semver.org/).
The current development version is `0.5.0+local`. See
[CHANGELOG.md](https://github.com/sachncs/laker/blob/master/CHANGELOG.md)
for the full history.

## What's NOT in the public surface

- `Core.build_kernel` is renamed to `Core.kernel` (the method named
  `kernel` is shadowed by the config attribute, so `build_kernel` is
  the canonical name). This is the one place where the single-word
  rule was relaxed to avoid the attribute / method collision.
- `Stream.update` is exposed publicly via `Laker.update`, but the
  underlying helper accepts a fully positional `model` argument for
  internal use. The public method is the cleaner wrapper.
- Module-private helpers (those starting with `_`) are not part of
  the API and may change without notice.
