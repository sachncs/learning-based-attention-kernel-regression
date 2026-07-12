"""LAKER: Learning-based Attention Kernel Regression.

LAKER is a PyTorch implementation of the algorithm from
Tao & Tan (2026), "Accelerating Regularized Attention Kernel Regression for
Spectrum Cartography", `arXiv:2604.25138 <https://arxiv.org/abs/2604.25138>`_.
It solves regularised attention kernel regression problems of the form

.. math::

    \\min_\\alpha \\|G \\alpha - y\\|_2^2 + \\lambda \\alpha^\\top G \\alpha

where :math:`G = \\exp(E E^\\top)` is an exponential attention kernel
induced by learned embeddings :math:`E`. The dominant computational cost is
solving the linear system :math:`(G + \\lambda I)\\alpha = y`, which LAKER
accelerates with a **learned data-dependent preconditioner** built by a
shrinkage-regularised Convex-Concave Procedure (CCCP). The preconditioner
reduces the system condition number by up to three orders of magnitude and
yields near size-independent Preconditioned Conjugate Gradient (PCG)
convergence.

The package exposes a high-level :class:`~laker.models.LAKERRegressor`
estimator with a familiar ``scikit-learn``-style API (``fit``/``predict``)
and a number of advanced features: low-rank kernel approximations (Nyström,
Random Fourier Features, sparse k-NN, SKI, spectral shaping, two-scale),
predictive variance / uncertainty quantification, mixed-precision
embeddings, automatic hyperparameter search (grid and Bayesian), online
streaming updates, end-to-end learned embeddings, multi-GPU distributed
matvec, and bilevel hyperparameter learning via implicit differentiation.

Architecture (top-level modules):

- :mod:`laker.backend`     — Device/dtype registry and tensor utilities.
- :mod:`laker.executor`    — Abstract :class:`Executor <laker.executor.Executor>`
  base for structured logging and timing.
- :mod:`laker.embeddings`  — Random-Fourier-feature + MLP
  :class:`PositionEmbedding <laker.embeddings.PositionEmbedding>`.
- :mod:`laker.data`        — Synthetic radio-field data generators.
- :mod:`laker.kernels`     — Matrix-free attention kernel operators
  (exact, Nyström, RFF, sparse k-NN, SKI, spectral, two-scale).
- :mod:`laker.preconditioner` — CCCP / adaptive preconditioners.
- :mod:`laker.solvers`     — PCG and reference linear-algebra solvers.
- :mod:`laker.core`        — Composition root that wires embeddings →
  kernels → preconditioner → PCG → prediction into a single pipeline.
- :mod:`laker.models`      — The :class:`~laker.models.LAKERRegressor`
  estimator.
- :mod:`laker.training`    — Embedding / residual-corrector / bilevel /
  uncertainty-aware training loops.
- :mod:`laker.streaming`   — Online updates, regularisation paths, and
  continuation schedules.
- :mod:`laker.search`      — Validation-based grid search and Bayesian
  optimisation over hyperparameters.
- :mod:`laker.bilevel`     — Outer-loop optimiser that uses implicit
  differentiation.
- :mod:`laker.implicit_diff` — Adjoint (hypergradient) computation
  through the PCG fixed-point.
- :mod:`laker.correctors`  — Tiny MLP residual correctors.
- :mod:`laker.persistence` — Save/load fitted estimators.
- :mod:`laker.distributed_kernels` — Multi-GPU matvec wrappers.
- :mod:`laker.utils`       — Numerical-stability helpers and the
  :class:`~laker.utils.GPSurrogate` used by Bayesian optimisation.
- :mod:`laker.visualize`   — Optional matplotlib-based plotting
  (``pip install laker[viz]``).
- :mod:`laker.benchmark`   — Benchmark harness for solver comparisons.

The default numerical configuration is single-precision
(``torch.float32``); switching to ``dtype=torch.float64`` is recommended
for the most ill-conditioned problems. The package follows the project's
:class:`~laker.executor.Executor` pattern for structured logging and the
"class + convenience wrapper" rule for public workflows
(see ``docs/patterns.md``).
"""

from laker.backend import get_default_device, set_default_device
from laker.benchmark import (
    BaselineBenchmark,
    SolverBenchmark,
    benchmark_laker_vs_baselines,
    benchmark_solver,
)
from laker.data import generate_grid, generate_radio_field
from laker.embeddings import PositionEmbedding
from laker.kernels import (
    AttentionKernelOperator,
    MonotoneSpectrumShaper,
    NystromAttentionKernelOperator,
    RandomFeatureAttentionKernelOperator,
    SKIAttentionKernelOperator,
    SparseKNNAttentionKernelOperator,
    SpectralAttentionKernelOperator,
    TwoScaleAttentionKernelOperator,
)
from laker.models import LAKERRegressor
from laker.preconditioner import CCCPPreconditioner
from laker.solvers import PreconditionedConjugateGradient
from laker.visualize import plot_convergence, plot_radio_map

__version__ = "0.4.0"

__all__ = [
    "get_default_device",
    "set_default_device",
    "BaselineBenchmark",
    "SolverBenchmark",
    "benchmark_laker_vs_baselines",
    "benchmark_solver",
    "generate_grid",
    "generate_radio_field",
    "PositionEmbedding",
    "AttentionKernelOperator",
    "MonotoneSpectrumShaper",
    "NystromAttentionKernelOperator",
    "RandomFeatureAttentionKernelOperator",
    "SparseKNNAttentionKernelOperator",
    "SKIAttentionKernelOperator",
    "SpectralAttentionKernelOperator",
    "TwoScaleAttentionKernelOperator",
    "CCCPPreconditioner",
    "PreconditionedConjugateGradient",
    "LAKERRegressor",
    "plot_convergence",
    "plot_radio_map",
]
