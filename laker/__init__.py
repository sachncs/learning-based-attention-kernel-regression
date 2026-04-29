"""LAKER: Learning-based Attention Kernel Regression."""

from laker.backend import get_default_device, set_default_device
from laker.benchmark import benchmark_laker_vs_baselines
from laker.data import generate_grid, generate_radio_field
from laker.embeddings import PositionEmbedding
from laker.kernels import AttentionKernelOperator
from laker.models import LAKERRegressor
from laker.preconditioner import CCCPPreconditioner
from laker.solvers import PreconditionedConjugateGradient
from laker.visualize import plot_convergence, plot_radio_map

__version__ = "0.1.0"

__all__ = [
    "get_default_device",
    "set_default_device",
    "benchmark_laker_vs_baselines",
    "generate_grid",
    "generate_radio_field",
    "PositionEmbedding",
    "AttentionKernelOperator",
    "CCCPPreconditioner",
    "PreconditionedConjugateGradient",
    "LAKERRegressor",
    "plot_convergence",
    "plot_radio_map",
]
