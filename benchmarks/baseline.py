"""Baseline vs Optimised comparison for README.

This module compares the *current* (optimised) implementation against
recorded baseline timing numbers for the three main computational
bottlenecks of LAKER:

1. Attention kernel matrix-vector product at :math:`n = 5000`.
2. CCCP preconditioner build at :math:`n = 5000`.
3. Full :class:`~laker.models.LAKERRegressor` fit at :math:`n = 1000`.

Each configuration is run with fixed random seeds
(``torch.manual_seed(42)``) so that results are deterministic on the
same hardware and PyTorch version.  Speed-up ratios are computed
against baseline measurements collected on 2026-04-30 (Apple M3,
Python 3.13, PyTorch 2.11.0).

Run this module directly to print the comparison table::

    python -m benchmarks.baseline
"""

import logging
from typing import Optional

import torch

from benchmarks.executor import BenchmarkExecutor
from laker.kernels import AttentionKernelOperator
from laker.models import LAKERRegressor
from laker.preconditioner import CCCPPreconditioner

logger = logging.getLogger(__name__)


class BaselineComparison:
    """Compares current performance against recorded baseline numbers.

    Runs a fixed set of benchmarks (kernel matvec, preconditioner build,
    full fit) at the baseline problem sizes and computes speed-up ratios
    relative to previously recorded timings.

    Args:
        executor: Optional pre-configured
            :class:`~benchmarks.executor.BenchmarkExecutor`.  When
            ``None`` a new executor with default settings is created.
    """

    def __init__(self, executor: Optional[BenchmarkExecutor] = None):
        self.executor = executor if executor is not None else BenchmarkExecutor()

    @classmethod
    def run_comparison_default(cls, label: str, dtype: torch.dtype, pcg_tol: float) -> dict:
        """Run a single baseline comparison configuration.

        Args:
            label: Identifier for this run.
            dtype: PyTorch dtype to use.
            pcg_tol: Tolerance for PCG solver.

        Returns:
            Dictionary with benchmark results.
        """
        comparison = cls()
        return comparison.run_comparison(label, dtype, pcg_tol)

    def format_number(self, n: float) -> str:
        """Format a floating-point number to three decimal places.

        Args:
            n: The number to format.

        Returns:
            String representation with exactly three digits after the
            decimal point.
        """
        return f"{n:.3f}"

    def run_comparison(self, label: str, dtype: torch.dtype, pcg_tol: float) -> dict:
        """Run a single comparison configuration.

        Executes three benchmarks sequentially with deterministic seeds:

        1. Attention kernel matvec (:math:`n = 5000`, 50 measured
           repetitions).
        2. CCCP preconditioner build (:math:`n = 5000`).
        3. Full model fit (:math:`n = 1000`).

        Args:
            label: Identifier prepended to benchmark names for logging.
            dtype: PyTorch dtype for all tensor operations.
            pcg_tol: Convergence tolerance for the PCG solver used in
                the full-fit benchmark.

        Returns:
            Dictionary with keys ``matvec_5000`` (ms),
            ``preconditioner_build_5000`` (ms), ``fit_1000`` (ms), and
            ``fit_1000_pcg_iters`` (PCG iteration count or ``None``).
        """
        torch.manual_seed(42)
        results = {}

        # 1. Kernel matvec n=5000
        embeddings = torch.randn(5000, 10, dtype=dtype)
        vector = torch.randn(5000, dtype=dtype)
        kernel = AttentionKernelOperator(embeddings, lambda_reg=1e-2, chunk_size=1024, dtype=dtype)
        for i in range(20):
            kernel.matvec(vector)

        result = self.executor.run_repeated(
            f"{label}_matvec_5000",
            lambda: kernel.matvec(vector),
            repetitions=50,
        )
        results["matvec_5000"] = result["mean_ms"]

        # 2. Preconditioner build n=5000
        torch.manual_seed(42)
        embeddings = torch.randn(5000, 10, dtype=dtype)
        kernel = AttentionKernelOperator(embeddings, lambda_reg=1e-2, dtype=dtype)
        preconditioner = CCCPPreconditioner(
            num_probes=100,
            gamma=1e-1,
            max_iter=20,
            tol=1e-4,
            verbose=False,
            dtype=dtype,
        )
        result = self.executor.run_once(
            f"{label}_preconditioner_build_5000",
            lambda: preconditioner.build(kernel.matvec, 5000),
        )
        results["preconditioner_build_5000"] = result["mean_ms"]

        # 3. Full fit n=1000
        torch.manual_seed(42)
        x_train = torch.rand(1000, 2, dtype=dtype) * 100.0
        y_train = torch.randn(1000, dtype=dtype)
        model = LAKERRegressor(
            embedding_dim=10,
            lambda_reg=1e-2,
            gamma=1e-1,
            num_probes=50,
            cccp_max_iter=20,
            cccp_tol=1e-4,
            pcg_tol=pcg_tol,
            pcg_max_iter=500,
            verbose=False,
            dtype=dtype,
        )
        result = self.executor.run_once(
            f"{label}_fit_1000",
            lambda: model.fit(x_train, y_train),
        )
        results["fit_1000"] = result["mean_ms"]
        results["fit_1000_pcg_iters"] = getattr(model, "pcg_iterations_", None)

        logger.info(
            "%20s  matvec=%sms  pre=%sms  fit=%sms  iters=%s",
            label,
            self.format_number(results["matvec_5000"]),
            self.format_number(results["preconditioner_build_5000"]),
            self.format_number(results["fit_1000"]),
            results["fit_1000_pcg_iters"],
        )
        return results

    def run_all(self) -> None:
        """Run all comparison configurations and print summary.

        Executes optimised benchmarks for both ``float64`` and
        ``float32`` dtypes, then computes and logs speed-up ratios
        against the hardcoded baseline numbers.
        """
        logger.info(
            "Label                  matvec(5000)  preconditioner_build(5000)  fit(1000)  pcg_iters"
        )
        logger.info("-" * 80)

        # Baseline numbers: measured on the original code (git stash) under
        # identical conditions (same script, same seeds, same machine).
        # Obtained 2026-04-30 on Apple M3, PyTorch 2.11.0, Python 3.13.
        baseline = {
            "float64": {
                "matvec_5000": 54.44,
                "preconditioner_build_5000": 180.02,
                "fit_1000": 1117.62,
                "fit_1000_pcg_iters": None,
            },
            "float32": {
                "matvec_5000": 25.98,
                "preconditioner_build_5000": 79.94,
                "fit_1000": 406.52,
                "fit_1000_pcg_iters": None,
            },
        }

        optimized64 = self.run_comparison("optimised-float64", torch.float64, 1e-10)
        optimized32 = self.run_comparison("optimised-float32", torch.float32, 1e-6)

        for key, name in [
            ("matvec_5000", "Kernel matvec n=5000"),
            ("preconditioner_build_5000", "Preconditioner build n=5000"),
            ("fit_1000", "Full fit n=1000"),
        ]:
            baseline64 = baseline["float64"][key]
            opt64 = optimized64[key]
            speedup64 = baseline64 / opt64 if opt64 > 0 else float("inf")
            baseline32 = baseline["float32"][key]
            opt32 = optimized32[key]
            speedup32 = baseline32 / opt32 if opt32 > 0 else float("inf")
            logger.info(
                "%30s  baseline64=%8.2fms  opt64=%8.2fms  speedup64=%6.2fx  "
                "|  baseline32=%8.2fms  opt32=%8.2fms  speedup32=%6.2fx",
                name,
                baseline64,
                opt64,
                speedup64,
                baseline32,
                opt32,
                speedup32,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    comparison = BaselineComparison()
    comparison.run_all()
