"""Reproducible benchmarks for LAKER critical paths.

Run with::

    python benchmarks/reproducible.py

This module provides deterministic, seed-controlled benchmarks for
every major LAKER component.  All random inputs are generated via
``torch.manual_seed(42)`` to ensure reproducibility across runs on the
same hardware and PyTorch version.

Benchmarked components:

* **Attention kernel matvec** — :math:`Kx` with
  :class:`~laker.kernels.AttentionKernelOperator`.
* **Approximation matvec** — Nyström, RFF, k-NN, and SKI kernel
  operators compared against the exact kernel.
* **CCCP preconditioner build** — randomised low-rank approximation of
  :math:`K^{-1}`.
* **PCG solve** — preconditioned conjugate gradient for
  :math:`(K + \\lambda I)\\alpha = y`.
* **Full model fit** — end-to-end
  :class:`~laker.models.LAKERRegressor` pipeline.

Environment assumptions:

* CPU: Apple M3 (Darwin)
* Python 3.13
* PyTorch 2.x+
* 20 warm-up iterations, 50 measured iterations for matvec benchmarks
* Single-trial for preconditioner build and PCG solve (low variance)

The :meth:`generate_report` method produces a Markdown-formatted
summary table suitable for inclusion in a README file.
"""

import logging
from typing import Optional

import torch

from benchmarks.executor import BenchmarkExecutor
from laker.embeddings import PositionEmbedding
from laker.kernels import (
    AttentionKernelOperator,
    NystromAttentionKernelOperator,
    RandomFeatureAttentionKernelOperator,
    SKIAttentionKernelOperator,
    SparseKNNAttentionKernelOperator,
)
from laker.models import LAKERRegressor
from laker.preconditioner import CCCPPreconditioner
from laker.solvers import PreconditionedConjugateGradient

logger = logging.getLogger(__name__)


class ReproducibleBenchmarkSuite:
    """Suite of reproducible benchmarks for LAKER critical paths.

    Provides deterministic benchmarks with fixed random seeds and
    generates Markdown-formatted reports.  Unlike
    :class:`~benchmarks.run.PerformanceBenchmarkSuite`, this suite uses
    realistic embeddings produced by
    :class:`~laker.embeddings.PositionEmbedding` rather than raw
    Gaussian random vectors.

    Args:
        executor: Optional pre-configured
            :class:`~benchmarks.executor.BenchmarkExecutor`.  When
            ``None`` a new executor with default settings is created.

    Attributes:
        dtype: Default floating-point dtype (``float32``).
        embedding_dim: Embedding dimension used for synthetic data.
        lambda_reg: Regularisation weight :math:`\\lambda` for the
            attention kernel.
    """

    def __init__(self, executor: Optional[BenchmarkExecutor] = None):
        self.executor = executor if executor is not None else BenchmarkExecutor()
        self.dtype = torch.float32
        self.embedding_dim = 10
        self.lambda_reg = 1e-2

    def warmup(self, kernel, vector, count: int = 20) -> None:
        """Warm up a kernel by running matvec multiple times.

        Executes ``count`` un-timed matrix-vector products to warm CPU
        caches and trigger any lazy initialisation in the kernel
        operator.

        Args:
            kernel: Kernel operator with a ``.matvec(vector)`` method.
            vector: Input vector of shape ``(n,)``.
            count: Number of warm-up iterations.
        """
        for i in range(count):
            kernel.matvec(vector)

    def make_embeddings(self, n: int, dim: int = 10) -> torch.Tensor:
        """Generate realistic embeddings via PositionEmbedding for benchmarking.

        Creates ``n`` random 2-D locations and passes them through a
        :class:`~laker.embeddings.PositionEmbedding` layer to produce
        deterministic embeddings of shape ``(n, dim)``.

        Args:
            n: Number of data points.
            dim: Embedding dimension.

        Returns:
            Tensor of shape ``(n, dim)`` with dtype ``self.dtype``.
        """
        torch.manual_seed(42)
        x = torch.rand(n, 2, dtype=self.dtype) * 100.0
        embed = PositionEmbedding(input_dim=2, embedding_dim=dim, dtype=self.dtype)
        with torch.no_grad():
            return embed(x)

    def standard_deviation(self, times: list) -> float:
        """Compute the population standard deviation of a list of times.

        Args:
            times: List of float values (typically wall-clock times).

        Returns:
            Population standard deviation.
        """
        mean = sum(times) / len(times)
        return (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5

    def kernel_matvec(
        self,
        n: int = 5000,
        chunk_size: Optional[int] = 1024,
        trials: int = 50,
        warmup: int = 20,
    ) -> dict:
        """Benchmark attention kernel matvec performance.

        Generates embeddings via :meth:`make_embeddings` and measures
        per-iteration matvec time using :meth:`BenchmarkExecutor.run`.

        Args:
            n: Number of data points.
            chunk_size: Chunk size for matrix-free evaluation.  When
                ``None`` the full kernel is materialised.
            trials: Number of measured iterations.
            warmup: Number of un-timed warm-up iterations.

        Returns:
            Dictionary with keys ``n``, ``chunk_size``,
            ``matvec_ms_mean``, and ``matvec_ms_std``.
        """
        embeddings = self.make_embeddings(n, dim=self.embedding_dim)
        vector = torch.randn(n, dtype=self.dtype)
        kernel = AttentionKernelOperator(
            embeddings,
            lambda_reg=self.lambda_reg,
            chunk_size=chunk_size,
            dtype=self.dtype,
        )
        self.warmup(kernel, vector, warmup)

        result = self.executor.run(
            f"kernel_matvec_n{n}",
            lambda: kernel.matvec(vector),
            trials=trials,
            warmup=0,
        )
        return {
            "n": n,
            "chunk_size": chunk_size,
            "matvec_ms_mean": result["mean_ms"],
            "matvec_ms_std": result["std_ms"],
        }

    def preconditioner_build(self, n: int = 5000, num_probes: int = 100) -> dict:
        """Benchmark CCCP preconditioner build time.

        Builds a :class:`~laker.preconditioner.CCCPPreconditioner` for
        an :math:`n \\times n` attention kernel and measures the wall-
        clock time of the :meth:`~laker.preconditioner.CCCPPreconditioner.build`
        call.

        Args:
            n: Number of data points.
            num_probes: Number of random probe vectors :math:`N_r` for
                the CCCP approximation.

        Returns:
            Dictionary with keys ``n``, ``num_probes``, and
            ``build_ms`` (build time in milliseconds).
        """
        embeddings = self.make_embeddings(n, dim=self.embedding_dim)
        kernel = AttentionKernelOperator(embeddings, lambda_reg=self.lambda_reg, dtype=self.dtype)
        preconditioner = CCCPPreconditioner(
            num_probes=num_probes,
            gamma=1e-1,
            max_iter=20,
            tol=1e-4,
            verbose=False,
            dtype=self.dtype,
        )

        result = self.executor.run_once(
            f"preconditioner_build_n{n}",
            lambda: preconditioner.build(kernel.matvec, n),
        )
        return {
            "n": n,
            "num_probes": num_probes,
            "build_ms": result["mean_ms"],
        }

    def pcg_solve(self, n: int = 5000, num_probes: int = 100) -> dict:
        """Benchmark PCG solve time.

        Solves :math:`(K + \\lambda I)\\alpha = b` for a random right-
        hand side :math:`b` using
        :class:`~laker.solvers.PreconditionedConjugateGradient` with
        tolerance :math:`10^{-10}` and up to 1000 iterations.

        Args:
            n: Number of data points.
            num_probes: Number of random probe vectors for the CCCP
                preconditioner.

        Returns:
            Dictionary with keys ``n``, ``num_probes``, ``solve_ms``
            (solve time in milliseconds), and ``pcg_iters`` (PCG
            iteration count).
        """
        embeddings = self.make_embeddings(n, dim=self.embedding_dim)
        kernel = AttentionKernelOperator(embeddings, lambda_reg=self.lambda_reg, dtype=self.dtype)
        rhs = torch.randn(n, dtype=self.dtype)

        preconditioner = CCCPPreconditioner(
            num_probes=num_probes,
            gamma=1e-1,
            max_iter=20,
            tol=1e-4,
            verbose=False,
            dtype=self.dtype,
        )
        preconditioner.build(kernel.matvec, n)

        pcg = PreconditionedConjugateGradient(tol=1e-10, max_iter=1000, verbose=False)

        result = self.executor.run_once(
            f"pcg_solve_n{n}",
            lambda: pcg.solve(kernel.matvec, preconditioner.apply, rhs),
        )
        return {
            "n": n,
            "num_probes": num_probes,
            "solve_ms": result["mean_ms"],
            "pcg_iters": pcg.iterations,
        }

    def full_fit(self, n: int = 500) -> dict:
        """Benchmark full LAKERRegressor fit time.

        Generates synthetic 2-D training data and measures the end-to-
        end :meth:`~laker.models.LAKERRegressor.fit` call.

        Args:
            n: Number of training samples.

        Returns:
            Dictionary with keys ``n``, ``fit_ms`` (total fit time in
            milliseconds), and ``pcg_iters`` (PCG iteration count).
        """
        torch.manual_seed(42)
        x_train = torch.rand(n, 2, dtype=self.dtype) * 100.0
        y_train = torch.randn(n, dtype=self.dtype)

        model = LAKERRegressor(
            embedding_dim=self.embedding_dim,
            lambda_reg=self.lambda_reg,
            gamma=1e-1,
            num_probes=50,
            cccp_max_iter=20,
            cccp_tol=1e-4,
            pcg_tol=1e-10,
            pcg_max_iter=1000,
            verbose=False,
            dtype=self.dtype,
        )

        result = self.executor.run_once(
            f"full_fit_n{n}",
            lambda: model.fit(x_train, y_train),
        )
        return {
            "n": n,
            "fit_ms": result["mean_ms"],
            "pcg_iters": getattr(model, "pcg_iterations_", None),
        }

    def approx_kernel_matvec(self, n: int = 2000, trials: int = 20) -> dict:
        """Benchmark matvec for all kernel approximations.

        Measures per-iteration matvec time for the exact, Nyström (200
        landmarks), RFF (400 features), k-NN (50 neighbours), and SKI
        (grid size 1024) kernel operators using the same embedding
        matrix.

        Args:
            n: Number of data points.
            trials: Number of measured iterations per operator.

        Returns:
            Dictionary with key ``n`` and sub-dictionaries keyed by
            ``"exact"``, ``"nystrom"``, ``"rff"``, ``"knn"``, and
            ``"ski"``, each containing ``mean`` (ms) and ``std`` (ms).
        """
        embeddings = self.make_embeddings(n, dim=self.embedding_dim)
        vector = torch.randn(n, dtype=self.dtype)

        results = {}

        # Exact
        operator = AttentionKernelOperator(embeddings, lambda_reg=self.lambda_reg, dtype=self.dtype)
        result = self.executor.run(
            "exact_matvec",
            lambda: operator.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["exact"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        # Nyström
        operator = NystromAttentionKernelOperator(
            embeddings, lambda_reg=self.lambda_reg, num_landmarks=200, dtype=self.dtype
        )
        result = self.executor.run(
            "nystrom_matvec",
            lambda: operator.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["nystrom"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        # RFF
        operator = RandomFeatureAttentionKernelOperator(
            embeddings, lambda_reg=self.lambda_reg, num_features=400, dtype=self.dtype
        )
        result = self.executor.run(
            "rff_matvec",
            lambda: operator.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["rff"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        # k-NN
        operator = SparseKNNAttentionKernelOperator(
            embeddings, lambda_reg=self.lambda_reg, k_neighbors=50, dtype=self.dtype
        )
        result = self.executor.run(
            "knn_matvec",
            lambda: operator.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["knn"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        # SKI
        operator = SKIAttentionKernelOperator(
            embeddings, lambda_reg=self.lambda_reg, grid_size=1024, dtype=self.dtype
        )
        result = self.executor.run(
            "ski_matvec",
            lambda: operator.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["ski"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        return {"n": n, **results}

    def generate_report(self) -> str:
        """Run all benchmarks and return a Markdown report.

        Executes kernel matvec, approximation matvec, preconditioner
        build, PCG solve, and full-fit benchmarks across multiple
        problem sizes, then assembles a Markdown document with
        per-benchmark tables.

        Returns:
            A Markdown-formatted string containing all benchmark result
            tables.
        """
        lines = [
            "# LAKER Benchmark Results",
            "",
            "**Date:** 2026-04-30  ",
            "**Platform:** Darwin (macOS)  ",
            "**PyTorch:** " + torch.__version__ + "  ",
            "**Dtype:** float32 (default)  ",
            "**Seed:** 42  ",
            "",
            "---",
            "",
        ]

        # Kernel matvec
        lines.append("## Kernel Matvec")
        lines.append("")
        lines.append("| n | chunk_size | mean (ms) | std (ms) |")
        lines.append("|---|------------|-----------|----------|")
        for n in [1000, 2000, 5000]:
            result = self.kernel_matvec(n=n, chunk_size=1024, trials=50, warmup=20)
            lines.append(
                f"| {result['n']} | {result['chunk_size']} | "
                f"{result['matvec_ms_mean']:.3f} | {result['matvec_ms_std']:.3f} |"
            )
        lines.append("")

        # Approximation matvec comparison
        lines.append("## Approximation Matvec Comparison (n=2000)")
        lines.append("")
        lines.append("| method | mean (ms) | std (ms) |")
        lines.append("|--------|-----------|----------|")
        result = self.approx_kernel_matvec(n=2000, trials=20)
        for method in ["exact", "nystrom", "rff", "knn", "ski"]:
            lines.append(
                f"| {method} | {result[method]['mean']:.3f} | {result[method]['std']:.3f} |"
            )
        lines.append("")

        # Preconditioner build
        lines.append("## Preconditioner Build")
        lines.append("")
        lines.append("| n | N_r | time (ms) |")
        lines.append("|---|-----|-----------|")
        for n in [1000, 2000, 5000]:
            result = self.preconditioner_build(n=n, num_probes=100)
            lines.append(f"| {result['n']} | {result['num_probes']} | {result['build_ms']:.2f} |")
        lines.append("")

        # PCG solve
        lines.append("## PCG Solve")
        lines.append("")
        lines.append("| n | N_r | time (ms) | iters |")
        lines.append("|---|-----|-----------|-------|")
        for n in [1000, 2000, 5000]:
            result = self.pcg_solve(n=n, num_probes=100)
            lines.append(
                f"| {result['n']} | {result['num_probes']} | "
                f"{result['solve_ms']:.2f} | {result['pcg_iters']} |"
            )
        lines.append("")

        # Full fit
        lines.append("## Full Fit")
        lines.append("")
        lines.append("| n | time (ms) | PCG iters |")
        lines.append("|---|-----------|-----------|")
        for n in [200, 500, 1000]:
            result = self.full_fit(n=n)
            lines.append(f"| {result['n']} | {result['fit_ms']:.2f} | {result['pcg_iters']} |")
        lines.append("")

        return "\n".join(lines)

    def save_report(self, path: str = "benchmarks/README.md") -> None:
        """Generate and save the benchmark report to a file.

        Args:
            path: Filesystem path where the Markdown report is written.
                Defaults to ``"benchmarks/README.md"``.
        """
        report = self.generate_report()
        with open(path, "w") as file:
            file.write(report)
        logger.info("Report saved to %s", path)

    def run_all(self) -> None:
        """Run all benchmarks, print report, and save to file.

        Executes the full benchmark suite, logs the Markdown report at
        ``INFO`` level, and writes it to ``benchmarks/README.md``.
        """
        report = self.generate_report()
        logger.info("\n%s", report)
        self.save_report()

    @classmethod
    def run_all_default(cls) -> str:
        """Run all reproducible benchmarks and return the markdown report.

        Returns:
            Markdown formatted string with all benchmark results.
        """
        suite = cls()
        return suite.generate_report()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    suite = ReproducibleBenchmarkSuite()
    suite.run_all()
