"""Benchmarking and comparison utilities for LAKER and baselines."""

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from laker.kernels import AttentionKernelOperator
from laker.preconditioner import CCCPPreconditioner
from laker.solvers import (
    GradientDescent,
    JacobiPreconditioner,
    PreconditionedConjugateGradient,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for a single benchmark run."""

    name: str
    n: int
    solve_time_seconds: float
    iterations: int
    final_residual: float
    condition_number: Optional[float] = None
    objective_gap: Optional[float] = None


class SolverBenchmark:
    """Benchmark a single solver configuration."""

    def __init__(
        self,
        name: str,
        operator: Callable[[torch.Tensor], torch.Tensor],
        preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]],
        rhs: torch.Tensor,
        reference_solution: Optional[torch.Tensor] = None,
        tol: float = 1e-10,
        max_iter: int = 1000,
        lambda_reg: float = 1e-2,
    ) -> None:
        """Initialise a single-solver benchmark run."""
        self.name = name
        self.operator = operator
        self.preconditioner = preconditioner
        self.rhs = rhs
        self.reference_solution = reference_solution
        self.tol = tol
        self.max_iter = max_iter
        self.lambda_reg = lambda_reg

    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return results.

        Returns:
            ``BenchmarkResult`` with timing and convergence metrics.

        """
        n = self.rhs.shape[0]
        pcg = PreconditionedConjugateGradient(tol=self.tol, max_iter=self.max_iter, verbose=False)

        start = time.perf_counter()
        if self.preconditioner is not None:
            solution = pcg.solve(self.operator, self.preconditioner, self.rhs)
        else:
            solution = pcg.solve(self.operator, lambda x: x, self.rhs)
        elapsed = time.perf_counter() - start

        rhs_norm = torch.linalg.norm(self.rhs).item()
        final_res = (
            torch.linalg.norm(self.operator(solution) - self.rhs).item() / rhs_norm
            if rhs_norm > 0
            else 0.0
        )

        obj_gap = None
        if self.reference_solution is not None:
            y = self.rhs
            alpha_star = self.reference_solution

            G_alpha_sol = self.operator(solution)
            resid = G_alpha_sol - y
            r_norm_sq = torch.dot(resid, resid).item()
            lambda_term = self.lambda_reg * torch.dot(solution, self.operator(solution)).item()
            obj_sol = r_norm_sq + lambda_term

            obj_ref = self.lambda_reg * torch.dot(y, alpha_star).item()
            obj_gap = abs(obj_sol - obj_ref) / abs(obj_ref) if abs(obj_ref) > 1e-12 else None

        return BenchmarkResult(
            name=self.name,
            n=n,
            solve_time_seconds=elapsed,
            iterations=pcg.iterations,
            final_residual=final_res,
            objective_gap=obj_gap,
        )


class BaselineBenchmark:
    """Run a head-to-head benchmark of LAKER vs baseline solvers."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        measurements: torch.Tensor,
        lambda_reg: float = 1e-2,
        reference_solution: Optional[torch.Tensor] = None,
        pcg_tol: float = 1e-10,
        pcg_max_iter: int = 1000,
    ) -> None:
        """Initialise a head-to-head benchmark run."""
        self.embeddings = embeddings
        self.measurements = measurements
        self.lambda_reg = lambda_reg
        self.reference_solution = reference_solution
        self.pcg_tol = pcg_tol
        self.pcg_max_iter = pcg_max_iter

    def run(self) -> List[BenchmarkResult]:
        """Execute all baseline comparisons.

        Returns:
            List of ``BenchmarkResult`` objects for each solver.

        """
        n = self.embeddings.shape[0]
        operator = AttentionKernelOperator(self.embeddings, lambda_reg=self.lambda_reg)

        results = []

        # 1. LAKER with learned preconditioner
        preconditioner = CCCPPreconditioner(
            num_probes=None,
            gamma=1e-1,
            max_iter=100,
            tol=1e-6,
            verbose=False,
            device=self.embeddings.device,
            dtype=self.embeddings.dtype,
        )
        start = time.perf_counter()
        preconditioner.build(operator.matvec, n)
        pre_time = time.perf_counter() - start

        res = SolverBenchmark(
            name="LAKER",
            operator=operator.matvec,
            preconditioner=preconditioner.apply,
            rhs=self.measurements,
            reference_solution=self.reference_solution,
            tol=self.pcg_tol,
            max_iter=self.pcg_max_iter,
            lambda_reg=self.lambda_reg,
        ).run()
        res.solve_time_seconds += pre_time
        results.append(res)

        # 2. Jacobi PCG
        jac = JacobiPreconditioner(operator.diagonal())
        results.append(
            SolverBenchmark(
                name="Jacobi PCG",
                operator=operator.matvec,
                preconditioner=jac.apply,
                rhs=self.measurements,
                reference_solution=self.reference_solution,
                tol=self.pcg_tol,
                max_iter=self.pcg_max_iter,
                lambda_reg=self.lambda_reg,
            ).run()
        )

        # 3. Unpreconditioned CG
        results.append(
            SolverBenchmark(
                name="CG (no precond)",
                operator=operator.matvec,
                preconditioner=None,
                rhs=self.measurements,
                reference_solution=self.reference_solution,
                tol=self.pcg_tol,
                max_iter=self.pcg_max_iter,
                lambda_reg=self.lambda_reg,
            ).run()
        )

        # 4. Gradient Descent (coarse tolerance)
        gd = GradientDescent(tol=1e-3, max_iter=50000, verbose=False)
        start = time.perf_counter()
        gd.solve(operator.matvec, self.measurements)
        gd_time = time.perf_counter() - start
        results.append(
            BenchmarkResult(
                name="Gradient Descent",
                n=n,
                solve_time_seconds=gd_time,
                iterations=gd.iterations,
                final_residual=gd.residual_norm,
            )
        )

        return results


def benchmark_solver(
    name: str,
    operator: Callable[[torch.Tensor], torch.Tensor],
    preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]],
    rhs: torch.Tensor,
    reference_solution: Optional[torch.Tensor] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
    lambda_reg: float = 1e-2,
) -> BenchmarkResult:
    """Wrap ``SolverBenchmark`` for a single solver run.

    Args:
        name: Human-readable solver name.
        operator: Callable applying ``A`` to a vector.
        preconditioner: Optional preconditioner callable.
        rhs: Right-hand side ``b``.
        reference_solution: Optional exact solution for objective-gap computation.
        tol: Solver tolerance.
        max_iter: Maximum iterations.
        lambda_reg: Regularisation ``lambda``.

    Returns:
        ``BenchmarkResult`` with timing and convergence metrics.

    """
    return SolverBenchmark(
        name=name,
        operator=operator,
        preconditioner=preconditioner,
        rhs=rhs,
        reference_solution=reference_solution,
        tol=tol,
        max_iter=max_iter,
        lambda_reg=lambda_reg,
    ).run()


def benchmark_laker_vs_baselines(
    embeddings: torch.Tensor,
    measurements: torch.Tensor,
    lambda_reg: float = 1e-2,
    reference_solution: Optional[torch.Tensor] = None,
    pcg_tol: float = 1e-10,
    pcg_max_iter: int = 1000,
) -> List[BenchmarkResult]:
    """Wrap ``BaselineBenchmark`` for head-to-head comparison.

    Args:
        embeddings: Tensor of shape ``(n, embedding_dim)``.
        measurements: Observations ``y`` of shape ``(n,)``.
        lambda_reg: Regularisation ``lambda``.
        reference_solution: Optional exact solution for gap computation.
        pcg_tol: PCG tolerance.
        pcg_max_iter: Maximum PCG iterations.

    Returns:
        List of ``BenchmarkResult`` objects for each solver.

    """
    return BaselineBenchmark(
        embeddings=embeddings,
        measurements=measurements,
        lambda_reg=lambda_reg,
        reference_solution=reference_solution,
        pcg_tol=pcg_tol,
        pcg_max_iter=pcg_max_iter,
    ).run()
