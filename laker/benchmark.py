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


def benchmark_solver(
    name: str,
    operator: Callable[[torch.Tensor], torch.Tensor],
    preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]],
    rhs: torch.Tensor,
    reference_solution: Optional[torch.Tensor] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> BenchmarkResult:
    """Benchmark a single solver configuration.

    Args:
        name: Human-readable solver name.
        operator: Callable applying ``A`` to a vector.
        preconditioner: Optional preconditioner callable.
        rhs: Right-hand side ``b``.
        reference_solution: Optional exact solution for objective-gap computation.
        tol: Solver tolerance.
        max_iter: Maximum iterations.

    Returns:
        ``BenchmarkResult`` with timing and convergence metrics.
    """
    n = rhs.shape[0]
    pcg = PreconditionedConjugateGradient(tol=tol, max_iter=max_iter, verbose=False)

    start = time.perf_counter()
    if preconditioner is not None:
        solution = pcg.solve(operator, preconditioner, rhs)
    else:
        solution = pcg.solve(operator, lambda x: x, rhs)
    elapsed = time.perf_counter() - start

    final_res = torch.linalg.norm(operator(solution) - rhs).item() / torch.linalg.norm(rhs).item()

    obj_gap = None
    if reference_solution is not None:
        obj_gap = (
            torch.norm(solution - reference_solution).item() / torch.norm(reference_solution).item()
        )

    return BenchmarkResult(
        name=name,
        n=n,
        solve_time_seconds=elapsed,
        iterations=pcg.iterations,
        final_residual=final_res,
        objective_gap=obj_gap,
    )


def benchmark_laker_vs_baselines(
    embeddings: torch.Tensor,
    measurements: torch.Tensor,
    lambda_reg: float = 1e-2,
    reference_solution: Optional[torch.Tensor] = None,
    pcg_tol: float = 1e-10,
    pcg_max_iter: int = 1000,
) -> List[BenchmarkResult]:
    """Run a head-to-head benchmark of LAKER vs baseline solvers.

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
    n = embeddings.shape[0]
    op = AttentionKernelOperator(embeddings, lambda_reg=lambda_reg)

    results = []

    # 1. LAKER with learned preconditioner
    pre = CCCPPreconditioner(
        num_probes=None,
        gamma=1e-1,
        max_iter=100,
        tol=1e-6,
        verbose=False,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    start = time.perf_counter()
    pre.build(op.matvec, n)
    pre_time = time.perf_counter() - start

    res = benchmark_solver(
        name="LAKER",
        operator=op.matvec,
        preconditioner=pre.apply,
        rhs=measurements,
        reference_solution=reference_solution,
        tol=pcg_tol,
        max_iter=pcg_max_iter,
    )
    res.solve_time_seconds += pre_time
    results.append(res)

    # 2. Jacobi PCG
    jac = JacobiPreconditioner(op.diagonal())
    results.append(
        benchmark_solver(
            name="Jacobi PCG",
            operator=op.matvec,
            preconditioner=jac.apply,
            rhs=measurements,
            reference_solution=reference_solution,
            tol=pcg_tol,
            max_iter=pcg_max_iter,
        )
    )

    # 3. Unpreconditioned CG
    results.append(
        benchmark_solver(
            name="CG (no precond)",
            operator=op.matvec,
            preconditioner=None,
            rhs=measurements,
            reference_solution=reference_solution,
            tol=pcg_tol,
            max_iter=pcg_max_iter,
        )
    )

    # 4. Gradient Descent (coarse tolerance)
    gd = GradientDescent(tol=1e-3, max_iter=50000, verbose=False)
    start = time.perf_counter()
    gd.solve(op.matvec, measurements)
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
