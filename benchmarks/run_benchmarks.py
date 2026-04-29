"""Performance benchmarks for LAKER critical paths."""

import time
from typing import Optional

import torch

from laker.kernels import AttentionKernelOperator
from laker.models import LAKERRegressor
from laker.preconditioner import CCCPPreconditioner
from laker.solvers import PreconditionedConjugateGradient


def benchmark_kernel_matvec(n: int = 5000, chunk_size: Optional[int] = 1024) -> dict:
    """Benchmark attention kernel matvec performance.

    Args:
        n: Problem size.
        chunk_size: Chunk size for matrix-free evaluation. None for explicit.

    Returns:
        Dictionary with timing results in milliseconds.
    """
    dtype = torch.float64
    de = 10
    e = torch.randn(n, de, dtype=dtype)
    x = torch.randn(n, dtype=dtype)

    kernel = AttentionKernelOperator(e, lambda_reg=1e-2, chunk_size=chunk_size, dtype=dtype)

    # Warmup
    _ = kernel.matvec(x)

    start = time.perf_counter()
    for _ in range(20):
        _ = kernel.matvec(x)
    elapsed = (time.perf_counter() - start) / 20 * 1000

    return {"n": n, "chunk_size": chunk_size, "matvec_ms": elapsed}


def benchmark_preconditioner_build(n: int = 5000, num_probes: int = 100) -> dict:
    """Benchmark CCCP preconditioner build time.

    Args:
        n: Problem size.
        num_probes: Number of random probes.

    Returns:
        Dictionary with timing results in milliseconds.
    """
    dtype = torch.float64
    de = 10
    e = torch.randn(n, de, dtype=dtype)
    kernel = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=dtype)

    pre = CCCPPreconditioner(
        num_probes=num_probes,
        gamma=1e-1,
        max_iter=20,
        tol=1e-4,
        verbose=False,
        dtype=dtype,
    )

    start = time.perf_counter()
    pre.build(kernel.matvec, n)
    elapsed = (time.perf_counter() - start) * 1000

    return {"n": n, "num_probes": num_probes, "build_ms": elapsed, "iters": pre.max_iter}


def benchmark_pcg_solve(n: int = 5000, num_probes: int = 100) -> dict:
    """Benchmark PCG solve time.

    Args:
        n: Problem size.
        num_probes: Number of random probes for preconditioner.

    Returns:
        Dictionary with timing results in milliseconds.
    """
    dtype = torch.float64
    de = 10
    e = torch.randn(n, de, dtype=dtype)
    kernel = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=dtype)
    b = torch.randn(n, dtype=dtype)

    pre = CCCPPreconditioner(
        num_probes=num_probes,
        gamma=1e-1,
        max_iter=20,
        tol=1e-4,
        verbose=False,
        dtype=dtype,
    )
    pre.build(kernel.matvec, n)

    pcg = PreconditionedConjugateGradient(tol=1e-8, max_iter=500, verbose=False)

    start = time.perf_counter()
    pcg.solve(kernel.matvec, pre.apply, b)
    elapsed = (time.perf_counter() - start) * 1000

    return {
        "n": n,
        "num_probes": num_probes,
        "solve_ms": elapsed,
        "pcg_iters": pcg.iterations,
    }


def benchmark_full_fit(n: int = 500) -> dict:
    """Benchmark full LAKERRegressor fit time.

    Args:
        n: Number of training samples.

    Returns:
        Dictionary with timing results in milliseconds.
    """
    dtype = torch.float64
    x_train = torch.rand(n, 2, dtype=dtype) * 100.0
    y_train = torch.randn(n, dtype=dtype)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-8,
        pcg_max_iter=500,
        verbose=False,
        dtype=dtype,
    )

    start = time.perf_counter()
    model.fit(x_train, y_train)
    elapsed = (time.perf_counter() - start) * 1000

    return {"n": n, "fit_ms": elapsed, "pcg_iters": model.kernel_operator.n}


if __name__ == "__main__":
    print("=" * 60)
    print("LAKER Performance Benchmarks")
    print("=" * 60)

    # Kernel matvec
    for n in [1000, 2000, 5000]:
        res = benchmark_kernel_matvec(n=n, chunk_size=1024)
        print(f'Kernel matvec n={n}: {res["matvec_ms"]:.2f} ms')

    print()

    # Preconditioner build
    for n in [1000, 2000, 5000]:
        res = benchmark_preconditioner_build(n=n, num_probes=100)
        print(f'Preconditioner build n={n}: {res["build_ms"]:.2f} ms')

    print()

    # PCG solve
    for n in [1000, 2000, 5000]:
        res = benchmark_pcg_solve(n=n, num_probes=100)
        print(f'PCG solve n={n}: {res["solve_ms"]:.2f} ms ' f'(iters={res["pcg_iters"]})')

    print()

    # Full fit
    for n in [200, 500, 1000]:
        res = benchmark_full_fit(n=n)
        print(f'Full fit n={n}: {res["fit_ms"]:.2f} ms')
