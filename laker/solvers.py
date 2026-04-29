"""Iterative solvers for regularised kernel linear systems."""

import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


class PreconditionedConjugateGradient:
    """Preconditioned Conjugate Gradient (PCG) solver.

    Solves ``A x = b`` where ``A`` is symmetric positive-definite, using a
    preconditioner ``P`` such that ``P A`` has a compressed spectrum.

    This is Algorithm 1 (lines 15--26) from the LAKER paper.

    Args:
        tol: Relative residual tolerance ``||r|| / ||b|| <= tol``.
        max_iter: Maximum iterations. If ``None``, defaults to ``n``.
        verbose: Whether to log convergence.
    """

    def __init__(
        self,
        tol: float = 1e-10,
        max_iter: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.tol = float(tol)
        self.max_iter = max_iter
        self.verbose = verbose
        self.iterations: int = 0
        self.residual_norm: float = float("inf")

    def solve(
        self,
        operator: Callable[[torch.Tensor], torch.Tensor],
        preconditioner: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve ``A x = b`` using PCG.

        Args:
            operator: Callable applying ``A`` to a vector.
            preconditioner: Callable applying ``P`` to a vector.
            rhs: Right-hand side tensor ``b`` of shape ``(n,)``.
            x0: Initial guess. If ``None``, zero vector is used.

        Returns:
            Solution tensor ``x`` of shape ``(n,)``.
        """
        n = rhs.shape[0]
        max_iter = self.max_iter if self.max_iter is not None else n

        if x0 is None:
            x = torch.zeros_like(rhs)
            r = rhs.clone()
        else:
            x = x0.clone()
            r = rhs - operator(x)

        z = preconditioner(r)
        p = z.clone()
        rz_old = torch.dot(r, z)
        b_norm = torch.linalg.norm(rhs)

        for iteration in range(max_iter):
            ap = operator(p)
            alpha = rz_old / (torch.dot(p, ap) + 1e-16)
            x = x + alpha * p
            r = r - alpha * ap
            self.residual_norm = torch.linalg.norm(r).item()
            rel_res = self.residual_norm / (b_norm + 1e-16)

            if rel_res <= self.tol:
                self.iterations = iteration + 1
                if self.verbose:
                    logger.info(
                        "PCG converged in %d iterations, rel_res=%.3e", self.iterations, rel_res
                    )
                return x

            z = preconditioner(r)
            rz_new = torch.dot(r, z)
            beta = rz_new / (rz_old + 1e-16)
            p = z + beta * p
            rz_old = rz_new

        self.iterations = max_iter
        if self.verbose:
            logger.warning("PCG did not converge in %d iterations, rel_res=%.3e", max_iter, rel_res)
        return x


class GradientDescent:
    """Unpreconditioned gradient descent baseline for benchmarking.

    Args:
        step_size: Fixed step size ``eta``. If ``None``, a conservative
            heuristic ``1 / max_eig`` is used (requires an extra matvec).
        tol: Relative residual tolerance.
        max_iter: Maximum iterations.
        verbose: Whether to log progress.
    """

    def __init__(
        self,
        step_size: Optional[float] = None,
        tol: float = 1e-3,
        max_iter: int = 50000,
        verbose: bool = False,
    ) -> None:
        self.step_size = step_size
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbose = verbose
        self.iterations: int = 0
        self.residual_norm: float = float("inf")

    def solve(
        self,
        operator: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve ``A x = b`` via gradient descent."""
        if x0 is None:
            x = torch.zeros_like(rhs)
        else:
            x = x0.clone()

        # Estimate step size via power iteration if not provided
        eta = self.step_size
        if eta is None:
            v = torch.randn_like(rhs)
            for _ in range(5):
                v = operator(v)
                v = v / torch.linalg.norm(v)
            max_eig = torch.dot(v, operator(v)).item()
            eta = 0.9 / max(abs(max_eig), 1e-8)
            if self.verbose:
                logger.info("GD estimated step size eta=%.3e", eta)

        b_norm = torch.linalg.norm(rhs)
        for iteration in range(self.max_iter):
            r = rhs - operator(x)
            self.residual_norm = torch.linalg.norm(r).item()
            rel_res = self.residual_norm / (b_norm + 1e-16)
            if rel_res <= self.tol:
                self.iterations = iteration + 1
                if self.verbose:
                    logger.info("GD converged in %d iterations", self.iterations)
                return x
            x = x + eta * r

        self.iterations = self.max_iter
        if self.verbose:
            logger.warning(
                "GD did not converge in %d iterations, rel_res=%.3e", self.max_iter, rel_res
            )
        return x


class JacobiPreconditioner:
    """Diagonal (Jacobi) preconditioner baseline.

    ``P = diag(lambda I + G)^{-1}`` as described in Section V-A-3.
    """

    def __init__(self, diagonal: torch.Tensor) -> None:
        self.inv_diag = 1.0 / diagonal.clamp(min=1e-12)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Jacobi preconditioner element-wise."""
        return self.inv_diag * x
