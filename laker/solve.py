"""Iterative solvers for regularised kernel linear systems.

Public classes:

* :class:`PCG` — Preconditioned Conjugate Gradient.
* :class:`Descent` — Bare gradient descent baseline.
* :class:`Jacobi` — Diagonal preconditioner baseline.
* :class:`Report` — Convergence result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class Report:
    """Convergence result returned alongside a solver's iterate.

    Attributes:
        converged: ``True`` when the relative residual met ``tol``.
        iterations: Number of iterations actually performed.
        residual: Final ``||r|| / ||b||`` (always finite; ``0.0`` for
            the zero-RHS short-circuit).
        reason: ``"converged"`` / ``"max_iter"`` / ``"zero_rhs"`` /
            ``"breakdown"`` / ``"nonfinite"``.
        per: Per-RHS statuses for the batched 2-D path (``None`` for 1-D).
    """

    converged: bool
    iterations: int
    residual: float
    reason: str
    per: Optional[List["Report"]] = None


class PCG:
    """Preconditioned Conjugate Gradient solver.

    Solves ``A x = b`` where ``A`` is symmetric positive-definite, using
    a preconditioner ``P`` such that ``P A`` has a compressed spectrum.

    Supports both 1-D and 2-D (batch) RHS, residual-replacement restart,
    and breakdown detection.
    """

    def __init__(
        self,
        tol: float = 1e-10,
        max_iter: Optional[int] = None,
        verbose: bool = True,
        restart: Optional[int] = None,
        eps: Optional[float] = None,
        autograd: bool = False,
    ) -> None:
        self.tol = float(tol)
        self.max_iter = max_iter
        self.verbose = verbose
        self.restart = restart
        self.eps = eps
        self.autograd = autograd
        self.iterations: int = 0
        self.residual: float = float("inf")

    def solve(
        self,
        op: Callable[[torch.Tensor], torch.Tensor],
        prec: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Report]:
        """Solve ``A x = b``.

        Args:
            op: Callable applying ``A`` to a vector or batch.
            prec: Callable applying ``P``.
            rhs: Right-hand side of shape ``(n,)`` or ``(n, k)``.
            x0: Optional initial guess.

        Returns:
            ``(solution, Report)``.

        Raises:
            ValueError: ``rhs`` is not 1-D or 2-D.
            RuntimeError: breakdown (non-positive curvature).
        """
        if rhs.dim() not in (1, 2):
            raise ValueError(f"rhs must be 1-D or 2-D, got shape {rhs.shape}")

        n = rhs.shape[0]
        max_iter = self.max_iter if self.max_iter is not None else n

        if x0 is None:
            x = torch.zeros_like(rhs)
            r = rhs.clone()
        else:
            x = x0.clone()
            r = rhs - op(x)

        z = prec(r)
        p = z.clone()
        rhs_norm = torch.linalg.norm(rhs)
        if rhs_norm.item() == 0:
            self.iterations = 0
            self.residual = 0.0
            return x, Report(converged=True, iterations=0, residual=0.0, reason="zero_rhs")

        if rhs.dim() == 1:
            return self.solve1(op, prec, rhs, x, r, z, p, rhs_norm, max_iter)
        return self.solve2(op, prec, rhs, x, r, z, p, rhs_norm, max_iter)

    def solve1(
        self,
        op: Callable[[torch.Tensor], torch.Tensor],
        prec: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        x: torch.Tensor,
        r: torch.Tensor,
        z: torch.Tensor,
        p: torch.Tensor,
        rhs_norm,
        max_iter: int,
    ) -> tuple[torch.Tensor, Report]:
        rz_old = torch.dot(r, z).item()
        rel = float("inf")
        for it in range(max_iter):
            av = op(p)
            pap = torch.dot(p, av).item()
            eps_bound = self.eps if self.eps is not None else torch.finfo(p.dtype).eps ** 0.5
            norm_p = torch.linalg.norm(p).item()
            norm_av = torch.linalg.norm(av).item()
            tol_scale = max(eps_bound * norm_p * norm_p, eps_bound * norm_p * norm_av)
            if pap <= -tol_scale:
                raise RuntimeError(
                    f"PCG breakdown at iteration {it}: non-positive curvature detected "
                    "(p^T A p <= 0)."
                )
            alpha = rz_old / pap
            if self.autograd:
                x = x + alpha * p
                r = r - alpha * av
            else:
                x.add_(p, alpha=alpha)
                r.add_(av, alpha=-alpha)
            if self.restart is not None and (it + 1) % self.restart == 0:
                r = rhs - op(x)
            self.residual = torch.linalg.norm(r).item()
            rel = self.residual / rhs_norm
            if rel <= self.tol:
                self.iterations = it + 1
                if self.verbose:
                    logger.info(
                        "PCG converged in %d iterations, rel_res=%.3e", self.iterations, rel
                    )
                return x, Report(
                    converged=True, iterations=self.iterations, residual=rel, reason="converged"
                )
            z = prec(r)
            rz_new = torch.dot(r, z).item()
            beta = rz_new / rz_old
            if self.autograd:
                p = z + beta * p
            else:
                p.mul_(beta).add_(z)
            rz_old = rz_new

        self.iterations = max_iter
        if self.verbose:
            logger.warning("PCG did not converge in %d iterations, rel_res=%.3e", max_iter, rel)
        return x, Report(
            converged=False, iterations=self.iterations, residual=rel, reason="max_iter"
        )

    def solve2(
        self,
        op: Callable[[torch.Tensor], torch.Tensor],
        prec: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        x: torch.Tensor,
        r: torch.Tensor,
        z: torch.Tensor,
        p: torch.Tensor,
        rhs_norm,
        max_iter: int,
    ) -> tuple[torch.Tensor, Report]:
        rz_old = torch.sum(r * z, dim=0)
        for it in range(max_iter):
            av = op(p)
            pap = torch.sum(p * av, dim=0)
            eps_bound = self.eps if self.eps is not None else torch.finfo(p.dtype).eps ** 0.5
            norm_p = torch.linalg.norm(p, dim=0)
            norm_av = torch.linalg.norm(av, dim=0)
            tol_scale = torch.maximum(eps_bound * norm_p * norm_p, eps_bound * norm_p * norm_av)
            if torch.any(pap <= -tol_scale):
                raise RuntimeError(
                    f"PCG breakdown at iteration {it}: non-positive curvature detected."
                )
            alpha = rz_old / pap
            if self.autograd:
                x = x + p * alpha.unsqueeze(0)
                r = r - av * alpha.unsqueeze(0)
            else:
                x.add_(p * alpha.unsqueeze(0))
                r.add_(av * (-alpha.unsqueeze(0)))
            if self.restart is not None and (it + 1) % self.restart == 0:
                r = rhs - op(x)
            self.residual = torch.linalg.norm(r).item()
            rel = self.residual / rhs_norm
            if rel <= self.tol:
                self.iterations = it + 1
                if self.verbose:
                    logger.info(
                        "PCG converged in %d iterations, rel_res=%.3e", self.iterations, rel
                    )
                return x, Report(
                    converged=True,
                    iterations=self.iterations,
                    residual=rel,
                    reason="converged",
                    per=[
                        Report(
                            converged=True,
                            iterations=self.iterations,
                            residual=float(rel),
                            reason="converged",
                        )
                        for _ in range(rhs.shape[1])
                    ],
                )
            z = prec(r)
            rz_new = torch.sum(r * z, dim=0)
            beta = rz_new / rz_old
            if self.autograd:
                p = z + p * beta.unsqueeze(0)
            else:
                p.mul_(beta.unsqueeze(0)).add_(z)
            rz_old = rz_new

        self.iterations = max_iter
        if self.verbose:
            logger.warning("PCG did not converge in %d iterations, rel_res=%.3e", max_iter, rel)
        return x, Report(
            converged=False,
            iterations=self.iterations,
            residual=rel,
            reason="max_iter",
            per=[
                Report(
                    converged=False,
                    iterations=self.iterations,
                    residual=float(rel),
                    reason="max_iter",
                )
                for _ in range(rhs.shape[1])
            ],
        )


class Descent:
    """Bare gradient descent baseline.

    Used by the benchmark suite as a worst-case reference against the
    LAKER preconditioner. Does not scale to large problems.
    """

    def __init__(
        self,
        step: Optional[float] = None,
        tol: float = 1e-3,
        max_iter: int = 50000,
        verbose: bool = False,
    ) -> None:
        self.step = step
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbose = verbose
        self.iterations: int = 0
        self.residual: float = float("inf")

    def solve(
        self,
        op: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve ``A x = b`` via gradient descent."""
        if x0 is None:
            x = torch.zeros_like(rhs)
        else:
            x = x0.clone()

        eta = self.step
        if eta is None:
            v = torch.randn_like(rhs)
            for _ in range(5):
                v = op(v)
                v = v / torch.linalg.norm(v)
            max_eig = torch.dot(v, op(v)).item()
            eta = 0.9 / max(abs(max_eig), 1e-8)
            if self.verbose:
                logger.info("Descent estimated step eta=%.3e", eta)

        b_norm = torch.linalg.norm(rhs)
        for it in range(self.max_iter):
            r = rhs - op(x)
            self.residual = torch.linalg.norm(r).item()
            rel = self.residual / b_norm
            if rel <= self.tol:
                self.iterations = it + 1
                if self.verbose:
                    logger.info("Descent converged in %d iterations", self.iterations)
                return x
            x = x + eta * r

        self.iterations = self.max_iter
        if self.verbose:
            logger.warning(
                "Descent did not converge in %d iterations, rel_res=%.3e",
                self.max_iter,
                rel,
            )
        return x


class Jacobi:
    """Diagonal preconditioner baseline ``P = diag(lambda I + G)^{-1}``."""

    def __init__(self, diag: torch.Tensor) -> None:
        eps = torch.finfo(diag.dtype).eps
        self.inv = 1.0 / diag.clamp(min=eps)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``P`` element-wise to ``x``."""
        return self.inv * x


__all__ = ["Report", "PCG", "Descent", "Jacobi"]
