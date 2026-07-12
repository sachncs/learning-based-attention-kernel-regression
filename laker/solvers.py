"""Iterative solvers for regularised kernel linear systems.

The dominant cost of a LAKER fit is solving the linear system

.. math::

    (G + \\lambda I) \\alpha = y

where :math:`G = \\exp(E E^\\top)` is the attention kernel matrix. This
module contains the iterative solvers used to perform that solve, plus
two baseline solvers used by the benchmarking harness for comparisons.

Public classes:

* :class:`PreconditionedConjugateGradient` — the production solver. It
  is Algorithm 1 (lines 15–26) of the LAKER paper, extended with
  residual-replacement restart, breakdown detection, and an optional
  autograd-safe branch for backprop through the iteration.
* :class:`GradientDescent` — bare unpreconditioned gradient descent,
  used as a slow-but-trivial baseline in benchmarks.
* :class:`JacobiPreconditioner` — diagonal preconditioner ``P =
  diag(\\lambda I + G)^{-1}``, used as a fast-but-weak baseline.

Performance and numerical notes:

* The PCG hot loop accepts either a 1-D ``(n,)`` or 2-D ``(n, k)`` RHS.
  The 1-D path uses scalar dot products for speed; the 2-D path uses
  column-wise vectorised operations.
* Residual-replacement restart (``restart_freq``) explicitly recomputes
  the residual from ``b - A x`` every ``k`` iterations. It is disabled
  by default because in ``float32`` the explicit subtraction causes
  catastrophic cancellation; it is useful for very long ``float64``
  runs.
* Breakdown detection compares ``p^T A p`` against a small
  Cauchy-Schwarz bound ``eps * ||p|| * ||A p||`` with
  ``eps = sqrt(finfo(dtype).eps)``. Negative curvature (``p^T A p < 0``)
  raises :class:`RuntimeError`.
"""

import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


class PreconditionedConjugateGradient:
    """Preconditioned Conjugate Gradient (PCG) solver.

    Solves ``A x = b`` where ``A`` is symmetric positive-definite, using
    a preconditioner ``P`` such that ``P A`` has a compressed spectrum
    (so PCG converges in many fewer iterations than vanilla CG).

    Supports both 1-D (single RHS) and 2-D (batch of RHS) inputs using
    vectorised block operations. Includes residual-replacement restart
    and breakdown detection.

    This is Algorithm 1 (lines 15–26) from the LAKER paper with two
    practical extensions:

    1. **Breakdown detection** — raises :class:`RuntimeError` if the
       search direction becomes non-positive-curvature (``p^T A p <= 0``).
    2. **Residual restart** — periodically recomputes the residual
       from ``b - A x`` to combat floating-point drift over very long
       iterations.

    Args:
        tol: Relative residual tolerance ``||r|| / ||b|| <= tol``.
            Default ``1e-10``.
        max_iter: Maximum iterations. If ``None``, defaults to ``n``
            (the system dimension).
        verbose: Whether to log convergence.
        restart_freq: If an integer, the residual is explicitly
            recomputed from ``b - A x`` every ``restart_freq``
            iterations to avoid round-off accumulation. ``None``
            disables recomputation. Disabled by default because it can
            hurt ``float32`` stability (catastrophic cancellation in
            the explicit subtraction).
        breakdown_eps: Threshold for the Cauchy-Schwarz breakdown
            bound. If ``None``, defaults to
            ``sqrt(finfo(dtype).eps)``.
        autograd_safe: If ``True``, the iteration uses out-of-place
            updates (``x = x + alpha * p``) instead of in-place
            (``x.add_(p, alpha=alpha)``). The out-of-place form is
            required when the iteration participates in an autograd
            graph (PyTorch cannot backprop through in-place tensor
            modifications of a leaf tensor). Default ``False``.

    Attributes:
        iterations: Number of PCG iterations executed by the last
            :meth:`solve` call (``0`` until the first call).
        residual_norm: ``||b - A x||`` at termination of the last
            :meth:`solve` call (``inf`` until the first call).

    """

    def __init__(
        self,
        tol: float = 1e-10,
        max_iter: Optional[int] = None,
        verbose: bool = True,
        restart_freq: Optional[int] = None,
        breakdown_eps: Optional[float] = None,
        autograd_safe: bool = False,
    ) -> None:
        """Initialise the PCG solver.

        Args:
            tol: Relative residual tolerance.
            max_iter: Maximum iterations (``None`` ⇒ default ``n``).
            verbose: Whether to log convergence progress.
            restart_freq: Residual-replacement frequency (``None`` =
                disabled).
            breakdown_eps: Breakdown threshold (``None`` ⇒ default
                ``sqrt(finfo(dtype).eps)``).
            autograd_safe: Use out-of-place updates for autograd
                compatibility.

        """
        self.tol = float(tol)
        self.max_iter = max_iter
        self.verbose = verbose
        self.restart_freq = restart_freq
        self.breakdown_eps = breakdown_eps
        self.autograd_safe = autograd_safe
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
            operator: Callable applying ``A`` to a vector or batch.
            preconditioner: Callable applying ``P`` to a vector or
                batch.
            rhs: Right-hand side tensor ``b`` of shape ``(n,)`` or
                ``(n, k)``.
            x0: Initial guess. If ``None``, the zero vector is used
                (cold start). Pass a warm-started vector (e.g. from a
                previous solve with similar hyperparameters) to reduce
                iteration count.

        Returns:
            Solution tensor ``x`` of the same shape as ``rhs``.

        Raises:
            ValueError: If ``rhs`` is not 1-D or 2-D.
            RuntimeError: If the iteration encounters non-positive
                curvature (breakdown) or another numerical pathology.

        """
        if rhs.dim() not in (1, 2):
            raise ValueError(f"rhs must be 1-D or 2-D, got shape {rhs.shape}")

        n = rhs.shape[0]
        max_iter = self.max_iter if self.max_iter is not None else n

        if x0 is None:
            x = torch.zeros_like(rhs)
            r = rhs.clone()
        else:
            # Warm start: copy the guess and recompute the residual
            # exactly from ``b - A x`` to avoid carrying forward any
            # stale residual state.
            x = x0.clone()
            r = rhs - operator(x)

        z = preconditioner(r)
        p = z.clone()
        rhs_norm = torch.linalg.norm(rhs)
        # If the RHS is identically zero, the solution is trivially
        # zero regardless of ``A``. Returning early avoids division by
        # zero in the relative-residual check below.
        if rhs_norm.item() == 0:
            return x

        if rhs.dim() == 1:
            return self.solve_1d(operator, preconditioner, rhs, x, r, z, p, rhs_norm, max_iter)
        else:
            return self.solve_2d(operator, preconditioner, rhs, x, r, z, p, rhs_norm, max_iter)

    def solve_1d(
        self,
        operator: Callable[[torch.Tensor], torch.Tensor],
        preconditioner: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        x: torch.Tensor,
        r: torch.Tensor,
        z: torch.Tensor,
        p: torch.Tensor,
        rhs_norm: float,
        max_iter: int,
    ) -> torch.Tensor:
        """1-D (single RHS) PCG using scalar dot products for speed.

        Internal helper called from :meth:`solve`. The 1-D path uses
        scalar ``torch.dot`` products which are roughly 2× faster than
        the column-wise sums used by the 2-D path.

        Args:
            operator: The matrix operator ``A``.
            preconditioner: The preconditioner ``P``.
            rhs: Right-hand side ``b`` of shape ``(n,)``.
            x: Current iterate (warm or cold).
            r: Current residual ``b - A x``.
            z: Preconditioned residual ``P r``.
            p: Current search direction.
            rhs_norm: ``||b||`` (precomputed).
            max_iter: Maximum iterations.

        Returns:
            Final iterate ``x`` of shape ``(n,)``.

        Raises:
            RuntimeError: On breakdown (negative curvature detected).

        """
        residual_z_old = torch.dot(r, z).item()
        relative_residual = float("inf")
        for iteration in range(max_iter):
            matrix_vector_product = operator(p)
            p_dot_ap = torch.dot(p, matrix_vector_product).item()
            # Cauchy-Schwarz bound on rounding error in ``p^T A p``:
            # ``|p^T A p| <= eps * ||p|| * ||A p||`` for symmetric ``A``.
            eps = (
                self.breakdown_eps
                if self.breakdown_eps is not None
                else torch.finfo(p.dtype).eps ** 0.5
            )
            if (
                p_dot_ap
                <= -eps
                * torch.linalg.norm(p).item()
                * torch.linalg.norm(matrix_vector_product).item()
            ):
                raise RuntimeError(
                    "PCG breakdown: non-positive curvature detected (p^T A p <= 0). "
                    "The operator may be indefinite or the preconditioner may be unsuitable."
                )

            alpha = residual_z_old / p_dot_ap
            if self.autograd_safe:
                # Out-of-place update required when backpropagating
                # through the iteration — PyTorch cannot record
                # gradients through in-place ``add_`` on a leaf tensor.
                x = x + alpha * p
                r = r - alpha * matrix_vector_product
            else:
                x.add_(p, alpha=alpha)
                r.add_(matrix_vector_product, alpha=-alpha)

            if self.restart_freq is not None and (iteration + 1) % self.restart_freq == 0:
                # Residual replacement: explicitly recompute ``r = b -
                # A x`` to drop round-off accumulated over many
                # in-place updates. Helpful in float64, harmful in
                # float32 (catastrophic cancellation).
                r = rhs - operator(x)

            self.residual_norm = torch.linalg.norm(r).item()
            relative_residual = self.residual_norm / rhs_norm

            if relative_residual <= self.tol:
                self.iterations = iteration + 1
                if self.verbose:
                    logger.info(
                        "PCG converged in %d iterations, rel_res=%.3e",
                        self.iterations,
                        relative_residual,
                    )
                return x

            z = preconditioner(r)
            residual_z_new = torch.dot(r, z).item()
            beta = residual_z_new / residual_z_old
            if self.autograd_safe:
                p = z + beta * p
            else:
                p.mul_(beta).add_(z)
            residual_z_old = residual_z_new

        self.iterations = max_iter
        if self.verbose:
            logger.warning(
                "PCG did not converge in %d iterations, rel_res=%.3e",
                max_iter,
                relative_residual,
            )
        return x

    def solve_2d(
        self,
        operator: Callable[[torch.Tensor], torch.Tensor],
        preconditioner: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        x: torch.Tensor,
        r: torch.Tensor,
        z: torch.Tensor,
        p: torch.Tensor,
        rhs_norm: float,
        max_iter: int,
    ) -> torch.Tensor:
        """2-D (batch RHS) PCG using column-wise vectorised dot products.

        Internal helper called from :meth:`solve`. The 2-D path handles
        ``k`` right-hand sides simultaneously by replacing scalar dot
        products with column-wise sums (``sum over axis 0``).

        Args:
            operator: The matrix operator ``A``.
            preconditioner: The preconditioner ``P``.
            rhs: Right-hand side ``B`` of shape ``(n, k)``.
            x: Current iterate (warm or cold).
            r: Current residual ``B - A X``.
            z: Preconditioned residual ``P R``.
            p: Current search direction.
            rhs_norm: ``||B||`` (precomputed).
            max_iter: Maximum iterations.

        Returns:
            Final iterate ``X`` of shape ``(n, k)``.

        Raises:
            RuntimeError: On breakdown (any column sees negative
                curvature).

        """
        residual_z_old = torch.sum(r * z, dim=0)
        for iteration in range(max_iter):
            matrix_vector_product = operator(p)
            p_dot_ap = torch.sum(p * matrix_vector_product, dim=0)
            eps = (
                self.breakdown_eps
                if self.breakdown_eps is not None
                else torch.finfo(p.dtype).eps ** 0.5
            )
            if torch.any(
                p_dot_ap
                <= -eps
                * torch.linalg.norm(p, dim=0)
                * torch.linalg.norm(matrix_vector_product, dim=0)
            ):
                raise RuntimeError(
                    "PCG breakdown: non-positive curvature detected (p^T A p <= 0). "
                    "The operator may be indefinite or the preconditioner may be unsuitable."
                )

            alpha = residual_z_old / p_dot_ap
            if self.autograd_safe:
                x = x + p * alpha.unsqueeze(0)
                r = r - matrix_vector_product * alpha.unsqueeze(0)
            else:
                x.add_(p * alpha.unsqueeze(0))
                r.add_(matrix_vector_product * (-alpha.unsqueeze(0)))

            if self.restart_freq is not None and (iteration + 1) % self.restart_freq == 0:
                r = rhs - operator(x)

            self.residual_norm = torch.linalg.norm(r).item()
            relative_residual = self.residual_norm / rhs_norm

            if relative_residual <= self.tol:
                self.iterations = iteration + 1
                if self.verbose:
                    logger.info(
                        "PCG converged in %d iterations, rel_res=%.3e",
                        self.iterations,
                        relative_residual,
                    )
                return x

            z = preconditioner(r)
            residual_z_new = torch.sum(r * z, dim=0)
            beta = residual_z_new / residual_z_old
            if self.autograd_safe:
                p = z + p * beta.unsqueeze(0)
            else:
                p.mul_(beta.unsqueeze(0)).add_(z)
            residual_z_old = residual_z_new

        self.iterations = max_iter
        if self.verbose:
            logger.warning(
                "PCG did not converge in %d iterations, rel_res=%.3e",
                max_iter,
                relative_residual,
            )
        return x


class GradientDescent:
    """Unpreconditioned gradient descent baseline for benchmarking.

    This solver is intentionally simple: it is used by the benchmark
    suite to provide a worst-case reference against which the LAKER
    preconditioner can be measured. It does **not** scale to large
    problems and is never used by the production estimator.

    Args:
        step_size: Fixed step size ``eta``. If ``None``, a conservative
            heuristic ``0.9 / max_eig(A)`` is computed on the fly via
            5 iterations of power iteration. The ``0.9`` factor leaves
            a safety margin below the divergence threshold ``1 / max_eig``.
        tol: Relative residual tolerance. Default ``1e-3`` (looser
            than PCG; GD is rarely expected to reach ``1e-10``).
        max_iter: Maximum iterations. Default ``50000``.
        verbose: Whether to log progress.

    Attributes:
        iterations: Number of GD iterations executed by the last
            :meth:`solve` call.
        residual_norm: ``||b - A x||`` at termination.

    """

    def __init__(
        self,
        step_size: Optional[float] = None,
        tol: float = 1e-3,
        max_iter: int = 50000,
        verbose: bool = False,
    ) -> None:
        """Initialise the gradient-descent solver.

        Args:
            step_size: Fixed step size ``eta`` (``None`` ⇒ auto via
                power iteration).
            tol: Relative residual tolerance.
            max_iter: Maximum iterations.
            verbose: Whether to log progress.

        """
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
        """Solve ``A x = b`` via gradient descent.

        Args:
            operator: Callable applying ``A`` to a vector.
            rhs: Right-hand side ``b``.
            x0: Initial guess (``None`` ⇒ zero vector).

        Returns:
            Approximate solution ``x`` of the same shape as ``rhs``.

        """
        if x0 is None:
            x = torch.zeros_like(rhs)
        else:
            x = x0.clone()

        # Estimate step size via 5 rounds of power iteration if not
        # provided. The ``0.9`` factor leaves a 10% safety margin below
        # ``1 / max_eig`` to avoid divergence on noisy eigenvalue
        # estimates.
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
            rel_res = self.residual_norm / b_norm
            if rel_res <= self.tol:
                self.iterations = iteration + 1
                if self.verbose:
                    logger.info("GD converged in %d iterations", self.iterations)
                return x
            x = x + eta * r

        self.iterations = self.max_iter
        if self.verbose:
            logger.warning(
                "GD did not converge in %d iterations, rel_res=%.3e",
                self.max_iter,
                rel_res,
            )
        return x


class JacobiPreconditioner:
    """Diagonal (Jacobi) preconditioner baseline.

    ``P = diag(\\lambda I + G)^{-1}`` as described in Section V-A-3 of
    the LAKER paper. Used by the benchmark suite as a fast-but-weak
    reference preconditioner.

    The diagonal is precomputed at construction; each call to
    :meth:`apply` is therefore an element-wise multiplication (no
    reduction).

    Args:
        diagonal: The diagonal of ``\\lambda I + G`` of shape ``(n,)``.

    Attributes:
        inv_diag: Cached ``1 / diag`` (clamped to ``finfo.eps``).

    """

    def __init__(self, diagonal: torch.Tensor) -> None:
        """Initialise the Jacobi preconditioner.

        Args:
            diagonal: Diagonal of ``\\lambda I + G``. Values very close
                to zero are clamped to ``finfo(dtype).eps`` so the
                inverse is always finite.

        """
        eps = torch.finfo(diagonal.dtype).eps
        self.inv_diag = 1.0 / diagonal.clamp(min=eps)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Jacobi preconditioner element-wise.

        Args:
            x: Vector of shape ``(n,)`` (or batch ``(n, k)``).

        Returns:
            ``P x`` of the same shape as ``x``.

        """
        return self.inv_diag * x