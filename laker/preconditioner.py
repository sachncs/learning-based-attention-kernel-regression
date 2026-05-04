"""Preconditioner learning via shrinkage-regularised CCCP."""

import logging
from typing import Callable, Optional

import torch

from laker.backend import get_default_device, get_default_dtype
from laker.utils import adaptive_shrinkage_rho, eigh_stable

logger = logging.getLogger(__name__)


def preconditioner_apply_core(
    x: torch.Tensor,
    isotropic_coef: float,
    q_basis: torch.Tensor,
    q_eigenvalues: torch.Tensor,
    q_eigenvectors: torch.Tensor,
) -> torch.Tensor:
    """Core of the preconditioner apply ``P = Sigma^{-1/2}``."""
    inv_sqrt_isotropic = isotropic_coef ** (-0.5)
    q_basis_projection = q_basis.T @ x
    coeffs = q_eigenvectors.T @ q_basis_projection
    if x.dim() == 1:
        coeffs = (q_eigenvalues.rsqrt() - inv_sqrt_isotropic) * coeffs
    else:
        coeffs = (q_eigenvalues.rsqrt() - inv_sqrt_isotropic).unsqueeze(-1) * coeffs
    q_correction = q_basis @ (q_eigenvectors @ coeffs)
    return inv_sqrt_isotropic * x + q_correction


class CCCPPreconditioner:
    """Learned data-dependent preconditioner for attention kernel regression.

    Implements Algorithm 1 (lines 4--13) from the LAKER paper. The
    preconditioner is obtained by solving a regularised MLE problem via the
    Convex-Concave Procedure (CCCP) with isotropic shrinkage. The resulting
    ``P = Sigma^{-1/2}`` is applied inside PCG to solve
    ``P (lambda I + G) alpha = P y``.

    For scalability the preconditioner is maintained in a **factored form**
    that exploits the fixed random-probe structure. Specifically ``Sigma``
    is always representable as ``isotropic_coef * I + Q_basis * q_basis_matrix * Q_basis^T``
    where ``Q_basis`` is an orthonormal basis for the span of the random probes.
    This reduces the per-iteration cost from ``O(n^3)`` to ``O(N_r^3)``,
    independent of the problem size ``n``.

    Args:
        num_probes: Number of random directions ``N_r``. If ``None``, an
            adaptive heuristic ``max(200, int(2 * sqrt(n)))`` is used.
        gamma: CCCP regularisation parameter ``gamma`` (paper default 0.1).
        epsilon: Numerical safeguard in the denominator of Eq. (35).
        base_rho: Base shrinkage parameter when ``N_r >= n``.
        max_iter: Maximum CCCP iterations.
        tol: Convergence tolerance on the relative change of ``isotropic_coef``
            and ``q_basis_matrix``.
        verbose: Whether to log progress.
        device: torch device.
        dtype: torch dtype.

    """

    def __init__(
        self,
        num_probes: Optional[int] = None,
        gamma: float = 1e-1,
        epsilon: float = 1e-8,
        base_rho: float = 0.05,
        max_iter: int = 200,
        tol: float = 1e-6,
        verbose: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        probe_strategy: str = "gaussian",
        power_iter_steps: int = 3,
    ) -> None:
        """Initialise the CCCP preconditioner."""
        self.num_probes = num_probes
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.base_rho = float(base_rho)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = verbose
        self.probe_strategy = probe_strategy
        self.power_iter_steps = int(power_iter_steps)

        if device is None:
            device = get_default_device()
        if dtype is None:
            dtype = get_default_dtype()
        self.device = device
        self.dtype = dtype

        # Internal state (populated by ``build``)
        self.n: Optional[int] = None
        self.nr: Optional[int] = None
        self.q_basis: Optional[torch.Tensor] = None  # (n, nr) orthonormal
        self.qr_r: Optional[torch.Tensor] = None  # (nr, nr) from QR
        self.isotropic_coef: Optional[float] = None
        self.q_basis_matrix: Optional[torch.Tensor] = None  # (nr, nr)
        self.q_eigenvalues: Optional[torch.Tensor] = None  # (nr,)
        self.q_eigenvectors: Optional[torch.Tensor] = None  # (nr, nr)

    def build(
        self,
        operator: Callable[[torch.Tensor], torch.Tensor],
        n: int,
        seed: Optional[int] = None,
    ) -> "CCCPPreconditioner":
        """Learn the preconditioner for an ``n x n`` operator.

        Args:
            operator: Callable that applies ``(lambda I + G)`` to a vector.
            n: Problem dimension.
            seed: Optional random seed for probe generation.

        Returns:
            ``self`` for method chaining.

        """
        self.n = n
        nr = self.num_probes if self.num_probes is not None else max(200, int(2 * n**0.5))
        self.nr = min(nr, n)
        if self.verbose:
            logger.info("Building CCCP preconditioner: n=%d, N_r=%d", n, self.nr)

        # 1. Generate random probes and normalise (Eq. 14)
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)
        random_probes = torch.randn(n, self.nr, device=self.device, dtype=self.dtype, generator=gen)

        # Optional: power-iteration-biased probe directions
        if self.probe_strategy == "power_iter":
            # To avoid breaking the CCCP MLE (which assumes isotropic Gaussian
            # probes), only a subset of the probes are power-iterated; the rest
            # remain pure Gaussian.  This preserves unbiased spectrum coverage
            # while concentrating extra mass on the dominant eigenspace.
            n_power = max(1, int(self.nr * 0.25))
            power_block = random_probes[:, :n_power]
            for _ in range(self.power_iter_steps):
                power_block = operator(power_block)
                power_block = torch.linalg.qr(power_block, mode="reduced")[0]
            random_probes = torch.cat([power_block, random_probes[:, n_power:]], dim=1)

        operator_probes = operator(random_probes)
        probe_norms = torch.linalg.norm(operator_probes, dim=0, keepdim=True)
        normalized_probes = operator_probes / probe_norms.clamp(min=self.epsilon)

        # 2. Compute fixed QR basis: U = Q R  (economy QR)
        q_basis, qr_r = torch.linalg.qr(normalized_probes, mode="reduced")
        self.q_basis = q_basis  # (n, nr)
        self.qr_r = qr_r  # (nr, nr)

        # 3. CCCP iteration (Eq. 35-37)
        shrinkage = adaptive_shrinkage_rho(self.nr, n, self.gamma, self.base_rho)
        regularization_scale = 1.0 + self.gamma / n
        isotropic_coef = 1.0
        q_basis_matrix = torch.zeros(self.nr, self.nr, device=self.device, dtype=self.dtype)

        # Pre-allocate buffers reused each iteration
        eye_nr = torch.eye(self.nr, device=self.device, dtype=self.dtype)
        factored_matrix = torch.empty(self.nr, self.nr, device=self.device, dtype=self.dtype)
        f_gamma_q_basis = torch.empty(self.nr, self.nr, device=self.device, dtype=self.dtype)
        shrunken_f_gamma = torch.empty(self.nr, self.nr, device=self.device, dtype=self.dtype)

        for iteration in range(self.max_iter):
            isotropic_coef_prev = isotropic_coef
            q_basis_matrix_prev = q_basis_matrix.clone()

            # Form M = isotropic_coef * I + q_basis_matrix and decompose
            torch.add(eye_nr * isotropic_coef, q_basis_matrix, out=factored_matrix)
            factored_eigvals, factored_eigvecs = eigh_stable(factored_matrix, eps=self.epsilon)

            # Compute denominators: denom_k = (R^T M^{-1} R)_{kk} + epsilon.
            # Avoid materialising the full inverse by working in the eigenbasis:
            #   R^T M^{-1} R = (V^T R)^T * diag(1/eig) * (V^T R)
            eigenbasis_projection = factored_eigvecs.T @ self.qr_r  # (nr, nr)
            # Scale columns of eigenbasis_projection by reciprocal eigenvalues
            scaled_eigenbasis_projection = (
                factored_eigvals.reciprocal().unsqueeze(-1) * eigenbasis_projection
            )
            inverse_m_r_product = eigenbasis_projection.T @ scaled_eigenbasis_projection
            probe_denominators = torch.diagonal(inverse_m_r_product) + self.epsilon

            # Weights w_k = (n / N_r) / denom_k
            probe_weights = (n / self.nr) / probe_denominators  # (nr,)

            # Build F_gamma in Q-basis:
            #   (1/(1+gamma/n)) * (sum_k w_k * u_bar_k u_bar_k^T + gamma I)
            # In Q basis: u_bar u_bar^T = Q R W R^T Q^T where W = diag(weights)
            r_weighted_by_probe = self.qr_r * probe_weights.unsqueeze(0)  # (nr, nr)
            weighted_r_r_transpose = r_weighted_by_probe @ self.qr_r.T  # (nr, nr)
            torch.mul(
                weighted_r_r_transpose + self.gamma * eye_nr,
                1.0 / regularization_scale,
                out=f_gamma_q_basis,
            )

            # Shrinkage: (1-rho) * F + rho * I
            torch.lerp(f_gamma_q_basis, eye_nr, shrinkage, out=shrunken_f_gamma)

            # Normalisation: preserve trace(Sigma) = n.
            shrunken_isotropic_coef = (
                1.0 - shrinkage
            ) * self.gamma / regularization_scale + shrinkage
            full_space_trace = (
                shrunken_isotropic_coef * (self.n - self.nr) + shrunken_f_gamma.diagonal().sum()
            )
            trace_scale = self.n / full_space_trace

            isotropic_coef = trace_scale * shrunken_isotropic_coef
            # q_basis_matrix = trace_scale * shrunken_f_gamma - isotropic_coef * I
            torch.mul(shrunken_f_gamma, trace_scale, out=q_basis_matrix)
            q_basis_matrix.diagonal().sub_(isotropic_coef)

            # Convergence check
            isotropic_rel = abs(isotropic_coef - isotropic_coef_prev) / (
                abs(isotropic_coef_prev) + 1e-12
            )
            q_basis_rel = torch.norm(q_basis_matrix - q_basis_matrix_prev, p="fro").item() / (
                torch.norm(q_basis_matrix_prev, p="fro").item() + 1e-12
            )
            if self.verbose and (iteration % 10 == 0 or max(isotropic_rel, q_basis_rel) < self.tol):
                logger.info(
                    "CCCP iter %d: isotropic_rel=%.3e, q_basis_rel=%.3e, isotropic_coef=%.6f",
                    iteration,
                    isotropic_rel,
                    q_basis_rel,
                    isotropic_coef,
                )
            if max(isotropic_rel, q_basis_rel) < self.tol:
                break

        self.isotropic_coef = isotropic_coef
        self.q_basis_matrix = q_basis_matrix

        # Final eigendecomposition of M = isotropic_coef * I + q_basis_matrix for fast applies
        torch.add(eye_nr * isotropic_coef, q_basis_matrix, out=factored_matrix)
        self.q_eigenvalues, self.q_eigenvectors = eigh_stable(factored_matrix, eps=self.epsilon)

        if self.verbose:
            logger.info("CCCP preconditioner built in %d iterations", iteration + 1)
        return self

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the preconditioner ``P = Sigma^{-1/2}`` to vector(s).

        Automatically dispatches to the 1-D or 2-D implementation based on
        the dimensionality of ``x``.

        Args:
            x: Tensor of shape ``(n,)`` or ``(n, k)``.

        Returns:
            Tensor of the same shape as ``x``.

        Raises:
            RuntimeError: If the preconditioner has not been built.
            ValueError: If ``x`` is not 1-D or 2-D.

        """
        if self.q_basis is None:
            raise RuntimeError("Preconditioner has not been built. Call build() first.")

        if x.dim() == 1:
            return self.apply_1d(x)
        if x.dim() == 2:
            return self.apply_2d(x)
        raise ValueError(f"x must be 1-D or 2-D, got shape {x.shape}")

    def apply_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Shared implementation for apply_1d and apply_2d.

        Args:
            x: Tensor of shape ``(n,)`` or ``(n, k)``.

        Returns:
            Tensor of the same shape as ``x``.

        """
        return preconditioner_apply_core(
            x,
            self.isotropic_coef,
            self.q_basis,
            self.q_eigenvalues,
            self.q_eigenvectors,
        )

    def apply_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply P to a single vector.

        Args:
            x: Tensor of shape ``(n,)``.

        Returns:
            Tensor of shape ``(n,)``.

        """
        return self.apply_impl(x)

    def apply_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply P to a batch of vectors (matrix).

        Args:
            x: Tensor of shape ``(n, k)``.

        Returns:
            Tensor of shape ``(n, k)``.

        """
        return self.apply_impl(x)

    def to_dense(self) -> torch.Tensor:
        """Materialise the full dense preconditioner matrix ``P = Sigma^{-1/2}``.

        Warning:
            ``O(n^2)`` memory. For debugging only.

        Returns:
            Dense matrix of shape ``(n, n)``.

        Raises:
            RuntimeError: If the preconditioner has not been built.

        """
        if self.q_basis is None:
            raise RuntimeError("Preconditioner has not been built.")
        n = self.q_basis.shape[0]
        preconditioner_covariance = self.isotropic_coef * torch.eye(
            n, device=self.device, dtype=self.dtype
        )
        preconditioner_covariance = (
            preconditioner_covariance + self.q_basis @ self.q_basis_matrix @ self.q_basis.T
        )
        # Compute Sigma^{-1/2} via eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(preconditioner_covariance)
        eigvals = eigvals.clamp(min=self.epsilon)
        return eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T


class AdaptivePreconditioner:
    """Lightweight policy that selects among preconditioners based on cheap diagnostics.

    Runs a small number of random probes to estimate the condition number of
    the operator via power iteration, then chooses:
    - ``JacobiPreconditioner`` if ``κ < 1e3`` (fast, well-conditioned).
    - ``CCCPPreconditioner`` if ``κ < 1e6`` (robust, moderate conditioning).
    - ``CCCPPreconditioner`` with increased probes if ``κ >= 1e6``.

    Args:
        gamma: CCCP regularisation parameter.
        num_probes: Base number of random directions for CCCP.
        epsilon: Numerical safeguard.
        base_rho: Base shrinkage parameter.
        max_iter: Maximum CCCP iterations.
        tol: CCCP convergence tolerance.
        verbose: Whether to log progress.
        device: torch device.
        dtype: torch dtype.

    """

    def __init__(
        self,
        gamma: float = 1e-1,
        num_probes: Optional[int] = None,
        epsilon: float = 1e-8,
        base_rho: float = 0.05,
        max_iter: int = 200,
        tol: float = 1e-6,
        verbose: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        probe_strategy: str = "gaussian",
        power_iter_steps: int = 3,
    ) -> None:
        """Initialise the adaptive preconditioner."""
        self.gamma = float(gamma)
        self.num_probes = num_probes
        self.epsilon = float(epsilon)
        self.base_rho = float(base_rho)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = verbose
        self.probe_strategy = probe_strategy
        self.power_iter_steps = int(power_iter_steps)
        if device is None:
            device = get_default_device()
        if dtype is None:
            dtype = get_default_dtype()
        self.device = device
        self.dtype = dtype
        self._inner = None
        self._inner_name = None

    def build(
        self,
        operator: Callable[[torch.Tensor], torch.Tensor],
        n: int,
        diagonal: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> "AdaptivePreconditioner":
        """Run diagnostics and select a preconditioner."""
        num_diag_probes = min(10, n)
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)
        probes = torch.randn(
            n,
            num_diag_probes,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        op_probes = operator(probes)

        # Power iteration for largest eigenvalue on a single probe
        v = op_probes[:, 0].clone()
        for _ in range(5):
            v = operator(v.unsqueeze(1)).squeeze(1)
            v = v / torch.linalg.norm(v)
        lambda_max = torch.dot(v, operator(v)).item()

        # Estimate smallest eigenvalue via Rayleigh quotient on random probes
        rayleighs = torch.sum(probes * op_probes, dim=0) / torch.sum(probes**2, dim=0)
        lambda_min = rayleighs.min().item()

        cond = lambda_max / max(lambda_min, 1e-12)

        if cond < 1e3 and diagonal is not None:
            from laker.solvers import JacobiPreconditioner

            self._inner = JacobiPreconditioner(diagonal)
            self._inner_name = "jacobi"
            if self.verbose:
                logger.info("AdaptivePreconditioner selected Jacobi (κ≈%.2e)", cond)
        elif cond < 1e6:
            cccp = CCCPPreconditioner(
                num_probes=self.num_probes,
                gamma=self.gamma,
                epsilon=self.epsilon,
                base_rho=self.base_rho,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                device=self.device,
                dtype=self.dtype,
                probe_strategy=self.probe_strategy,
                power_iter_steps=self.power_iter_steps,
            )
            cccp.build(operator, n, seed=seed)
            self._inner = cccp
            self._inner_name = "cccp"
            if self.verbose:
                logger.info("AdaptivePreconditioner selected CCCP (κ≈%.2e)", cond)
        else:
            cccp = CCCPPreconditioner(
                num_probes=(
                    self.num_probes if self.num_probes is not None else max(200, int(2 * n**0.5))
                )
                * 2,
                gamma=self.gamma,
                epsilon=self.epsilon,
                base_rho=self.base_rho,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                device=self.device,
                dtype=self.dtype,
                probe_strategy=self.probe_strategy,
                power_iter_steps=self.power_iter_steps,
            )
            cccp.build(operator, n, seed=seed)
            self._inner = cccp
            self._inner_name = "cccp_aggressive"
            if self.verbose:
                logger.info(
                    "AdaptivePreconditioner selected aggressive CCCP (κ≈%.2e)",
                    cond,
                )
        return self

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the selected preconditioner."""
        if self._inner is None:
            raise RuntimeError("Preconditioner has not been built. Call build() first.")
        if self._inner_name == "jacobi":
            return self._inner.apply(x)
        return self._inner.apply(x)
