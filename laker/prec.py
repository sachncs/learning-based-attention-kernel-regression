"""Preconditioner learning via shrinkage-regularised CCCP.

Public classes:

* :class:`CCCP` — shrinkage-regularised Convex-Concave Procedure.
* :class:`Adaptive` — wrapper that picks among Jacobi / CCCP / aggressive CCCP.

The preconditioner is maintained in a **factored form**:
``Sigma = c * I + Q B Q^T`` where ``Q`` is an orthonormal basis for the
operator-probed directions, ``B`` is a small dense matrix, and ``c`` is
an isotropic scalar. This reduces per-iteration cost from ``O(n^3)`` to
``O(N_r^3)``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch

from laker.backend import Backend
from laker.math import Math

if TYPE_CHECKING:
    from laker.solve import Jacobi

logger = logging.getLogger(__name__)


def apply_core(
    x: torch.Tensor,
    iso: float,
    basis: torch.Tensor,
    evals: torch.Tensor,
    evecs: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Core of the preconditioner apply ``P = Sigma^{-1/2}``."""
    iso_safe = max(float(iso), eps)
    inv_sqrt_iso = iso_safe ** (-0.5)
    proj = basis.T @ x
    coeffs = evecs.T @ proj
    if x.dim() == 1:
        coeffs = (evals.rsqrt() - inv_sqrt_iso) * coeffs
    else:
        coeffs = (evals.rsqrt() - inv_sqrt_iso).unsqueeze(-1) * coeffs
    return inv_sqrt_iso * x + basis @ (evecs @ coeffs)


class CCCP:
    """Learned data-dependent preconditioner for attention kernel regression.

    Implements Algorithm 1 (lines 4-13) of the LAKER paper. Builds
    ``P = Sigma^{-1/2}`` from random probe applications using the
    Convex-Concave Procedure (CCCP).
    """

    def __init__(
        self,
        num: Optional[int] = None,
        gamma: float = 1e-1,
        eps: float = 1e-8,
        base: float = 0.05,
        max_iter: int = 200,
        tol: float = 1e-6,
        verbose: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        probe: str = "gaussian",
        power: int = 3,
    ) -> None:
        self.num = num
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.base = float(base)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = verbose
        self.probe = probe
        self.power = int(power)

        if device is None:
            device = Backend.device
        if dtype is None:
            dtype = Backend.dtype
        self.device = device
        self.dtype = dtype

        # Problem dimensions (populated by build()).
        self.size: Optional[int] = None
        self.num_probes: Optional[int] = None

        # Spectral factors from the probe QR decomposition.
        self.basis: Optional[torch.Tensor] = None
        self.tri_factor: Optional[torch.Tensor] = None

        # CCCP iteration state (populated by build()).
        self.iso: Optional[float] = None
        self.b: Optional[torch.Tensor] = None
        self.evals: Optional[torch.Tensor] = None
        self.evecs: Optional[torch.Tensor] = None

    def build(
        self,
        op: Callable[[torch.Tensor], torch.Tensor],
        n: int,
        seed: Optional[int] = None,
    ) -> "CCCP":
        """Learn the preconditioner for an ``n x n`` operator."""
        self.size = n
        nr = self.num if self.num is not None else max(200, int(2 * n**0.5))
        self.num_probes = min(nr, n)
        if self.verbose:
            logger.info("Building CCCP preconditioner: n=%d, N_r=%d", n, self.num_probes)

        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)
        probes = torch.randn(
            n, self.num_probes, device=self.device, dtype=self.dtype, generator=gen
        )

        if self.probe == "power":
            n_power = max(1, int(self.num_probes * 0.25))
            power_block = probes[:, :n_power]
            for _ in range(self.power):
                power_block = op(power_block)
                power_block = torch.linalg.qr(power_block, mode="reduced")[0]
            probes = torch.cat([power_block, probes[:, n_power:]], dim=1)

        probed = op(probes)
        norms = torch.linalg.norm(probed, dim=0, keepdim=True)
        unit = probed / norms.clamp(min=self.eps)

        ortho_basis, tri_factor = torch.linalg.qr(unit, mode="reduced")
        self.basis = ortho_basis
        self.tri_factor = tri_factor

        rho = Math.shrink(self.num_probes, n, self.gamma, self.base)
        reg = 1.0 + self.gamma / n
        iso = 1.0
        b = torch.zeros(self.num_probes, self.num_probes, device=self.device, dtype=self.dtype)

        eye = torch.eye(self.num_probes, device=self.device, dtype=self.dtype)
        matrix_buf = torch.empty(
            self.num_probes, self.num_probes, device=self.device, dtype=self.dtype
        )
        f_gamma_buf = torch.empty(
            self.num_probes, self.num_probes, device=self.device, dtype=self.dtype
        )
        shrunk_buf = torch.empty(
            self.num_probes, self.num_probes, device=self.device, dtype=self.dtype
        )

        last_iter = 0
        for it in range(self.max_iter):
            iso_prev = iso
            b_prev = b.clone()

            torch.add(eye * iso, b, out=matrix_buf)
            evals, evecs = Math.eigh(matrix_buf, eps=self.eps)

            proj = evecs.T @ self.tri_factor
            scaled_proj = evals.reciprocal().unsqueeze(-1) * proj
            inv_proj = proj.T @ scaled_proj
            denominators = torch.diagonal(inv_proj) + self.eps

            weights = (n / self.num_probes) / denominators

            weighted_factor = self.tri_factor * weights.unsqueeze(0)
            quadratic = weighted_factor @ self.tri_factor.T
            torch.mul(quadratic + self.gamma * eye, 1.0 / reg, out=f_gamma_buf)

            torch.lerp(f_gamma_buf, eye, rho, out=shrunk_buf)

            iso_shrunk = (1.0 - rho) * self.gamma / reg + rho
            full_trace = iso_shrunk * (self.size - self.num_probes) + shrunk_buf.diagonal().sum()
            trace_scale = self.size / full_trace

            iso = trace_scale * iso_shrunk
            torch.mul(shrunk_buf, trace_scale, out=b)
            b.diagonal().sub_(iso)

            iso_rel = abs(iso - iso_prev) / (abs(iso_prev) + 1e-12)
            b_rel = torch.norm(b - b_prev, p="fro").item() / (
                torch.norm(b_prev, p="fro").item() + 1e-12
            )
            if self.verbose and (it % 10 == 0 or max(iso_rel, b_rel) < self.tol):
                logger.info(
                    "CCCP iter %d: iso_rel=%.3e, b_rel=%.3e, iso=%.6f",
                    it,
                    iso_rel,
                    b_rel,
                    iso,
                )
            if max(iso_rel, b_rel) < self.tol:
                last_iter = it
                break
            last_iter = it

        self.iso = iso
        self.b = b

        torch.add(eye * iso, b, out=matrix_buf)
        self.evals, self.evecs = Math.eigh(matrix_buf, eps=self.eps)
        # Reuse scratch buffer to avoid extra allocation.

        if not (self.iso == self.iso) or self.iso <= 0.0:
            raise RuntimeError(
                f"CCCP preconditioner produced non-positive iso={self.iso!r} after "
                f"{last_iter + 1} iterations; check gamma and the operator spectrum."
            )

        if self.verbose:
            logger.info("CCCP preconditioner built in %d iterations", last_iter + 1)
        return self

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``P = Sigma^{-1/2}`` to vector(s)."""
        if self.basis is None:
            raise RuntimeError("Preconditioner has not been built. Call build() first.")
        if x.dim() not in (1, 2):
            raise ValueError(f"x must be 1-D or 2-D, got shape {x.shape}")
        return apply_core(x, self.iso, self.basis, self.evals, self.evecs, eps=self.eps)

    def dense(self) -> torch.Tensor:
        """Materialise the full dense preconditioner matrix (debug only)."""
        if self.basis is None:
            raise RuntimeError("Preconditioner has not been built.")
        n = self.basis.shape[0]
        cov = self.iso * torch.eye(n, device=self.device, dtype=self.dtype)
        cov = cov + self.basis @ self.b @ self.basis.T
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.clamp(min=self.eps)
        return evecs @ torch.diag(evals.rsqrt()) @ evecs.T


class Adaptive:
    """Lightweight policy that selects a preconditioner based on cheap diagnostics.

    Runs power iteration on random probes to estimate ``kappa`` of the
    operator and selects:

    * ``Jacobi`` if ``kappa < 1e3``.
    * ``CCCP`` if ``kappa < 1e6``.
    * ``CCCP`` with doubled probe budget if ``kappa >= 1e6``.
    """

    def __init__(
        self,
        gamma: float = 1e-1,
        num: Optional[int] = None,
        eps: float = 1e-8,
        base: float = 0.05,
        max_iter: int = 200,
        tol: float = 1e-6,
        verbose: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        probe: str = "gaussian",
        power: int = 3,
    ) -> None:
        self.gamma = float(gamma)
        self.num = num
        self.eps = float(eps)
        self.base = float(base)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = verbose
        self.probe = probe
        self.power = int(power)
        if device is None:
            device = Backend.device
        if dtype is None:
            dtype = Backend.dtype
        self.device = device
        self.dtype = dtype
        self.inner: Optional[Union["Jacobi", CCCP]] = None
        self.choice: Optional[str] = None

    def build(
        self,
        op: Callable[[torch.Tensor], torch.Tensor],
        n: int,
        diag: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> "Adaptive":
        """Run diagnostics and select a preconditioner."""
        num_diag = min(10, n)
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)
        probes = torch.randn(n, num_diag, device=self.device, dtype=self.dtype, generator=gen)
        probed = op(probes)

        v = probed[:, 0].clone()
        for _ in range(5):
            v = op(v.unsqueeze(1)).squeeze(1)
            v = v / torch.linalg.norm(v)
        lam_max = torch.dot(v, op(v)).item()

        rayleighs = torch.sum(probes * probed, dim=0) / torch.sum(probes**2, dim=0)
        lam_min = rayleighs.min().item()

        cond = lam_max / max(lam_min, 1e-12)

        if cond < 1e3 and diag is not None:
            from laker.solve import Jacobi

            self.inner = Jacobi(diag)
            self.choice = "jacobi"
            if self.verbose:
                logger.info("Adaptive chose Jacobi (κ≈%.2e)", cond)
        elif cond < 1e6:
            self.inner = self.make(num=self.num)
            self.inner.build(op, n, seed=seed)
            self.choice = "cccp"
            if self.verbose:
                logger.info("Adaptive chose CCCP (κ≈%.2e)", cond)
        else:
            aggressive_num = (self.num if self.num is not None else max(200, int(2 * n**0.5))) * 2
            self.inner = self.make(num=aggressive_num)
            self.inner.build(op, n, seed=seed)
            self.choice = "aggressive"
            if self.verbose:
                logger.info("Adaptive chose aggressive CCCP (κ≈%.2e)", cond)
        return self

    def make(self, num: Optional[int]) -> CCCP:
        return CCCP(
            num=num,
            gamma=self.gamma,
            eps=self.eps,
            base=self.base,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            device=self.device,
            dtype=self.dtype,
            probe=self.probe,
            power=self.power,
        )

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.inner is None:
            raise RuntimeError("Preconditioner has not been built. Call build() first.")
        return self.inner.apply(x)


__all__ = ["CCCP", "Adaptive", "apply_core"]
