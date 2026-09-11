"""Math, RNG, and GP surrogate helpers.

Single public class :class:`Math` with single-word static methods.
GPSurrogate is exposed as :class:`GP` (Bayesian-optimisation surrogate).
"""

from __future__ import annotations

import math as _math
import os
from typing import Optional

import numpy as np
import torch


class Math:
    """Numerical and RNG helpers used across the package."""

    @staticmethod
    def exp(gram: torch.Tensor, skip: bool = False) -> torch.Tensor:
        """Element-wise exp with dtype-aware overflow guard."""
        if not skip:
            if gram.dtype == torch.float16:
                cap = 11.0
            elif gram.dtype == torch.bfloat16:
                cap = 80.0
            elif gram.dtype == torch.float32:
                cap = 80.0
            else:
                cap = 700.0
            if gram.requires_grad:
                return torch.exp(gram.clamp(max=cap))
            out = gram.clone()
            out.clamp_(max=cap)
            return torch.exp(out, out=out)
        if gram.requires_grad:
            return torch.exp(gram)
        return torch.exp(gram)

    @staticmethod
    def pdf(x: torch.Tensor) -> torch.Tensor:
        """Standard normal PDF."""
        return torch.exp(-0.5 * x**2) / _math.sqrt(2.0 * _math.pi)

    @staticmethod
    def cdf(x: torch.Tensor) -> torch.Tensor:
        """Standard normal CDF via Abramowitz & Stegun 7.1.26."""
        ax = torch.abs(x)
        t = 1.0 / (1.0 + 0.2316419 * ax)
        poly = t * (
            0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
        )
        p = Math.pdf(ax)
        tail = p * poly
        # 1 - tail for x >= 0, tail for x < 0.
        return torch.where(x >= 0, 1.0 - tail, tail)

    @staticmethod
    def sinh(x: torch.Tensor) -> torch.Tensor:
        return torch.sinh(x)

    @staticmethod
    def normalize(mat: torch.Tensor) -> torch.Tensor:
        """Scale a positive-definite matrix so ``trace = n``."""
        n = mat.shape[0]
        tr = torch.trace(mat)
        if torch.abs(tr) < 1e-30:
            return mat
        return mat * (n / tr)

    @staticmethod
    def latin(n: int, dims: int, seed: Optional[int] = None) -> np.ndarray:
        """Latin-hypercube sample of shape ``(n, dims)`` in ``[0, 1]``."""
        rng = np.random.default_rng(seed)
        cut = np.linspace(0.0, 1.0, n + 1)
        u = rng.uniform(0.0, 1.0, size=(dims, n))
        return (cut[:-1] + u * (1.0 / n)).T

    @staticmethod
    def power(operator, n: int, num: int = 10, seed: int = 0) -> float:
        """Estimate the spectral norm of a linear operator via power iteration."""
        gen = torch.Generator().manual_seed(seed)
        v = torch.randn(n, generator=gen)
        v = v / v.norm()
        norm = 0.0
        for _ in range(num):
            u = operator(v)
            norm = u.norm().item()
            if norm == 0.0:
                return 0.0
            v = u / norm
        return norm

    @staticmethod
    def chol(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Cholesky factor with diagonal-jitter fallback."""
        try:
            return torch.linalg.cholesky(matrix)
        except RuntimeError:
            n = matrix.shape[0]
            jitter = eps * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
            return torch.linalg.cholesky(matrix + jitter)

    @staticmethod
    def bytes(dtype: torch.dtype) -> int:
        """Bytes per scalar for memory budget calculations."""
        if hasattr(dtype, "itemsize"):
            return dtype.itemsize
        return 4

    @staticmethod
    def inv(diagonal: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Invert a diagonal, clamping small/negative entries to ``eps``."""
        return 1.0 / diagonal.clamp(min=eps)

    @staticmethod
    def seed() -> Optional[int]:
        """Return the global LAKER seed if set."""
        v = os.environ.get("LAKER_SEED")
        return int(v) if v else None

    @staticmethod
    def seed_set(value: int) -> None:
        """Seed torch and numpy RNG, persist ``LAKER_SEED`` env var."""
        torch.manual_seed(int(value))
        np.random.default_rng(int(value))
        os.environ["LAKER_SEED"] = str(int(value))

    @staticmethod
    def dense(sparse: torch.Tensor) -> torch.Tensor:
        """Materialise a sparse tensor to dense, if needed."""
        if sparse.is_sparse:
            return sparse.to_dense()
        return sparse

    @staticmethod
    def shrink(
        probes: int,
        size: int,
        gamma: float,
        base: float = 0.05,
    ) -> float:
        """Adaptive shrinkage ``rho`` for the CCCP preconditioner."""
        if probes >= size:
            return base
        ratio = probes / size
        return float(min(base + (1.0 - base) * (1.0 - ratio) * min(1.0, gamma * 10.0), 0.5))

    @staticmethod
    def eigh(matrix: torch.Tensor, eps: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor]:
        """Eigendecomposition with eigenvalue clamping for PSD safety."""
        vals, vecs = torch.linalg.eigh(matrix)
        vals = vals.clamp(min=eps)
        return vals, vecs


class GP:
    """GP surrogate for Bayesian Optimisation.

    Operates on a fixed ``(d, 2)`` bounds array. Dimensions listed in
    ``log_indices`` are mapped to log-space before the kernel evaluation.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        log: Optional[list[int]] = None,
        sigma: float = 1.0,
        scale: float = 0.2,
        noise: float = 1e-4,
    ) -> None:
        self.bounds: np.ndarray = bounds.astype(np.float64)
        self.d = bounds.shape[0]
        self.log = list(log or [])
        self.sigma = float(sigma)
        self.scale = float(scale)
        self.noise = float(noise)
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.l: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.length = float(scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float64)
        z = x.copy()
        for i in self.log:
            log_lo = np.log10(max(self.bounds[i, 0] * 0.1, 1e-12))
            log_hi = np.log10(float(self.bounds[i, 1]) * 10.0)
            z[:, i] = np.log10(np.clip(z[:, i], 10.0**log_lo, 10.0**log_hi))
            z[:, i] = (z[:, i] - log_lo) / (log_hi - log_lo)
        non_log = [i for i in range(self.d) if i not in set(self.log)]
        if non_log:
            lin_lo = self.bounds[non_log, 0]
            lin_hi = self.bounds[non_log, 1]
            z[:, non_log] = (z[:, non_log] - lin_lo) / (lin_hi - lin_lo)
        return np.clip(z, 0.0, 1.0)

    def kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        sq = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
        return self.sigma**2 * np.exp(-0.5 * sq / (self.length**2 + 1e-12))

    def ml(self, candidate: float) -> float:
        old = self.length
        self.length = candidate
        try:
            k = self.kernel(self.x, self.x)
            k[np.diag_indices_from(k)] += self.noise**2
            try:
                l = np.linalg.cholesky(k + 1e-8 * np.eye(k.shape[0]))
                a = np.linalg.solve(l.T, np.linalg.solve(l, self.y))
                return (
                    -0.5 * float(np.dot(self.y, a))
                    - float(np.sum(np.log(np.diag(l))))
                    - 0.5 * k.shape[0] * np.log(2 * _math.pi)
                )
            except np.linalg.LinAlgError:
                return float("-inf")
        finally:
            self.length = old

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP to observations ``(x, y)`` with marginal-likelihood tuning."""
        self.x = self.transform(x)
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) + 1e-8
        self.y = (y - self.y_mean) / self.y_std
        best = self.length
        bestml = float("-inf")
        for cand in np.logspace(-2, 0, 20):
            ml = self.ml(float(cand))
            if ml > bestml:
                bestml = ml
                best = float(cand)
        self.length = best
        k = self.kernel(self.x, self.x)
        k[np.diag_indices_from(k)] += self.noise**2
        self.l = np.linalg.cholesky(k + 1e-8 * np.eye(k.shape[0]))
        self.alpha = np.linalg.solve(self.l.T, np.linalg.solve(self.l, self.y))

    def predict(self, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, var)`` predictive distribution at ``x_new``."""
        x_new_t = self.transform(x_new)
        k_s = self.kernel(self.x, x_new_t)
        k_ss = self.kernel(x_new_t, x_new_t)
        k_ss[np.diag_indices_from(k_ss)] += self.noise**2
        v = np.linalg.solve(self.l, k_s)
        mu = np.dot(k_s.T, self.alpha)
        var = np.diag(k_ss) - np.sum(v**2, axis=0)
        var = np.clip(var, 1e-12, None)
        mu = mu * self.y_std + self.y_mean
        var = var * (self.y_std**2)
        return mu, var

    def improve(self, x_new: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement at ``x_new``."""
        mu, var = self.predict(x_new)
        sigma = np.sqrt(var)
        best = float(self.y.min()) * self.y_std + self.y_mean
        with np.errstate(divide="warn", invalid="warn"):
            z = (best - mu - xi) / (sigma + 1e-12)
        ei = (best - mu - xi) * cdf_np(z) + sigma * pdf_np(z)
        ei[sigma < 1e-12] = 0.0
        return ei


def pdf_np(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF for numpy arrays."""
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * _math.pi)


def cdf_np(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via Abramowitz & Stegun 7.1.26."""
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911
    sign = np.sign(x)
    x = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * np.exp(-x * x))
    return 0.5 * (1.0 + sign * y)


__all__ = ["Math", "GP", "pdf_np", "cdf_np"]
