"""Utility functions for numerical stability and convergence."""

import math
from typing import Optional

import numpy
import torch


def trace_normalize(mat: torch.Tensor) -> torch.Tensor:
    """Normalize a positive-definite matrix so that ``trace(mat) == n``.

    This corresponds to Eq. (34) and (37) in the LAKER paper.

    Args:
        mat: Square tensor of shape ``(n, n)``.

    Returns:
        Normalized matrix with unit mean eigenvalue.

    """
    n = mat.shape[0]
    trace = torch.trace(mat)
    return mat / (trace / n)


def adaptive_shrinkage_rho(
    num_probes: int,
    problem_size: int,
    gamma: float,
    base_rho: float = 0.05,
) -> float:
    """Compute adaptive shrinkage parameter ``rho`` based on undersampling ratio.

    When ``num_probes < problem_size`` the shrinkage is increased for stability.
    This mirrors the adaptive strategy described in Section V-A-2.

    Args:
        num_probes: Number of random probe directions ``N_r``.
        problem_size: Problem dimension ``n``.
        gamma: CCCP regularization parameter ``gamma``.
        base_rho: Base shrinkage value when fully sampled.

    Returns:
        Shrinkage parameter in ``[0, 1]``.

    """
    if num_probes >= problem_size:
        return base_rho
    ratio = num_probes / problem_size
    rho = base_rho + (1.0 - base_rho) * (1.0 - ratio) * min(1.0, gamma * 10.0)
    return float(min(rho, 0.5))


def eigh_stable(
    mat: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable symmetric eigendecomposition with eigenvalue clamping.

    Args:
        mat: Symmetric tensor of shape ``(n, n)``.
        eps: Minimum eigenvalue after clamping.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` where eigenvalues are sorted
        in ascending order and clamped to ``[eps, inf)``.

    """
    eigenvalues, eigenvectors = torch.linalg.eigh(mat)
    eigenvalues = eigenvalues.clamp(min=eps)
    return eigenvalues, eigenvectors


class GPSurrogate:
    """Gaussian Process surrogate for Bayesian Optimisation.

    Operates on a **normalised** parameter space where each dimension is
    mapped to ``[0, 1]`` via log-transformation where appropriate.

    Kernel: ``k(x, x') = sigma_f^2 * exp(-0.5 * r^2 / l^2)`` with fixed
    small observation noise ``sigma_n^2``.
    """

    def __init__(
        self,
        bounds: numpy.ndarray,
        log_indices: Optional[list[int]] = None,
        sigma_f: float = 1.0,
        length_scale: float = 0.2,
        sigma_n: float = 1e-4,
    ):
        """Initialise the Gaussian-process surrogate for Bayesian optimisation.

        Args:
        bounds: ``(d, 2)`` array with ``[min, max]`` per dimension.
        log_indices: Dimensions that should be searched on log-scale.
        sigma_f: Signal variance.
        length_scale: Initial length scale (same for all dims).
        sigma_n: Observation noise (fixed, small).

        """
        self.bounds: numpy.ndarray = bounds.astype(numpy.float64)
        self.d = bounds.shape[0]
        self.log_indices = log_indices or []
        self.sigma_f = float(sigma_f)
        self.length_scale = float(length_scale)
        self.sigma_n = float(sigma_n)

        self.X: Optional[numpy.ndarray] = None
        self.y: Optional[numpy.ndarray] = None
        self.K_inv: Optional[numpy.ndarray] = None
        self.alpha_vec: Optional[numpy.ndarray] = None

    def transform(self, x: numpy.ndarray) -> numpy.ndarray:
        """Map raw parameters to normalised [0, 1] space."""
        x = numpy.atleast_2d(x).astype(numpy.float64)
        z = x.copy()
        for i in self.log_indices:
            lb = self.bounds[i, 0] * 0.1
            ub = self.bounds[i, 1] * 10
            lb = max(lb, 1e-12)
            z[:, i] = numpy.log10(numpy.clip(z[:, i], lb, ub))
        z = (z - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        return numpy.clip(z, 0.0, 1.0)

    def kernel(self, x1: numpy.ndarray, x2: numpy.ndarray) -> numpy.ndarray:
        """RBF kernel in normalised space."""
        sqdist = (
            numpy.sum(x1**2, axis=1).reshape(-1, 1)
            + numpy.sum(x2**2, axis=1)
            - 2 * numpy.dot(x1, x2.T)
        )
        return self.sigma_f**2 * numpy.exp(-0.5 * sqdist / (self.length_scale**2 + 1e-12))

    def fit(self, X: numpy.ndarray, y: numpy.ndarray) -> None:
        """Fit GP to observations."""
        self.X = self.transform(X)
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8
        self.y = (y - self.y_mean) / self.y_std

        best_length_scale = self.length_scale
        best_ml = float("-inf")
        for cand_length_scale in numpy.logspace(-2, 0, 20):
            ml = self.marginal_likelihood(cand_length_scale)
            if ml > best_ml:
                best_ml = ml
                best_length_scale = cand_length_scale
        self.length_scale = best_length_scale

        K = self.kernel(self.X, self.X)
        K[numpy.diag_indices_from(K)] += self.sigma_n**2
        self.L = numpy.linalg.cholesky(K + 1e-8 * numpy.eye(K.shape[0]))
        self.alpha_vec = numpy.linalg.solve(self.L.T, numpy.linalg.solve(self.L, self.y))

    def marginal_likelihood(self, candidate_length_scale: float) -> float:
        """Compute log marginal likelihood for a candidate length scale."""
        old_length_scale = self.length_scale
        self.length_scale = candidate_length_scale
        K = self.kernel(self.X, self.X)
        K[numpy.diag_indices_from(K)] += self.sigma_n**2
        try:
            L = numpy.linalg.cholesky(K + 1e-8 * numpy.eye(K.shape[0]))
            alpha = numpy.linalg.solve(L.T, numpy.linalg.solve(L, self.y))
            ml = (
                -0.5 * numpy.dot(self.y, alpha)
                - numpy.sum(numpy.log(numpy.diag(L)))
                - 0.5 * K.shape[0] * numpy.log(2 * math.pi)
            )
        except numpy.linalg.LinAlgError:
            ml = float("-inf")
        self.length_scale = old_length_scale
        return ml

    def predict(self, X_new: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Return posterior mean and variance."""
        X_new_t = self.transform(X_new)
        K_s = self.kernel(self.X, X_new_t)
        K_ss = self.kernel(X_new_t, X_new_t)
        K_ss[numpy.diag_indices_from(K_ss)] += self.sigma_n**2

        v = numpy.linalg.solve(self.L, K_s)
        mu = numpy.dot(K_s.T, self.alpha_vec)
        var = numpy.diag(K_ss) - numpy.sum(v**2, axis=0)
        var = numpy.clip(var, 1e-12, None)

        mu = mu * self.y_std + self.y_mean
        var = var * (self.y_std**2)
        return mu, var

    def expected_improvement(self, X_new: numpy.ndarray, xi: float = 0.01) -> numpy.ndarray:
        """Compute the Expected Improvement acquisition function."""
        mu, var = self.predict(X_new)
        sigma = numpy.sqrt(var)
        y_best = self.y.min() * self.y_std + self.y_mean

        with numpy.errstate(divide="warn", invalid="warn"):
            z = (y_best - mu - xi) / (sigma + 1e-12)
        ei = (y_best - mu - xi) * scipy_norm_cdf(z) + sigma * scipy_norm_pdf(z)
        ei[sigma < 1e-12] = 0.0
        return ei


def scipy_norm_pdf(x: numpy.ndarray) -> numpy.ndarray:
    """Evaluate the standard-normal PDF (no scipy dependency)."""
    return numpy.exp(-0.5 * x**2) / numpy.sqrt(2.0 * math.pi)


def scipy_norm_cdf(x: numpy.ndarray) -> numpy.ndarray:
    """Abramowitz and Stegun approximation of the normal CDF."""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = numpy.sign(x)
    x = numpy.abs(x) / numpy.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * numpy.exp(-x * x))
    return 0.5 * (1.0 + sign * y)
