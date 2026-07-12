"""Numerical-stability helpers and the GP surrogate for Bayesian Optimisation.

The :mod:`laker.utils` module collects small but critical utilities used
across the package:

* :func:`trace_normalize` — enforces a unit-mean eigenvalue on a
  positive-definite matrix (LAKER paper, Eqs. 34 and 37). Used by the
  CCCP preconditioner to keep the learned covariance on a consistent
  scale.
* :func:`adaptive_shrinkage_rho` — implements the adaptive shrinkage
  schedule from Section V-A-2 of the LAKER paper. When the number of
  random probes ``N_r`` is much smaller than the problem size ``n`` the
  shrinkage is increased to keep the CCCP iteration well-conditioned.
* :func:`eigh_stable` — a wrapper around :func:`torch.linalg.eigh` that
  clamps the resulting eigenvalues to ``[eps, inf)`` so downstream
  divisions by ``sqrt(lambda)`` cannot blow up.
* :class:`GPSurrogate` — a small Gaussian-process surrogate used by
  :mod:`laker.search` for Bayesian hyperparameter optimisation. The
  class operates on a normalised parameter space where each dimension
  is mapped to ``[0, 1]`` via an optional log-transform. The kernel is
  the standard RBF ``k(x, x') = sigma_f^2 exp(-0.5 r^2 / l^2)`` with a
  fixed small observation noise ``sigma_n^2``.
* :func:`scipy_norm_pdf`, :func:`scipy_norm_cdf` — scipy-free pure
  NumPy/Python implementations of the standard normal probability
  density function and cumulative distribution function. The CDF
  approximation uses formula 26.2.17 from Abramowitz & Stegun (1964)
  with a maximum absolute error of ``7.5e-8``. These exist so that
  :class:`GPSurrogate` and the expected-improvement acquisition
  function have **no scipy dependency**, enabling deployment in
  minimal Python environments.

The module does not depend on the rest of LAKER; everything here is
self-contained and safe to import in isolation.
"""

import math
from typing import Optional

import numpy
import torch


def trace_normalize(mat: torch.Tensor) -> torch.Tensor:
    """Normalize a positive-definite matrix so that ``trace(mat) == n``.

    This corresponds to Eq. (34) and (37) in the LAKER paper. The
    operation divides the matrix by its mean eigenvalue, which is
    equivalent to scaling the eigenvalues uniformly.

    Args:
        mat: Square tensor of shape ``(n, n)``. Should be
            positive-definite (or at least positive-semidefinite) for
            the result to be meaningful.

    Returns:
        Normalized matrix with unit mean eigenvalue (``trace(out) == n``).

    Raises:
        RuntimeError: If the trace is exactly zero (propagated from
            PyTorch's division).

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

    When ``num_probes < problem_size`` the shrinkage is increased for
    stability. This mirrors the adaptive strategy described in Section
    V-A-2 of the LAKER paper.

    The function blends two contributions:

    1. A linear interpolation ``base_rho + (1 - base_rho)(1 - ratio)``
       that increases shrinkage as the undersampling ratio ``ratio =
       num_probes / problem_size`` decreases.
    2. A ``gamma``-driven scaling ``min(1, gamma * 10)`` that further
       inflates shrinkage when the CCCP regularisation is large.

    The result is then clamped to a maximum of ``0.5`` to prevent the
    preconditioner from collapsing into a pure diagonal.

    Args:
        num_probes: Number of random probe directions ``N_r``.
        problem_size: Problem dimension ``n``.
        gamma: CCCP regularization parameter ``gamma``.
        base_rho: Base shrinkage value when fully sampled
            (``num_probes >= problem_size``).

    Returns:
        Shrinkage parameter in ``[0, 1]`` and at most ``0.5``.

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

    Performs a Hermitian eigendecomposition via :func:`torch.linalg.eigh`
    and then clamps the resulting eigenvalues to ``[eps, inf)`. The
    clamping step protects downstream consumers from division by
    numerical zeros that arise from ill-conditioned matrices.

    Args:
        mat: Symmetric (or Hermitian) tensor of shape ``(n, n)``. The
            matrix is *not* checked for symmetry — callers should
            pre-symmetrise with ``0.5 * (M + M.T)`` if necessary.
        eps: Minimum eigenvalue after clamping. Default ``1e-10``.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)``. Eigenvalues are sorted
        in ascending order and clamped to ``[eps, inf)``. Eigenvectors
        are arranged in the corresponding columns.

    """
    eigenvalues, eigenvectors = torch.linalg.eigh(mat)
    eigenvalues = eigenvalues.clamp(min=eps)
    return eigenvalues, eigenvectors


class GPSurrogate:
    """Gaussian Process surrogate for Bayesian Optimisation.

    Operates on a **normalised** parameter space where each dimension is
    mapped to ``[0, 1]`` via an optional log-transformation
    (:meth:`transform`). Inputs in log-indexed dimensions are clipped
    to ``[bounds[i, 0] * 0.1, bounds[i, 1] * 10]`` before the ``log10``
    transform so out-of-range queries do not produce ``NaN``.

    Kernel: ``k(x, x') = sigma_f^2 * exp(-0.5 * r^2 / l^2)`` with a
    fixed small observation noise ``sigma_n^2``.

    Length-scale selection: :meth:`fit` performs a coarse log-space grid
    search over ``l in [10^-2, 1]`` (20 candidate values) and selects
    the length scale that maximises the log marginal likelihood. The
    grid is deliberately coarse because the surrogate is only used as a
    cheap acquisition surface for the BO outer loop; the inner PCG
    solve dominates the cost anyway.

    Args:
        bounds: ``(d, 2)`` array with ``[min, max]`` per dimension.
        log_indices: Dimensions that should be searched on log-scale.
            Indices are 0-based into ``bounds``. Default ``None``
            (linear scale on every dimension).
        sigma_f: Signal variance. Default ``1.0``.
        length_scale: Initial length scale (used as the starting point
            of the grid search). Default ``0.2``.
        sigma_n: Observation noise (fixed, small). Default ``1e-4``.

    Attributes:
        X: Training inputs in normalised space (``None`` until
            :meth:`fit` is called).
        y: Standardised training targets (``None`` until :meth:`fit`).
        K_inv: Inverse of the noisy training kernel (``None`` until
            :meth:`fit`).
        alpha_vec: Cached ``K^{-1} y`` for fast mean predictions
            (``None`` until :meth:`fit`).

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
            log_indices: Dimensions that should be searched on
                log-scale. Indices are 0-based into ``bounds``. Input
                values in these dimensions are clipped to
                ``[bounds[i, 0] * 0.1, bounds[i, 1] * 10]`` before a
                ``log10`` transform is applied. Default ``None`` (linear
                scale on every dimension).
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

        # Cached Cholesky factor and ``K^{-1} y`` populated by ``fit``.
        self.X: Optional[numpy.ndarray] = None
        self.y: Optional[numpy.ndarray] = None
        self.K_inv: Optional[numpy.ndarray] = None
        self.alpha_vec: Optional[numpy.ndarray] = None

    def transform(self, x: numpy.ndarray) -> numpy.ndarray:
        """Map raw parameters to the normalised ``[0, 1]`` space.

        For dimensions in ``self.log_indices`` the transformation is:

        1. Clip to ``[bounds[i, 0] * 0.1, bounds[i, 1] * 10]`` (a small
           fudge factor so the bounds aren't *hard* during BO
           exploration).
        2. Apply ``log10``.
        3. Apply the standard min-max normalisation.

        Linear dimensions skip the log step. The result is clipped to
        ``[0, 1]`` so subsequent kernel evaluations never produce
        negative-squared-distances out-of-range.

        Args:
            x: 1-D or 2-D array of raw parameters.

        Returns:
            Array of the same shape as ``x`` (after ``atleast_2d``),
            mapped to ``[0, 1]^d``.

        """
        x = numpy.atleast_2d(x).astype(numpy.float64)
        z = x.copy()
        for i in self.log_indices:
            # The ``* 0.1`` / ``* 10`` factors widen the bounds so BO
            # can briefly explore outside the user-provided range; the
            # final ``max(lb, 1e-12)`` guard prevents ``log10(0)``.
            lb = self.bounds[i, 0] * 0.1
            ub = float(self.bounds[i, 1]) * 10
            lb = max(lb, 1e-12)
            z[:, i] = numpy.log10(numpy.clip(z[:, i], lb, ub))
        z = (z - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        return numpy.clip(z, 0.0, 1.0)

    def kernel(self, x1: numpy.ndarray, x2: numpy.ndarray) -> numpy.ndarray:
        """RBF kernel in normalised space.

        Args:
            x1: Array of shape ``(n1, d)``.
            x2: Array of shape ``(n2, d)``.

        Returns:
            Kernel matrix of shape ``(n1, n2)`` with entries
            ``sigma_f^2 * exp(-0.5 * ||x1_i - x2_j||^2 / l^2)``.
            A ``+ 1e-12`` epsilon on the length scale protects against
            division by zero.

        """
        # ``||x1||^2 + ||x2||^2 - 2 x1 . x2`` evaluated without
        # materialising any pair-wise distance matrix is significantly
        # faster than calling ``scipy.spatial.distance.cdist``.
        sqdist = (
            numpy.sum(x1**2, axis=1).reshape(-1, 1)
            + numpy.sum(x2**2, axis=1)
            - 2 * numpy.dot(x1, x2.T)
        )
        return self.sigma_f**2 * numpy.exp(-0.5 * sqdist / (self.length_scale**2 + 1e-12))

    def fit(self, X: numpy.ndarray, y: numpy.ndarray) -> None:
        """Fit the GP surrogate to observations.

        Steps:

        1. Standardise ``y`` to zero mean / unit variance so the
           observation-noise prior ``sigma_n`` is scale-invariant.
        2. Sweep a coarse log-space grid of length scales (20 values
           between ``10^{-2}`` and ``1``), select the one that
           maximises the log marginal likelihood, and store it.
        3. Cache the Cholesky factor of the noisy kernel matrix and
           ``K^{-1} y`` for fast posterior computations.

        Args:
            X: Training inputs of shape ``(n, d)`` in raw parameter
                space (will be normalised internally).
            y: Training targets of shape ``(n,)``.

        Side effects:
            Stores ``self.X``, ``self.y_mean``, ``self.y_std``,
            ``self.y``, ``self.length_scale``, ``self.L``, and
            ``self.alpha_vec``.

        """
        self.X = self.transform(X)
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8
        self.y = (y - self.y_mean) / self.y_std

        # Coarse log-space grid: ``logspace(-2, 0, 20)`` gives 20
        # candidates spanning two orders of magnitude. The grid is
        # deliberately narrow because the GP is only a cheap
        # acquisition surface — the inner PCG solve dominates cost.
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
        """Compute the log marginal likelihood for a candidate length scale.

        Temporarily overwrites ``self.length_scale`` so :meth:`kernel`
        uses the candidate value, then restores the original length
        scale before returning. Failures from a non-positive-definite
        kernel (e.g. ``length_scale`` too small) are caught and the
        method returns ``-inf`` so the outer grid search ignores that
        candidate.

        Args:
            candidate_length_scale: Length scale to evaluate.

        Returns:
            Log marginal likelihood ``log p(y | X, l)`` under the RBF
            kernel. Lower (more negative) is worse; ``-inf`` indicates
            that the candidate produced a non-PD kernel.

        """
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
            # Non-PD kernel for this length scale — treat as ``-inf``
            # so the grid search skips it.
            ml = float("-inf")
        self.length_scale = old_length_scale
        return ml

    def predict(self, X_new: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Return posterior mean and variance at query points.

        Args:
            X_new: Query inputs of shape ``(n, d)`` in raw parameter
                space (will be normalised internally).

        Returns:
            Tuple ``(mu, var)`` of posterior mean and variance, each
            of shape ``(n,)`` and de-standardised back to the original
            ``y`` scale.

        """
        X_new_t = self.transform(X_new)
        K_s = self.kernel(self.X, X_new_t)
        K_ss = self.kernel(X_new_t, X_new_t)
        K_ss[numpy.diag_indices_from(K_ss)] += self.sigma_n**2

        v = numpy.linalg.solve(self.L, K_s)
        mu = numpy.dot(K_s.T, self.alpha_vec)
        var = numpy.diag(K_ss) - numpy.sum(v**2, axis=0)
        var = numpy.clip(var, 1e-12, None)

        # De-standardise so callers see predictions on the original
        # ``y`` scale.
        mu = mu * self.y_std + self.y_mean
        var = var * (self.y_std**2)
        return mu, var

    def expected_improvement(self, X_new: numpy.ndarray, xi: float = 0.01) -> numpy.ndarray:
        """Compute the Expected Improvement acquisition function.

        Uses the standard EI formula ``(y_best - mu - xi) Phi(z) +
        sigma phi(z)`` where ``z = (y_best - mu - xi) / sigma``,
        ``Phi`` is the standard-normal CDF, and ``phi`` the PDF.
        Implemented in pure NumPy via :func:`scipy_norm_cdf` /
        :func:`scipy_norm_pdf` so there is no scipy dependency.

        Args:
            X_new: Query points of shape ``(n, d)``.
            xi: Exploration-exploitation trade-off parameter
                (``0`` = pure exploitation). Default ``0.01``.

        Returns:
            EI values of shape ``(n,)``. Zero where ``sigma`` is
            effectively zero (the posterior is deterministic).

        """
        mu, var = self.predict(X_new)
        sigma = numpy.sqrt(var)
        y_best = self.y.min() * self.y_std + self.y_mean

        # ``numpy.errstate`` suppresses divide-by-zero warnings for the
        # EI formula — the resulting ``NaN`` for ``sigma == 0`` is then
        # masked to ``0.0`` at the end.
        with numpy.errstate(divide="warn", invalid="warn"):
            z = (y_best - mu - xi) / (sigma + 1e-12)
        ei = (y_best - mu - xi) * scipy_norm_cdf(z) + sigma * scipy_norm_pdf(z)
        ei[sigma < 1e-12] = 0.0
        return ei


def scipy_norm_pdf(x: numpy.ndarray) -> numpy.ndarray:
    """Evaluate the standard-normal PDF (no scipy dependency).

    Args:
        x: Array of arbitrary shape.

    Returns:
        Array of the same shape as ``x`` with entries
        ``exp(-0.5 x^2) / sqrt(2 pi)``.

    """
    return numpy.exp(-0.5 * x**2) / numpy.sqrt(2.0 * math.pi)


def scipy_norm_cdf(x: numpy.ndarray) -> numpy.ndarray:
    """Abramowitz and Stegun approximation of the normal CDF.

    Implements formula ``26.2.17`` from Abramowitz & Stegun (1964),
    Handbook of Mathematical Functions, which gives a maximum absolute
    error of ``7.5e-8`` over all of ``R``. The implementation has
    **no scipy dependency** so it can be used in environments where
    scipy is unavailable.

    Args:
        x: Array of arbitrary shape.

    Returns:
        Array of the same shape as ``x`` with entries approximating
        ``Phi(x)``, the standard-normal cumulative distribution
        function evaluated at ``x``.

    """
    # Coefficients of the rational approximation (A&S 26.2.17).
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