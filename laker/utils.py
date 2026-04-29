"""Utility functions for numerical stability and convergence."""

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
