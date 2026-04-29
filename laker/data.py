"""Synthetic data generation for spectrum cartography experiments."""

from typing import Optional, Tuple

import torch


def generate_radio_field(
    locations: torch.Tensor,
    transmitters: torch.Tensor,
    powers: torch.Tensor,
    path_loss_exponent: float = 2.0,
    reference_distance: float = 1.0,
    shadow_sigma: float = 1.5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic radio field from multiple transmitters.

    The received signal strength at each location is computed as a
    superposition of log-distance path loss terms plus log-normal
    shadowing:

    .. math::
        r(x) = \\sum_j \\left( P_j - 10 \\eta \\log_{10} \\frac{d_j}{d_0} \\right) + \\epsilon

    where :math:`d_j = \\|x - \\text{tx}_j\\|_2` and
    :math:`\\epsilon \\sim \\mathcal{N}(0, \\sigma_\\epsilon^2)`.

    Args:
        locations: Sensor locations of shape ``(n, dx)``.
        transmitters: Transmitter coordinates of shape ``(num_tx, dx)``.
        powers: Transmitter power levels in dBm of shape ``(num_tx,)``.
        path_loss_exponent: Path-loss exponent ``eta`` (default 2.0 for free space).
        reference_distance: Reference distance ``d_0`` in metres.
        shadow_sigma: Standard deviation of log-normal shadowing in dB.
        seed: Optional random seed for reproducible noise.

    Returns:
        Tuple ``(rss_clean, rss_noisy)`` where each is a tensor of shape ``(n,)``.

    Example:
        >>> locs = torch.rand(100, 2) * 100.0
        >>> tx = torch.tensor([[30.0, 70.0], [70.0, 30.0]])
        >>> pwr = torch.tensor([-40.0, -45.0])
        >>> clean, noisy = generate_radio_field(locs, tx, pwr)
    """
    if seed is not None:
        gen = torch.Generator(device=locations.device)
        gen.manual_seed(seed)
    else:
        gen = None

    n = locations.shape[0]
    rss_clean = torch.zeros(n, device=locations.device, dtype=locations.dtype)

    for tx_loc, tx_pwr in zip(transmitters, powers):
        distances = torch.norm(locations - tx_loc, dim=1)
        distances = distances.clamp(min=reference_distance)
        path_loss = 10.0 * path_loss_exponent * torch.log10(distances / reference_distance)
        rss_clean += tx_pwr - path_loss

    noise = torch.randn(n, device=locations.device, dtype=locations.dtype, generator=gen)
    rss_noisy = rss_clean + shadow_sigma * noise
    return rss_clean, rss_noisy


def generate_grid(
    bounds: Tuple[float, float, float, float],
    grid_size: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Generate a regular 2-D evaluation grid.

    Args:
        bounds: ``(x_min, x_max, y_min, y_max)``.
        grid_size: Number of points along each axis.
        device: torch device.
        dtype: torch dtype.

    Returns:
        Tensor of shape ``(grid_size**2, 2)`` with grid coordinates.
    """
    x_min, x_max, y_min, y_max = bounds
    x = torch.linspace(x_min, x_max, grid_size, device=device, dtype=dtype)
    y = torch.linspace(y_min, y_max, grid_size, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xx.ravel(), yy.ravel()], dim=1)
