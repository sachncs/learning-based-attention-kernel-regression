"""Synthetic data generation for spectrum cartography experiments."""

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class RadioFieldGenerator:
    """Generator for synthetic radio propagation fields.

    Produces received signal strength (RSS) maps from multiple transmitters
    using log-distance path loss with optional log-normal shadowing.
    """

    def __init__(
        self,
        path_loss_exponent: float = 2.0,
        reference_distance: float = 1.0,
        shadow_sigma: float = 1.5,
    ):
        """Initialise the radio-map generator."""
        self.path_loss_exponent = float(path_loss_exponent)
        self.reference_distance = float(reference_distance)
        self.shadow_sigma = float(shadow_sigma)

    def generate(
        self,
        locations: torch.Tensor,
        transmitters: torch.Tensor,
        powers: torch.Tensor,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Generate a synthetic radio field from multiple transmitters.

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
            seed: Optional random seed for reproducible noise.

        Returns:
            Tuple ``(rss_clean, rss_noisy)`` where each is a tensor of shape ``(n,)``.

        Raises:
            ValueError: If input shapes are inconsistent.

        """
        if locations.dim() != 2:
            raise ValueError(f"locations must be 2-D, got shape {locations.shape}")
        if transmitters.dim() != 2:
            raise ValueError(f"transmitters must be 2-D, got shape {transmitters.shape}")
        if powers.dim() != 1:
            raise ValueError(f"powers must be 1-D, got shape {powers.shape}")
        if transmitters.shape[0] != powers.shape[0]:
            raise ValueError(
                "transmitters and powers must have same length, "
                f"got {transmitters.shape[0]} and {powers.shape[0]}"
            )
        if locations.shape[1] != transmitters.shape[1]:
            raise ValueError(
                "locations and transmitters must have same spatial dimension, "
                f"got {locations.shape[1]} and {transmitters.shape[1]}"
            )

        if seed is not None:
            gen = torch.Generator(device=locations.device)
            gen.manual_seed(seed)
        else:
            gen = None

        n = locations.shape[0]
        rss_clean = torch.zeros(n, device=locations.device, dtype=locations.dtype)

        for transmitter_location, transmitter_power in zip(transmitters, powers):
            distances = torch.norm(locations - transmitter_location, dim=1)
            distances = distances.clamp(min=self.reference_distance)
            path_loss = (
                10.0 * self.path_loss_exponent * torch.log10(distances / self.reference_distance)
            )
            rss_clean += transmitter_power - path_loss

        noise = torch.randn(n, device=locations.device, dtype=locations.dtype, generator=gen)
        rss_noisy = rss_clean + self.shadow_sigma * noise
        logger.info(
            "Generated radio field: n=%d, tx=%d, path_loss_exp=%.1f, shadow_sigma=%.2f",
            n,
            transmitters.shape[0],
            self.path_loss_exponent,
            self.shadow_sigma,
        )
        return rss_clean, rss_noisy


def generate_radio_field(
    locations: torch.Tensor,
    transmitters: torch.Tensor,
    powers: torch.Tensor,
    path_loss_exponent: float = 2.0,
    reference_distance: float = 1.0,
    shadow_sigma: float = 1.5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Generate a synthetic radio field from multiple transmitters.

    Convenience wrapper around ``RadioFieldGenerator.generate()``.

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

    """
    generator = RadioFieldGenerator(
        path_loss_exponent=path_loss_exponent,
        reference_distance=reference_distance,
        shadow_sigma=shadow_sigma,
    )
    return generator.generate(locations, transmitters, powers, seed=seed)


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
