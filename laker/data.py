"""Synthetic data generation for spectrum cartography experiments.

The :mod:`laker.data` module exposes two complementary generators used
throughout the LAKER examples, tests, and benchmarks:

* :func:`generate_radio_field` (and its underlying
  :class:`RadioFieldGenerator`) synthesises a *received signal strength*
  (RSS) map from a small set of point transmitters using the standard
  log-distance path loss model with optional log-normal shadowing. The
  resulting field is the canonical benchmark for spectrum cartography
  algorithms in the LAKER paper (Tao & Tan, 2026).
* :func:`generate_grid` produces a regular 2-D evaluation grid for
  visualising reconstructed RSS maps.

Both generators are deterministic when ``seed`` is provided, so they
are suitable for reproducible benchmarks and unit tests. The path-loss
model assumes a free-space-like exponent (``eta=2``) by default, but
arbitrary exponents are supported.
"""

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class RadioFieldGenerator:
    """Generator for synthetic radio propagation fields.

    Produces received signal strength (RSS) maps from multiple
    transmitters using log-distance path loss with optional log-normal
    shadowing. The model is the standard one used in the LAKER paper:

    .. math::

        r(x) = \\sum_j \\bigl(P_j - 10 \\eta \\log_{10}(d_j / d_0)\\bigr)
        + \\sigma_\\epsilon \\epsilon

    where :math:`d_j = \\|x - \\text{tx}_j\\|_2`,
    :math:`\\epsilon \\sim \\mathcal{N}(0, 1)`, and
    :math:`(\\eta, d_0, \\sigma_\\epsilon)` are constructor arguments.

    The class is intentionally stateless apart from its three
    hyperparameters; the per-call state is just the inputs and the
    optional RNG seed, so a single generator instance can be reused
    across many calls without interference.

    Args:
        path_loss_exponent: Path-loss exponent :math:`\\eta` (default
            ``2.0`` corresponding to free-space propagation).
        reference_distance: Reference distance :math:`d_0` (metres)
            below which the log-distance term is clamped to avoid
            ``log(0)``. Default ``1.0``.
        shadow_sigma: Standard deviation :math:`\\sigma_\\epsilon` of
            the log-normal shadowing term (dB). Set to ``0`` for a
            noise-free field.

    """

    def __init__(
        self,
        path_loss_exponent: float = 2.0,
        reference_distance: float = 1.0,
        shadow_sigma: float = 1.5,
    ):
        """Initialise the radio-map generator.

        Stores the three hyperparameters; no RNG state is created here
        (a fresh per-call generator is built inside :meth:`generate`
        only when ``seed`` is provided).

        Args:
            path_loss_exponent: Path-loss exponent ``eta``.
            reference_distance: Reference distance ``d_0``.
            shadow_sigma: Shadowing standard deviation.

        """
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
        superposition of log-distance path-loss terms plus log-normal
        shadowing:

        .. math::

            r(x) = \sum_j \left( P_j - 10 \eta \log_{10}
            \frac{d_j}{d_0} \right) + \sigma_\epsilon \epsilon

        where :math:`d_j = \|x - \text{tx}_j\|_2` and
        :math:`\epsilon \sim \mathcal{N}(0, 1)`.

        Args:
            locations: Sensor locations of shape ``(n, dx)``.
            transmitters: Transmitter coordinates of shape
                ``(num_tx, dx)``.
            powers: Transmitter power levels in dBm of shape
                ``(num_tx,)``.
            seed: Optional random seed for reproducible shadowing. If
                ``None`` the global RNG is used.

        Returns:
            Tuple ``(rss_clean, rss_noisy)`` where each is a tensor of
            shape ``(n,)``. ``rss_clean`` is the noise-free path-loss
            field; ``rss_noisy`` adds shadowing.

        Raises:
            ValueError: If ``locations``, ``transmitters``, or ``powers``
                have the wrong dimensionality, if the number of
                transmitters does not match the number of powers, or if
                the spatial dimension of locations and transmitters
                disagree.

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
            # Per-call generator keeps seed independence across calls —
            # the global RNG is untouched, so concurrent calls do not
            # race on shared state.
            gen = torch.Generator(device=locations.device)
            gen.manual_seed(seed)
        else:
            gen = None

        n = locations.shape[0]
        rss_clean = torch.zeros(n, device=locations.device, dtype=locations.dtype)

        # Iterating over transmitters is intentional: ``num_tx`` is
        # typically ≤ 10 (handful of emitters), so the Python-level loop
        # cost is negligible compared to the subsequent PCG solve.
        # Vectorising this loop would require a broadcasted
        # ``(n, num_tx)`` distance matrix that we then keep in memory
        # for no good reason.
        for transmitter_location, transmitter_power in zip(transmitters, powers):
            distances = torch.norm(locations - transmitter_location, dim=1)
            # Clamp to ``reference_distance`` so the log-distance term
            # never sees a non-positive argument (``log10(0) = -inf``).
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

    Convenience wrapper around :meth:`RadioFieldGenerator.generate` for
    callers that prefer a single function call. The full path-loss
    model is described in the :class:`RadioFieldGenerator` docstring.

    Args:
        locations: Sensor locations of shape ``(n, dx)``.
        transmitters: Transmitter coordinates of shape ``(num_tx, dx)``.
        powers: Transmitter power levels in dBm of shape ``(num_tx,)``.
        path_loss_exponent: Path-loss exponent ``eta`` (default ``2.0``
            for free space).
        reference_distance: Reference distance ``d_0`` in metres.
        shadow_sigma: Standard deviation of log-normal shadowing in dB.
        seed: Optional random seed for reproducible shadowing.

    Returns:
        Tuple ``(rss_clean, rss_noisy)`` where each is a tensor of
        shape ``(n,)``.

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

    Builds a ``grid_size x grid_size`` lattice of points in the bounding
    box ``[x_min, x_max] x [y_min, y_max]`` and returns the points in
    row-major order. The output shape is ``(grid_size ** 2, 2)``.

    Args:
        bounds: Tuple ``(x_min, x_max, y_min, y_max)``. ``x_min`` must
            be strictly less than ``x_max``; the same applies to the
            ``y`` bounds.
        grid_size: Number of points along each axis. Must be at least
            ``2``.
        device: Target :class:`torch.device` (defaults to
            :func:`laker.backend.get_default_device`).
        dtype: Target :class:`torch.dtype` (defaults to
            :func:`laker.backend.get_default_dtype`).

    Returns:
        Tensor of shape ``(grid_size**2, 2)`` whose rows are the
        ``(x, y)`` coordinates of every grid point.

    """
    x_min, x_max, y_min, y_max = bounds
    x = torch.linspace(x_min, x_max, grid_size, device=device, dtype=dtype)
    y = torch.linspace(y_min, y_max, grid_size, device=device, dtype=dtype)
    # ``indexing="ij"`` produces matrix-style indices so the resulting
    # ``(grid_size, grid_size)`` tensor corresponds to row-major
    # traversal of the lattice.
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xx.ravel(), yy.ravel()], dim=1)