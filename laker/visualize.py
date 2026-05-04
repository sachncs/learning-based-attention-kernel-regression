"""Visualisation utilities for radio maps and convergence diagnostics."""

import logging
from typing import Optional, Tuple

import numpy
import torch

logger = logging.getLogger(__name__)


class Visualizer:
    """High-level visualiser for LAKER regression outputs and diagnostics."""

    def __init__(self, figsize: Tuple[int, int] = (6, 5)):
        """Initialise the visualiser."""
        self.figsize = figsize

    def radio_map_to_image(
        self,
        predictions: torch.Tensor,
        grid_size: int,
        extent: Optional[Tuple[float, float, float, float]] = None,
    ) -> numpy.ndarray:
        """Convert flat predictions on a regular grid to a 2-D image array.

        Args:
            predictions: Flat tensor of shape ``(grid_size**2,)``.
            grid_size: Number of points along each spatial axis.
            extent: ``(x_min, x_max, y_min, y_max)`` for axis labels.

        Returns:
            2-D NumPy array of shape ``(grid_size, grid_size)``.

        """
        img = predictions.detach().cpu().numpy().reshape(grid_size, grid_size)
        return img

    def plot_radio_map(
        self,
        predictions: torch.Tensor,
        grid_size: int,
        title: str = "Radio Map Reconstruction",
        extent: Optional[Tuple[float, float, float, float]] = None,
        colorbar_label: str = "RSS (dBm)",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> tuple:
        """Plot a 2-D radio map using Matplotlib.

        Args:
            predictions: Flat tensor of shape ``(grid_size**2,)``.
            grid_size: Number of points along each spatial axis.
            title: Plot title.
            extent: ``(x_min, x_max, y_min, y_max)``.
            colorbar_label: Label for the colorbar.
            vmin: Minimum value for the colormap.
            vmax: Maximum value for the colormap.

        Returns:
            Matplotlib figure and axes objects.

        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Matplotlib is required for plotting. Install it with: pip install matplotlib"
            ) from exc

        img = self.radio_map_to_image(predictions, grid_size, extent)
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(
            img,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)
        fig.tight_layout()
        logger.info("Plotted radio map: grid_size=%d", grid_size)
        return fig, ax

    def plot_convergence(
        self,
        objective_gaps: list,
        labels: Optional[list] = None,
        title: str = "Convergence Behaviour",
        xlabel: str = "Iteration",
        ylabel: str = "Relative Objective Gap",
        figsize: Optional[Tuple[int, int]] = None,
    ) -> tuple:
        """Plot convergence curves for one or more solvers.

        Args:
            objective_gaps: List of lists, where each inner list contains the
                objective gap at each iteration for a single solver.
            labels: Optional list of labels for each curve.
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            figsize: Figure size in inches. Defaults to ``self.figsize``.

        Returns:
            Matplotlib figure and axes objects.

        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Matplotlib is required for plotting. Install it with: pip install matplotlib"
            ) from exc

        figsize = figsize if figsize is not None else self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        for idx, gaps in enumerate(objective_gaps):
            label = labels[idx] if labels and idx < len(labels) else f"Solver {idx + 1}"
            ax.semilogy(gaps, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        fig.tight_layout()
        logger.info("Plotted convergence curves: %d series", len(objective_gaps))
        return fig, ax


def radio_map_to_image(
    predictions: torch.Tensor,
    grid_size: int,
    extent: Optional[Tuple[float, float, float, float]] = None,
) -> numpy.ndarray:
    """Convert flat predictions on a regular grid to a 2-D image array.

    Convenience wrapper around ``Visualizer.radio_map_to_image()``.

    Args:
        predictions: Flat tensor of shape ``(grid_size**2,)``.
        grid_size: Number of points along each spatial axis.
        extent: ``(x_min, x_max, y_min, y_max)`` for axis labels.

    Returns:
        2-D NumPy array of shape ``(grid_size, grid_size)``.

    """
    img = predictions.detach().cpu().numpy().reshape(grid_size, grid_size)
    return img


def plot_radio_map(
    predictions: torch.Tensor,
    grid_size: int,
    title: str = "Radio Map Reconstruction",
    extent: Optional[Tuple[float, float, float, float]] = None,
    colorbar_label: str = "RSS (dBm)",
    figsize: Tuple[int, int] = (6, 5),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> tuple:
    """Plot a 2-D radio map using Matplotlib.

    Convenience wrapper around ``Visualizer.plot_radio_map()``.

    Args:
        predictions: Flat tensor of shape ``(grid_size**2,)``.
        grid_size: Number of points along each spatial axis.
        title: Plot title.
        extent: ``(x_min, x_max, y_min, y_max)``.
        colorbar_label: Label for the colorbar.
        figsize: Figure size in inches.
        vmin: Minimum value for the colormap.
        vmax: Maximum value for the colormap.

    Returns:
        Matplotlib figure and axes objects.

    """
    visualizer = Visualizer(figsize=figsize)
    return visualizer.plot_radio_map(
        predictions, grid_size, title, extent, colorbar_label, vmin, vmax
    )


def plot_convergence(
    objective_gaps: list,
    labels: Optional[list] = None,
    title: str = "Convergence Behaviour",
    xlabel: str = "Iteration",
    ylabel: str = "Relative Objective Gap",
    figsize: Tuple[int, int] = (6, 4),
) -> tuple:
    """Plot convergence curves for one or more solvers.

    Convenience wrapper around ``Visualizer.plot_convergence()``.

    Args:
        objective_gaps: List of lists, where each inner list contains the
            objective gap at each iteration for a single solver.
        labels: Optional list of labels for each curve.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size in inches.

    Returns:
        Matplotlib figure and axes objects.

    """
    visualizer = Visualizer(figsize=figsize)
    return visualizer.plot_convergence(objective_gaps, labels, title, xlabel, ylabel, figsize)
