"""Tests for laker.visualize module."""

import numpy
import pytest
import torch

from laker.visualize import (
    Visualizer,
    plot_convergence,
    plot_radio_map,
    radio_map_to_image,
)


def test_radio_map_to_image():
    """radio_map_to_image should reshape flat predictions correctly."""
    grid_size = 5
    preds = torch.arange(grid_size * grid_size, dtype=torch.float32)
    img = radio_map_to_image(preds, grid_size)
    assert img.shape == (grid_size, grid_size)
    assert numpy.allclose(img[0, 0], 0.0)
    assert numpy.allclose(img[4, 4], 24.0)


def test_visualizer_radio_map_to_image_with_extent():
    """Visualizer.radio_map_to_image should accept extent parameter."""
    grid_size = 4
    preds = torch.randn(grid_size * grid_size)
    img = Visualizer().radio_map_to_image(preds, grid_size, extent=(0, 100, 0, 100))
    assert img.shape == (grid_size, grid_size)


def test_plot_radio_map():
    """plot_radio_map should return figure and axes."""
    pytest.importorskip("matplotlib")
    grid_size = 4
    preds = torch.randn(grid_size * grid_size)
    fig, ax = plot_radio_map(preds, grid_size, title="Test Map")
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Test Map"


def test_visualizer_plot_radio_map_with_bounds():
    """Visualizer.plot_radio_map should accept vmin and vmax."""
    pytest.importorskip("matplotlib")
    grid_size = 4
    preds = torch.randn(grid_size * grid_size)
    fig, ax = Visualizer(figsize=(4, 4)).plot_radio_map(preds, grid_size, vmin=-2.0, vmax=2.0)
    assert fig is not None
    assert ax is not None


def test_plot_convergence():
    """plot_convergence should return figure and axes."""
    pytest.importorskip("matplotlib")
    gaps = [[1.0, 0.1, 0.01], [1.0, 0.5, 0.25]]
    labels = ["Solver A", "Solver B"]
    fig, ax = plot_convergence(gaps, labels=labels)
    assert fig is not None
    assert ax is not None


def test_visualizer_plot_convergence_defaults():
    """Visualizer.plot_convergence should work without labels."""
    pytest.importorskip("matplotlib")
    gaps = [[1.0, 0.1, 0.01]]
    fig, ax = Visualizer().plot_convergence(gaps)
    assert fig is not None
    assert ax is not None


def test_plot_radio_map_no_matplotlib():
    """plot_radio_map should raise ImportError when matplotlib is missing."""
    real_import = __builtins__["__import__"]

    def mock_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("builtins.__import__", mock_import)
        with pytest.raises(ImportError, match="Matplotlib is required"):
            Visualizer().plot_radio_map(torch.randn(4), 2)


def test_plot_convergence_no_matplotlib():
    """plot_convergence should raise ImportError when matplotlib is missing."""
    real_import = __builtins__["__import__"]

    def mock_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("builtins.__import__", mock_import)
        with pytest.raises(ImportError, match="Matplotlib is required"):
            Visualizer().plot_convergence([[1.0, 0.1]])
