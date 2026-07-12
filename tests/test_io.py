"""Tests for model serialization and I/O."""

import os
import tempfile

import torch

from laker.models import LAKERRegressor


def test_save_and_load():
    """Test that save and load roundtrip preserves model state and predictions."""
    n = 40
    x = torch.rand(n, 2) * 100.0
    y = torch.randn(n)

    model = LAKERRegressor(
        embedding_dim=4,
        num_probes=20,
        cccp_max_iter=10,
        verbose=False,
    )
    model.fit(x, y)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        model.save(path)
        loaded = LAKERRegressor.load(path)

        assert loaded.alpha is not None
        assert loaded.embeddings is not None
        assert torch.allclose(loaded.alpha, model.alpha)
        assert torch.allclose(loaded.embeddings, model.embeddings)

        # Predictions should match
        x_test = torch.rand(10, 2) * 100.0
        pred_orig = model.predict(x_test)
        pred_load = loaded.predict(x_test)
        torch.testing.assert_close(pred_orig, pred_load, rtol=1e-5, atol=1e-5)
    finally:
        os.unlink(path)


def test_save_and_load_with_corrector():
    """Test that save and load preserves residual corrector and predictions."""
    n = 40
    x = torch.rand(n, 2) * 100.0
    y = torch.randn(n)

    model = LAKERRegressor(
        embedding_dim=4,
        num_probes=20,
        cccp_max_iter=10,
        verbose=False,
    )
    model.fit(x, y)
    model.fit_residual_corrector(x, y, epochs=50, patience=10)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        model.save(path)
        loaded = LAKERRegressor.load(path)

        assert loaded.residual_corrector is not None
        x_test = torch.rand(10, 2) * 100.0
        pred_orig = model.predict(x_test)
        pred_load = loaded.predict(x_test)
        torch.testing.assert_close(pred_orig, pred_load, rtol=1e-5, atol=1e-5)
    finally:
        os.unlink(path)
