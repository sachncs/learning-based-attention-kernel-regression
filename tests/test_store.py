"""Tests for :mod:`laker.store`."""

import os

import pytest
import torch

from laker import Laker
from laker.store import Store


@pytest.fixture
def fitted_model():
    torch.manual_seed(0)
    x = torch.rand(30, 2, dtype=torch.float64) * 100
    y = torch.sin(x[:, 0] / 50)
    m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
    m.fit(x, y)
    return m, x, y


class TestSaveLoad:
    def test_save_load_predictions_match(self, fitted_model, tmp_path):
        m, x, y = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        assert os.path.exists(path)
        m2 = Laker.load(path)
        torch.testing.assert_close(m.predict(x), m2.predict(x))

    def test_save_load_score_matches(self, fitted_model, tmp_path):
        m, x, y = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        m2 = Laker.load(path)
        s1 = m.score(x, y)
        s2 = m2.score(x, y)
        assert abs(s1 - s2) < 1e-8

    def test_save_load_variance_matches(self, fitted_model, tmp_path):
        m, x, _ = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        m2 = Laker.load(path)
        v1 = m.variance(x)
        v2 = m2.variance(x)
        torch.testing.assert_close(v1, v2, atol=1e-6, rtol=1e-6)

    def test_save_rejects_unfitted(self, tmp_path):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.save(str(tmp_path / "x.pt"))

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            Store.load("/nonexistent/path.pt")

    def test_save_includes_format_version(self, fitted_model, tmp_path):
        m, _, _ = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        state = torch.load(path, weights_only=True)
        assert "format" in state
        assert state["format"] >= 2

    def test_save_includes_hyperparameters(self, fitted_model, tmp_path):
        m, _, _ = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        state = torch.load(path, weights_only=True)
        for key in ("embed_dim", "lam", "gamma", "num", "kernel_type"):
            assert key in state

    def test_save_includes_fitted_tensors(self, fitted_model, tmp_path):
        m, _, _ = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        state = torch.load(path, weights_only=True)
        assert "embed" in state
        assert "coef" in state


class TestKernelRoundTrip:
    @pytest.mark.parametrize(
        "kernel,kwargs",
        [
            ("exact", {}),
            ("nystrom", {"landmarks": 5}),
            ("fourier", {"features": 32}),
            ("neighbors", {"neighbors": 4}),
            ("grid", {"grid_size": 32}),
            ("spectrum", {"knots": 5}),
        ],
    )
    def test_kernelpersistence(self, tmp_path, kernel, kwargs):
        torch.manual_seed(0)
        x = torch.rand(15, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, kernel_type=kernel, dtype=torch.float64, verbose=False, **kwargs)
        m.fit(x, y)
        path = str(tmp_path / "model.pt")
        m.save(path)
        m2 = Laker.load(path)
        torch.testing.assert_close(m.predict(x), m2.predict(x), atol=1e-6, rtol=1e-6)

    def test_hybrid_kernelpersistence(self, tmp_path):
        torch.manual_seed(0)
        x = torch.rand(15, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(
            embed_dim=4,
            kernel_type="hybrid",
            landmarks=5,
            neighbors=4,
            dtype=torch.float64,
            verbose=False,
        )
        m.fit(x, y)
        path = str(tmp_path / "model.pt")
        m.save(path)
        m2 = Laker.load(path)
        torch.testing.assert_close(m.predict(x), m2.predict(x), atol=1e-6, rtol=1e-6)


class TestCorrectorPersistence:
    def test_corrector_round_trip(self, tmp_path):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50) + 0.1 * torch.randn(30, dtype=torch.float64)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        m.correct(x, y, val=0.2, epochs=3, patience=5, lr=1e-2, seed=0)
        path = str(tmp_path / "model.pt")
        m.save(path)
        m2 = Laker.load(path)
        assert m2.corrector is not None
        torch.testing.assert_close(m.predict(x), m2.predict(x), atol=1e-6, rtol=1e-6)


class TestDeviceDtype:
    def test_load_preserves_float64(self, fitted_model, tmp_path):
        m, x, _ = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        m2 = Laker.load(path)
        assert m2.dtype == torch.float64
        assert m2.coef.dtype == torch.float64

    def test_load_preserves_float16(self, fitted_model, tmp_path):
        m, x, _ = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        state = torch.load(path, weights_only=True)
        state["dtype"] = "torch.float16"
        torch.save(state, path)
        m2 = Laker.load(path)
        assert m2.dtype == torch.float16

    def test_load_preserves_bfloat16_embed(self, fitted_model, tmp_path):
        m, x, _ = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        state = torch.load(path, weights_only=True)
        state["embed_dtype"] = "torch.bfloat16"
        torch.save(state, path)
        m2 = Laker.load(path)
        assert m2.embed_dtype == torch.bfloat16

    def test_load_unknown_dtype_raises(self, fitted_model, tmp_path):
        m, x, _ = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        state = torch.load(path, weights_only=True)
        state["dtype"] = "torch.float7"
        torch.save(state, path)
        with pytest.raises(ValueError, match="unsupported dtype"):
            Store.load(path)

    def test_load_missing_file_raises_with_path(self, tmp_path):
        path = str(tmp_path / "missing.pt")
        with pytest.raises(FileNotFoundError, match=path):
            Store.load(path)

    def test_load_non_laker_file_raises(self, tmp_path):
        path = str(tmp_path / "not-laker.pt")
        torch.save({"weight": torch.zeros(3)}, path)
        with pytest.raises(ValueError, match="not a LAKER model file"):
            Store.load(path)

    def test_load_future_format_rejected(self, tmp_path):
        path = str(tmp_path / "future.pt")
        torch.save(
            {
                "format": 99,
                "dtype": "torch.float32",
                "embed_dim": 4,
                "lam": 1e-2,
            },
            path,
        )
        with pytest.raises(ValueError, match="newer LAKER version"):
            Store.load(path)

    def test_load_missing_required_field(self, tmp_path):
        path = str(tmp_path / "incomplete.pt")
        torch.save(
            {
                "format": 2,
                "dtype": "torch.float32",
                "embed_dim": 4,
            },
            path,
        )
        with pytest.raises(KeyError, match="missing required field"):
            Store.load(path)
