"""Tests for :mod:`laker.stream`."""

import pytest
import torch

from laker import Laker


class TestUpdate:
    def test_update_grows_state(self):
        torch.manual_seed(0)
        n = 20
        x = torch.rand(n, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        assert m.coef.shape[0] == n

        x_new = torch.rand(5, 2, dtype=torch.float64) * 100
        y_new = torch.sin(x_new[:, 0] / 50)
        m.update(x_new, y_new, threshold=100, seed=0)
        assert m.coef.shape[0] == n + 5
        assert m.embed.shape[0] == n + 5

    def test_update_matches_fresh_fit_on_concatenated_data(self):
        torch.manual_seed(0)
        n = 20
        x = torch.rand(n, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)

        x_new = torch.rand(5, 2, dtype=torch.float64) * 100
        y_new = torch.sin(x_new[:, 0] / 50)
        m.update(x_new, y_new, threshold=100, seed=0)

        ref = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        ref.fit(torch.cat([x, x_new]), torch.cat([y, y_new]))
        torch.testing.assert_close(m.coef, ref.coef, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(m.y_train, torch.cat([y, y_new]))

    def test_update_rejects_unfitted(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        x = torch.rand(5, 2, dtype=torch.float64)
        y = torch.rand(5, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.update(x, y)

    def test_update_threshold_raises(self):
        torch.manual_seed(0)
        x = torch.rand(20, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        x_big = torch.rand(150, 2, dtype=torch.float64) * 100
        y_big = torch.sin(x_big[:, 0] / 50)
        with pytest.raises(RuntimeError, match="threshold"):
            m.update(x_big, y_big, threshold=100, autofit=False)

    def test_update_autofits_when_threshold_exceeded(self):
        torch.manual_seed(0)
        x = torch.rand(20, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        x_big = torch.rand(150, 2, dtype=torch.float64) * 100
        y_big = torch.sin(x_big[:, 0] / 50)
        m.update(x_big, y_big, threshold=100)
        assert m.coef.shape[0] == 20 + 150
        assert torch.isfinite(m.coef).all()

    def test_update_reproducible_with_seed(self):
        torch.manual_seed(0)
        x = torch.rand(20, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m1 = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m2 = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m1.fit(x, y)
        m2.fit(x, y)
        x_new = torch.rand(5, 2, dtype=torch.float64) * 100
        y_new = torch.sin(x_new[:, 0] / 50)
        m1.update(x_new, y_new, threshold=100, seed=42)
        m2.update(x_new, y_new, threshold=100, seed=42)
        torch.testing.assert_close(m1.coef, m2.coef)
        assert m1.prec.iso == m2.prec.iso

    def test_update_shape_validation(self):
        torch.manual_seed(0)
        x = torch.rand(20, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        with pytest.raises(ValueError, match="x_new must be 2-D"):
            m.update(torch.randn(5, dtype=torch.float64), torch.randn(5, dtype=torch.float64))


class TestPath:
    def test_path_returns_dict(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        path = m.path(x, y, [0.1, 0.01, 0.001], reuse=True)
        assert "lam" in path
        assert "coef" in path
        assert "iters" in path
        assert "rel" in path
        assert len(path["lam"]) == 3
        assert len(path["coef"]) == 3
        assert len(path["iters"]) == 3
        assert len(path["rel"]) == 3

    def test_path_sorted_descending(self):
        torch.manual_seed(0)
        x = torch.rand(20, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        path = m.path(x, y, [0.001, 0.01, 0.1, 1.0], reuse=True)
        assert path["lam"] == sorted(path["lam"], reverse=True)

    def test_path_rejects_empty_grid(self):
        torch.manual_seed(0)
        x = torch.rand(20, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        with pytest.raises(ValueError, match="must not be empty"):
            m.path(x, y, [])


class TestContinuation:
    def test_continuation_runs(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.continuation(x, y, lo=1e-3, hi=1.0, stages=3)
        assert m.lam == 1e-3
        assert torch.isfinite(m.coef).all()

    def test_continuation_rejects_bad_stages(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        x = torch.rand(10, 2, dtype=torch.float64)
        y = torch.rand(10, dtype=torch.float64)
        with pytest.raises(ValueError, match="stages"):
            m.continuation(x, y, stages=0)

    def test_continuation_rejects_nonpositive(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        x = torch.rand(10, 2, dtype=torch.float64)
        y = torch.rand(10, dtype=torch.float64)
        with pytest.raises(ValueError, match="positive"):
            m.continuation(x, y, lo=0.0)
