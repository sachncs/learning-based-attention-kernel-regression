"""Tests for :mod:`laker.search`."""

import torch

from laker import Laker


class TestGrid:
    def test_grid_picks_best(self):
        torch.manual_seed(0)
        x = torch.rand(40, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50) + 0.1 * torch.randn(40, dtype=torch.float64)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.searcher.grid(
            m,
            x,
            y,
            lam_grid=[1e-3, 1e-2, 1e-1],
            gamma_grid=[0.0, 0.1, 1.0],
            num_grid=[20, 50],
            val=0.2,
            warm=True,
            seed=0,
        )
        assert m.lam in [1e-3, 1e-2, 1e-1]
        assert m.gamma in [0.0, 0.1, 1.0]
        assert m.num in [20, 50]

    def test_grid_final_lam_changed(self):
        torch.manual_seed(0)
        x = torch.rand(40, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, lam=1.0, dtype=torch.float64, verbose=False)
        m.searcher.grid(
            m,
            x,
            y,
            lam_grid=[1e-4, 1e-3, 1e-2],
            gamma_grid=[0.0, 0.1],
            num_grid=[20],
            val=0.2,
            warm=True,
            seed=0,
        )
        assert m.lam != 1.0

    def test_grid_via_public_search(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.search(x, y, lam_grid=[1e-2, 1e-1], val=0.2, warm=True, seed=0)
        assert m.lam in [1e-2, 1e-1]


class TestBayes:
    def test_bayes_runs(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.bayes(
            x,
            y,
            val=0.2,
            n_calls=5,
            n_init=3,
            seed=0,
        )
        assert m.lam > 0
        assert m.gamma >= 0
        assert m.num > 0

    def test_bayes_reproducible_with_seed(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m1 = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m2 = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m1.bayes(x, y, val=0.2, n_calls=5, n_init=3, seed=42)
        m2.bayes(x, y, val=0.2, n_calls=5, n_init=3, seed=42)
        assert m1.lam == m2.lam
        assert m1.gamma == m2.gamma
        assert m1.num == m2.num
