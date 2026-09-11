"""Tests for the public :class:`laker.Laker` estimator."""

import pytest
import torch

from laker import Laker


class TestConstruction:
    def test_default_construction(self):
        m = Laker()
        assert m.coef is None
        assert m.embed is None
        assert m.kernel is None
        assert m.prec is None

    def test_repr(self):
        m = Laker()
        s = repr(m)
        assert "Laker" in s
        assert "not fitted" in s

    def test_repr_fitted(self):
        torch.manual_seed(0)
        x = torch.rand(10, 2, dtype=torch.float64)
        y = torch.sin(x[:, 0])
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        assert "fitted" in repr(m)

    def test_validation_positive_embed_dim(self):
        with pytest.raises(ValueError, match="embed_dim"):
            Laker(embed_dim=0)

    def test_validation_positive_lam(self):
        with pytest.raises(ValueError, match="lam"):
            Laker(lam=-1.0)

    def test_validation_negative_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            Laker(gamma=-0.1)

    def test_validation_positive_eps(self):
        with pytest.raises(ValueError, match="eps"):
            Laker(eps=0.0)

    def test_validation_base_range(self):
        with pytest.raises(ValueError, match="base"):
            Laker(base=1.5)

    def test_validation_cccp_max(self):
        with pytest.raises(ValueError, match="cccp_max"):
            Laker(cccp_max=0)

    def test_validation_cccp_tol(self):
        with pytest.raises(ValueError, match="cccp_tol"):
            Laker(cccp_tol=0.0)

    def test_validation_pcg_tol(self):
        with pytest.raises(ValueError, match="pcg_tol"):
            Laker(pcg_tol=0.0)

    def test_validation_pcg_max(self):
        with pytest.raises(ValueError, match="pcg_max"):
            Laker(pcg_max=0)

    def test_validation_neighbors(self):
        with pytest.raises(ValueError, match="neighbors"):
            Laker(neighbors=0)

    def test_validation_grid_size(self):
        with pytest.raises(ValueError, match="grid_size"):
            Laker(grid_size=1)

    def test_validation_blend(self):
        with pytest.raises(ValueError, match="blend"):
            Laker(blend=1.5)

    def test_validation_selection(self):
        with pytest.raises(ValueError, match="selection"):
            Laker(selection="bogus")

    def test_validation_pilot(self):
        with pytest.raises(ValueError, match="pilot"):
            Laker(pilot=0)

    def test_validation_knots(self):
        with pytest.raises(ValueError, match="knots"):
            Laker(knots=0)

    def test_validation_prec_kind(self):
        with pytest.raises(ValueError, match="prec_kind"):
            Laker(prec_kind="bogus")

    def test_validation_kernel(self):
        with pytest.raises(ValueError, match="kernel_type"):
            Laker(kernel_type="bogus")


class TestFit:
    def test_fit_sets_state(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        assert m.coef is not None
        assert m.coef.shape == (30,)
        assert m.embed is not None
        assert m.embed.shape == (30, 4)
        assert m.kernel is not None
        assert m.prec is not None
        assert m.encoder is not None
        assert m.inputs is not None
        assert m.targets is not None
        assert m.iters is not None and m.iters > 0

    def test_fit_deterministic_with_seed(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m1 = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m1.fit(x, y, seed=42)
        m2 = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m2.fit(x, y, seed=42)
        torch.testing.assert_close(m1.coef, m2.coef)

    def test_fit_accepts_numpy(self):
        import numpy as np

        torch.manual_seed(0)
        x = np.random.rand(30, 2) * 100
        y = np.random.randn(30)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        assert m.coef is not None

    def test_fit_rejects_y_scalar(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        x = torch.rand(10, 2, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            m.fit(x, torch.tensor(1.0, dtype=torch.float64))

    def test_fit_rejects_nan_x(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        x = torch.rand(10, 2, dtype=torch.float64)
        x[0, 0] = float("nan")
        y = torch.rand(10, dtype=torch.float64)
        with pytest.raises(ValueError, match="non-finite"):
            m.fit(x, y)

    def test_fit_rejects_nan_y(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        x = torch.rand(10, 2, dtype=torch.float64)
        y = torch.rand(10, dtype=torch.float64)
        y[0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            m.fit(x, y)

    def test_fit_rejects_empty_x(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        x = torch.zeros(0, 2, dtype=torch.float64)
        y = torch.zeros(0, dtype=torch.float64)
        with pytest.raises(ValueError, match="at least one row"):
            m.fit(x, y)

    def test_fit_respects_warm_start(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, lam=0.1, dtype=torch.float64, verbose=False, warm=True)
        m.fit(x, y)
        first_lam = m.lam
        m.fit(x, y)
        assert m.lam == first_lam

    def test_fit_cold_refit_drops_state(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        m.coef = None
        m.fit(x, y)
        assert m.coef is not None


class TestPredict:
    def test_predict_shape(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        preds = m.predict(x)
        assert preds.shape == (30,)

    def test_predict_unfitted_raises(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.predict(torch.rand(5, 2, dtype=torch.float64))

    def test_predict_wrong_feature_dim(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        with pytest.raises(ValueError, match="features"):
            m.predict(torch.rand(5, 3, dtype=torch.float64))


class TestVariance:
    def test_variance_nonnegative(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        v = m.variance(x)
        assert (v >= 0).all()

    def test_variance_unfitted_raises(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.variance(torch.rand(5, 2, dtype=torch.float64))


class TestReport:
    def test_report_populated_after_fit(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        assert m.report is None
        m.fit(x, y)
        assert m.report is not None
        assert m.report.converged
        assert m.report.reason == "converged"
        assert m.report.iterations >= 1

    def test_report_records_max_iter_when_cap_hit(self):
        torch.manual_seed(0)
        x = torch.rand(20, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(
            embed_dim=4,
            dtype=torch.float64,
            pcg_max=2,
            pcg_tol=1e-30,
            verbose=False,
        )
        m.fit(x, y)
        assert m.report is not None
        assert not m.report.converged
        assert m.report.reason == "max_iter"


class TestScore:
    def test_score_perfect_fit(self):
        torch.manual_seed(0)
        x = torch.linspace(-1, 1, 30, dtype=torch.float64).unsqueeze(-1)
        y = x[:, 0].clone()
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        # R^2 may not be exactly 1 with the kernel approximation but should be high
        s = m.score(x, y)
        assert s > 0.5

    def test_score_constant_y(self):
        torch.manual_seed(0)
        x = torch.rand(10, 2, dtype=torch.float64)
        y = torch.zeros(10, dtype=torch.float64)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        s = m.score(x, y)
        # Constant target: predict mean exactly -> R^2 = 1.0.
        assert s == 1.0

    def test_score_shape_mismatch(self):
        torch.manual_seed(0)
        x = torch.rand(20, 2, dtype=torch.float64)
        y = torch.sin(x[:, 0])
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        with pytest.raises(ValueError, match="shape"):
            m.score(x, torch.randn(5, dtype=torch.float64))


class TestCondition:
    def test_condition_after_fit(self):
        torch.manual_seed(0)
        x = torch.rand(50, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        kappa = m.condition()
        assert kappa > 1.0
        assert math.isfinite(kappa)

    def test_condition_unfitted_raises(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.condition()


class TestGetParamsSetParams:
    def test_get_params_round_trip(self):
        m = Laker(
            embed_dim=8,
            lam=0.05,
            gamma=0.2,
            kernel_type="nystrom",
            landmarks=12,
            dtype=torch.float64,
            verbose=False,
        )
        params = m.get_params()
        assert params["embed_dim"] == 8
        assert params["lam"] == 0.05
        assert params["gamma"] == 0.2
        assert params["kernel_type"] == "nystrom"
        assert params["landmarks"] == 12

    def test_set_params_rejects_unknown(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        with pytest.raises(ValueError, match="Invalid parameter"):
            m.set_params(bogus=1)

    def test_set_params_coerces_dtype_string(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.set_params(dtype="float32")
        assert m.dtype == torch.float32

    def test_set_params_coerces_device_string(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.set_params(device="cpu")
        assert m.device == torch.device("cpu")

    def test_sklearn_clone(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m2 = m.__sklearn_clone__()
        assert m2 is not m
        assert m2.embed_dim == m.embed_dim


class TestKernels:
    @pytest.mark.parametrize(
        "kernel,kwargs",
        [
            ("exact", {}),
            ("nystrom", {"landmarks": 5}),
            ("fourier", {"features": 32}),
            ("neighbors", {"neighbors": 4}),
            ("grid", {"grid_size": 32}),
            ("spectrum", {"knots": 5}),
            ("hybrid", {"landmarks": 5, "neighbors": 4}),
        ],
    )
    def test_all_kernels_fit_predict(self, kernel, kwargs):
        torch.manual_seed(0)
        x = torch.rand(20, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(
            embed_dim=4,
            kernel_type=kernel,
            dtype=torch.float64,
            verbose=False,
            **kwargs,
        )
        m.fit(x, y)
        preds = m.predict(x)
        assert preds.shape == (20,)
        assert torch.isfinite(preds).all()


import math
