"""Tests for input validation and edge cases."""

import pytest
import torch

from laker.models import LAKERRegressor


def test_invalid_embedding_dim():
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        LAKERRegressor(embedding_dim=0)


def test_invalid_lambda_reg():
    with pytest.raises(ValueError, match="lambda_reg must be positive"):
        LAKERRegressor(lambda_reg=0.0)
    with pytest.raises(ValueError, match="lambda_reg must be positive"):
        LAKERRegressor(lambda_reg=-1.0)


def test_invalid_gamma():
    with pytest.raises(ValueError, match="gamma must be non-negative"):
        LAKERRegressor(gamma=-0.1)


def test_invalid_epsilon():
    with pytest.raises(ValueError, match="epsilon must be positive"):
        LAKERRegressor(epsilon=0.0)


def test_invalid_base_rho():
    with pytest.raises(ValueError, match="base_rho must be in \\[0, 1\\]"):
        LAKERRegressor(base_rho=-0.1)
    with pytest.raises(ValueError, match="base_rho must be in \\[0, 1\\]"):
        LAKERRegressor(base_rho=1.5)


def test_invalid_max_iter():
    with pytest.raises(ValueError, match="cccp_max_iter must be positive"):
        LAKERRegressor(cccp_max_iter=0)
    with pytest.raises(ValueError, match="pcg_max_iter must be positive"):
        LAKERRegressor(pcg_max_iter=0)


def test_invalid_tol():
    with pytest.raises(ValueError, match="cccp_tol must be positive"):
        LAKERRegressor(cccp_tol=0.0)
    with pytest.raises(ValueError, match="pcg_tol must be positive"):
        LAKERRegressor(pcg_tol=0.0)


def test_predict_before_fit():
    model = LAKERRegressor(verbose=False)
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        model.predict(torch.rand(5, 2))


def test_save_before_fit():
    model = LAKERRegressor(verbose=False)
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        model.save("/tmp/test.pt")


def test_fit_mismatched_shapes():
    model = LAKERRegressor(verbose=False)
    x = torch.rand(10, 2)
    y = torch.rand(10, 2)  # wrong shape
    with pytest.raises(ValueError, match="y must be 1-D"):
        model.fit(x, y)


def test_get_set_params():
    model = LAKERRegressor(lambda_reg=0.5, verbose=False)
    params = model.get_params()
    assert params["lambda_reg"] == 0.5
    model.set_params(lambda_reg=0.1)
    assert model.lambda_reg == 0.1
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(invalid_param=1)
