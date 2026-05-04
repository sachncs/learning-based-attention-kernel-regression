"""Tests for the high-level LAKERRegressor."""

import torch

from laker.models import LAKERRegressor


def test_regressor_fit_predict():
    """End-to-end fit and predict on a small synthetic problem."""
    n = 80
    x_train = torch.rand(n, 2) * 100.0
    y_train = torch.randn(n)

    model = LAKERRegressor(
        embedding_dim=6,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=40,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-8,
        pcg_max_iter=200,
        chunk_size=32,
        verbose=False,
    )
    model.fit(x_train, y_train)

    assert model.alpha is not None
    assert model.alpha.shape == (n,)

    m = 30
    x_test = torch.rand(m, 2) * 100.0
    y_pred = model.predict(x_test)
    assert y_pred.shape == (m,)


def test_regressor_paper_example():
    """Reproduce the n=3 worked example from the paper (Section IV-E)."""
    # Exact embeddings and observations from Eq. (53)
    e = torch.tensor(
        [[0.241, 0.444], [-0.336, 0.112], [-0.220, 0.353]],
        dtype=torch.float64,
    )
    y = torch.tensor([-66.14, -65.77, -77.30], dtype=torch.float64)

    # Build kernel explicitly to verify
    g = torch.exp(e @ e.T)
    a_mat = g + 0.1 * torch.eye(3, dtype=torch.float64)
    alpha_exact = torch.linalg.solve(a_mat, y)

    # LAKER should recover the same coefficients
    # We bypass the embedding module by providing a custom one
    class FixedEmbedding(torch.nn.Module):
        def forward(self, x):
            return e

    model = LAKERRegressor(
        embedding_dim=2,
        lambda_reg=0.1,
        gamma=0.0,  # minimal regularisation for tiny problem
        num_probes=3,
        cccp_max_iter=10,
        cccp_tol=1e-6,
        pcg_tol=1e-12,
        pcg_max_iter=10,
        chunk_size=None,
        embedding_module=FixedEmbedding(),
        verbose=False,
        dtype=torch.float64,
    )
    # Dummy locations (ignored by FixedEmbedding)
    x_dummy = torch.zeros(3, 2, dtype=torch.float64)
    model.fit(x_dummy, y)

    torch.testing.assert_close(model.alpha, alpha_exact, rtol=1e-4, atol=1e-3)

    # Verify prediction at query point from Eq. (57-58)
    e_star = torch.tensor([[0.051, 0.452]], dtype=torch.float64)
    k_star = torch.exp(e_star @ e.T)
    pred_manual = (k_star @ alpha_exact).item()

    class QueryEmbedding(torch.nn.Module):
        def forward(self, x):
            return e_star

    model.embedding_model = QueryEmbedding()
    pred_model = model.predict(torch.zeros(1, 2, dtype=torch.float64)).item()
    assert abs(pred_model - pred_manual) < 1e-3


def test_regressor_score():
    """Score should be negative RMSE."""
    n = 60
    x = torch.rand(n, 2) * 100.0
    y = torch.randn(n)

    model = LAKERRegressor(
        embedding_dim=4,
        num_probes=30,
        cccp_max_iter=10,
        verbose=False,
    )
    model.fit(x, y)
    score = model.score(x, y)
    assert score <= 0.0  # negative RMSE


def test_residual_corrector_fitted_and_active():
    """Residual corrector should be trained and produce non-zero corrections."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2) * 100.0
    y = torch.randn(n)

    model = LAKERRegressor(
        embedding_dim=6,
        lambda_reg=1e-2,
        num_probes=40,
        cccp_max_iter=20,
        verbose=False,
    )
    model.fit(x, y)

    assert model.residual_corrector is None

    model.fit_residual_corrector(x, y, epochs=100, patience=10)

    assert model.residual_corrector is not None
    with torch.no_grad():
        corr = model.residual_corrector(x).squeeze()
    assert torch.norm(corr).item() > 0.0


def test_bilevel_runs_and_produces_fitted_model():
    """Bilevel hyperparameter learning should run and produce a fitted model."""
    torch.manual_seed(42)
    n = 60
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    n_val = 12
    x_train = x[n_val:]
    y_train = y[n_val:]
    x_val = x[:n_val]
    y_val = y[:n_val]

    model = LAKERRegressor(
        embedding_dim=4,
        lambda_reg=1e-1,
        num_probes=30,
        cccp_max_iter=10,
        pcg_tol=1e-6,
        pcg_max_iter=200,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x_train, y_train)

    base_pred = model.predict(x_val)
    base_loss = torch.mean((base_pred - y_val) ** 2).item()

    model.fit_bilevel(x_train, y_train, x_val, y_val, lr=5e-3, epochs=5, patience=3)

    assert model.alpha is not None
    bilevel_pred = model.predict(x_val)
    bilevel_loss = torch.mean((bilevel_pred - y_val) ** 2).item()

    # Bilevel should not catastrophically worsen validation loss
    assert bilevel_loss < base_loss * 5.0


def test_uncertainty_aware_runs_and_produces_fitted_model():
    """Uncertainty-aware training should run and produce a fitted model."""
    torch.manual_seed(42)
    n = 60
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=4,
        lambda_reg=1e-1,
        num_probes=30,
        cccp_max_iter=10,
        pcg_tol=1e-6,
        pcg_max_iter=200,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)

    assert model.alpha is not None

    model.fit_uncertainty_aware(x, y, lr=1e-2, epochs=10, beta=0.1, variance_subset=0.3, patience=3)

    assert model.alpha is not None
    x_test = torch.rand(10, 2, dtype=torch.float64) * 100.0
    y_pred = model.predict(x_test)
    assert y_pred.shape == (10,)
    var = model.predict_variance(x_test)
    assert var.shape == (10,)
    assert torch.all(var >= 0)
