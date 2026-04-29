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
