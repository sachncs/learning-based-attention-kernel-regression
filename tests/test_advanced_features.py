"""Tests for advanced features: mixed-precision, grid search, low-rank kernels."""

import torch

from laker.models import LAKERRegressor


def test_mixed_precision_embedding():
    """Mixed-precision embeddings should still solve correctly."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float32) * 100.0
    y = torch.randn(n, dtype=torch.float32)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        embedding_dtype=torch.float16,
        dtype=torch.float32,
        verbose=False,
    )
    model.fit(x, y)

    assert model.embeddings.dtype == torch.float32
    assert model.alpha is not None
    y_pred = model.predict(x[:10])
    assert y_pred.shape == (10,)


def test_nystrom_integration():
    """LAKERRegressor with Nyström kernel should fit and predict."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        kernel_approx="nystrom",
        num_landmarks=50,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)
    assert model.alpha is not None

    x_test = torch.rand(10, 2, dtype=torch.float64) * 100.0
    y_pred = model.predict(x_test)
    assert y_pred.shape == (10,)


def test_rff_integration():
    """LAKERRegressor with RFF kernel should fit and predict."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        kernel_approx="rff",
        num_features=200,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)
    assert model.alpha is not None

    x_test = torch.rand(10, 2, dtype=torch.float64) * 100.0
    y_pred = model.predict(x_test)
    assert y_pred.shape == (10,)


def test_grid_search():
    """fit_with_search should select hyperparameters and fit."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit_with_search(
        x,
        y,
        val_fraction=0.2,
        lambda_reg_grid=[1e-2, 1e-1],
        gamma_grid=[0.0, 1e-1],
        num_probes_grid=[30, 50],
    )

    assert model.alpha is not None
    assert model.lambda_reg in [1e-2, 1e-1]
    assert model.gamma in [0.0, 1e-1]
    assert model.num_probes in [30, 50]


def test_invalid_kernel_approx():
    """Invalid kernel_approx should raise ValueError."""
    with torch.no_grad():
        try:
            LAKERRegressor(kernel_approx="invalid")
            assert False, "Expected ValueError"
        except ValueError:
            pass


def test_fit_path():
    """fit_path should return a sequence of alphas for each lambda."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        dtype=torch.float64,
        verbose=False,
    )
    path = model.fit_path(x, y, lambda_reg_grid=[1e-1, 1e-2, 1e-3])

    assert len(path["lambda_reg"]) == 3
    assert len(path["alphas"]) == 3
    assert len(path["pcg_iters"]) == 3
    assert len(path["final_rel_res"]) == 3
    # Warm-start should reduce iterations for smaller lambda (descending order)
    assert path["lambda_reg"][0] > path["lambda_reg"][1]


def test_predict_variance_exact():
    """predict_variance should return non-negative values for exact kernel."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)

    x_test = torch.rand(20, 2, dtype=torch.float64) * 100.0
    var = model.predict_variance(x_test)
    assert var.shape == (20,)
    assert torch.all(var >= 0)


def test_predict_variance_rff():
    """predict_variance should return non-negative values for RFF kernel."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        kernel_approx="rff",
        num_features=200,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)

    x_test = torch.rand(20, 2, dtype=torch.float64) * 100.0
    var = model.predict_variance(x_test)
    assert var.shape == (20,)
    assert torch.all(var >= 0)


def test_knn_integration():
    """LAKERRegressor with sparse k-NN kernel should fit and predict."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        kernel_approx="knn",
        k_neighbors=20,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)
    assert model.alpha is not None

    x_test = torch.rand(10, 2, dtype=torch.float64) * 100.0
    y_pred = model.predict(x_test)
    assert y_pred.shape == (10,)


def test_knn_matvec_consistency():
    """Sparse k-NN matvec should match dense for small k."""
    from laker.kernels import (
        AttentionKernelOperator,
        SparseKNNAttentionKernelOperator,
    )

    torch.manual_seed(42)
    n = 50
    e = torch.randn(n, 10, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)

    dense_op = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    sparse_op = SparseKNNAttentionKernelOperator(
        e, lambda_reg=1e-2, k_neighbors=n, dtype=torch.float64
    )

    y_dense = dense_op.matvec(x)
    y_sparse = sparse_op.matvec(x)
    assert torch.allclose(y_dense, y_sparse, atol=1e-6)


def test_bo_search():
    """fit_with_bo should select hyperparameters and fit."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit_with_bo(
        x,
        y,
        val_fraction=0.2,
        n_calls=8,
        n_initial_points=3,
        lambda_reg_bounds=(1e-2, 1e-1),
        gamma_bounds=(0.0, 1e-1),
        num_probes_bounds=(30, 50),
    )
    assert model.alpha is not None
    assert model.lambda_reg > 0
    assert model.gamma >= 0


def test_partial_fit():
    """partial_fit should incrementally update the model."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x[:80], y[:80])
    assert model.alpha is not None
    old_n = model.embeddings.shape[0]

    model.partial_fit(x[80:90], y[80:90])
    assert model.embeddings.shape[0] == old_n + 10
    assert model.alpha.shape[0] == old_n + 10


def test_learned_embeddings():
    """fit_learned_embeddings should reduce training loss."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)
    initial_res = torch.linalg.norm(model.kernel_operator.matvec(model.alpha) - y).item()

    model.fit_learned_embeddings(x, y, lr=1e-2, epochs=20, rebuild_freq=5, patience=10)
    final_res = torch.linalg.norm(model.kernel_operator.matvec(model.alpha) - y).item()
    assert final_res <= initial_res


def test_distributed_fallback():
    """Distributed kernel should fall back to single-device when only one GPU."""
    from laker.distributed_kernels import DistributedAttentionKernelOperator

    torch.manual_seed(42)
    n = 50
    e = torch.randn(n, 10, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)

    dist_op = DistributedAttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    assert dist_op.single_device

    y_dist = dist_op.matvec(x)
    y_diag = dist_op.diagonal()
    assert y_dist.shape == (n,)
    assert y_diag.shape == (n,)


def test_ski_integration():
    """LAKERRegressor with SKI kernel should fit and predict."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=6,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        kernel_approx="ski",
        grid_size=64,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)
    assert model.alpha is not None

    x_test = torch.rand(10, 2, dtype=torch.float64) * 100.0
    y_pred = model.predict(x_test)
    assert y_pred.shape == (10,)


def test_spectral_integration():
    """LAKERRegressor with spectral kernel should fit and predict."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        kernel_approx="spectral",
        spectral_knots=5,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)
    assert model.alpha is not None

    x_test = torch.rand(10, 2, dtype=torch.float64) * 100.0
    y_pred = model.predict(x_test)
    assert y_pred.shape == (10,)


def test_twoscale_integration():
    """LAKERRegressor with two-scale kernel should fit and predict."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        kernel_approx="twoscale",
        num_landmarks=50,
        k_neighbors=20,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)
    assert model.alpha is not None

    x_test = torch.rand(10, 2, dtype=torch.float64) * 100.0
    y_pred = model.predict(x_test)
    assert y_pred.shape == (10,)


def test_continuation_integration():
    """fit_continuation should produce a fitted model with decreasing schedule."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit_continuation(x, y, lambda_max=1.0, lambda_min=1e-2, n_stages=4)

    assert model.alpha is not None
    assert hasattr(model, "path_")
    path = model.path_
    assert len(path["lambda_reg"]) == 4
    assert (
        path["lambda_reg"][0]
        > path["lambda_reg"][1]
        > path["lambda_reg"][2]
        > path["lambda_reg"][3]
    )
    assert abs(path["lambda_reg"][-1] - 1e-2) < 1e-10

    x_test = torch.rand(10, 2, dtype=torch.float64) * 100.0
    y_pred = model.predict(x_test)
    assert y_pred.shape == (10,)
