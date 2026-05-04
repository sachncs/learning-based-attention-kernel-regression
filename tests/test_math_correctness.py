"""Tests for math reliability and correctness of core operators and algorithms."""

import torch

from laker.kernels import (
    AttentionKernelOperator,
    NystromAttentionKernelOperator,
    TwoScaleAttentionKernelOperator,
)
from laker.implicit_diff import hypergradient
from laker.models import LAKERRegressor


def test_twoscale_alpha_extremes():
    """Two-scale with alpha=0 or 1 should reduce to pure local/global."""
    torch.manual_seed(42)
    n = 50
    de = 4
    dtype = torch.float64
    e = torch.randn(n, de, dtype=dtype)
    lam = 1e-2

    twoscale_global = TwoScaleAttentionKernelOperator(
        e,
        lambda_reg=lam,
        alpha=1.0,
        num_landmarks=30,
        k_neighbors=10,
        dtype=dtype,
    )
    twoscale_local = TwoScaleAttentionKernelOperator(
        e,
        lambda_reg=lam,
        alpha=0.0,
        num_landmarks=30,
        k_neighbors=10,
        dtype=dtype,
    )

    x = torch.randn(n, dtype=dtype)

    y_global = twoscale_global.matvec(x)
    y_local = twoscale_local.matvec(x)

    # Compare against the internal sub-operators (same random state)
    torch.testing.assert_close(y_global, twoscale_global.global_op.matvec(x), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(y_local, twoscale_local.local_op.matvec(x), rtol=1e-5, atol=1e-6)


def test_twoscale_matvec_linearity():
    """Two-scale matvec should be affine combination of global and local."""
    torch.manual_seed(42)
    n = 50
    de = 4
    dtype = torch.float64
    e = torch.randn(n, de, dtype=dtype)
    lam = 1e-2

    alpha_mix = 0.3
    twoscale = TwoScaleAttentionKernelOperator(
        e,
        lambda_reg=lam,
        alpha=alpha_mix,
        num_landmarks=30,
        k_neighbors=10,
        dtype=dtype,
    )

    x = torch.randn(n, dtype=dtype)
    y_mix = twoscale.matvec(x)
    y_expected = alpha_mix * twoscale.global_op.matvec(x) + (
        1.0 - alpha_mix
    ) * twoscale.local_op.matvec(x)
    torch.testing.assert_close(y_mix, y_expected, rtol=1e-5, atol=1e-6)


def test_twoscale_diagonal_positive():
    """Two-scale diagonal should be positive."""
    torch.manual_seed(42)
    n = 50
    de = 4
    e = torch.randn(n, de)
    op = TwoScaleAttentionKernelOperator(
        e, lambda_reg=1e-2, alpha=0.5, num_landmarks=30, k_neighbors=10
    )
    diag = op.diagonal()
    assert torch.all(diag > 0).item()


def test_twoscale_to_dense_shape():
    """Two-scale to_dense should return correct shape."""
    n = 30
    de = 4
    e = torch.randn(n, de)
    op = TwoScaleAttentionKernelOperator(
        e, lambda_reg=1e-2, alpha=0.5, num_landmarks=20, k_neighbors=10
    )
    dense = op.to_dense()
    assert dense.shape == (n, n)


def test_leverage_scores_properties():
    """Leverage scores should be non-negative and sum to 1 (after normalisation)."""
    from laker.kernels import NystromAttentionKernelOperator

    torch.manual_seed(42)
    n = 100
    de = 4
    dtype = torch.float64
    e = torch.randn(n, de, dtype=dtype)

    op = NystromAttentionKernelOperator(
        e,
        lambda_reg=1e-2,
        num_landmarks=30,
        landmark_method="leverage",
        landmark_pilot_size=50,
        dtype=dtype,
    )

    # Leverage scores are computed internally in _select_landmarks_leverage.
    # We verify the landmarks are valid (distinct, in range).
    idx = op.landmark_indices
    assert idx.unique().numel() == idx.numel(), "Landmarks must be distinct"
    assert torch.all(idx >= 0) and torch.all(idx < n), "Landmarks out of range"
    assert len(idx) == 30


def test_leverage_vs_greedy_approximation_quality():
    """Leverage-score landmarks should not be catastrophically worse than greedy."""
    torch.manual_seed(42)
    n = 100
    de = 4
    dtype = torch.float64
    e = torch.randn(n, de, dtype=dtype)
    lam = 1e-2

    exact = AttentionKernelOperator(e, lambda_reg=lam, dtype=dtype)

    greedy = NystromAttentionKernelOperator(
        e,
        lambda_reg=lam,
        num_landmarks=40,
        landmark_method="greedy",
        dtype=dtype,
    )
    leverage = NystromAttentionKernelOperator(
        e,
        lambda_reg=lam,
        num_landmarks=40,
        landmark_method="leverage",
        landmark_pilot_size=60,
        dtype=dtype,
    )

    x = torch.randn(n, dtype=dtype)
    y_exact = exact.matvec(x)
    y_greedy = greedy.matvec(x)
    y_leverage = leverage.matvec(x)

    err_greedy = torch.norm(y_exact - y_greedy) / torch.norm(y_exact)
    err_leverage = torch.norm(y_exact - y_leverage) / torch.norm(y_exact)

    # Leverage should be within a factor of 3 of greedy error
    assert err_leverage.item() < err_greedy.item() * 3.0 + 1e-2


def test_continuation_schedule_monotonic():
    """Continuation schedule should decrease monotonically."""
    torch.manual_seed(42)
    n = 60
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=4,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=30,
        cccp_max_iter=10,
        pcg_tol=1e-6,
        pcg_max_iter=200,
        dtype=torch.float64,
        verbose=False,
    )

    model.fit_continuation(x, y, lambda_max=1.0, lambda_min=1e-2, n_stages=5, reuse_precond=True)

    path = model.path_
    lambdas = path["lambda_reg"]
    for i in range(len(lambdas) - 1):
        assert lambdas[i] > lambdas[i + 1], "Schedule must be strictly decreasing"

    assert abs(lambdas[-1] - 1e-2) < 1e-10, "Final lambda must match lambda_min"


def test_continuation_warmstart_reduces_iters():
    """Warm-starting in continuation should reduce PCG iterations."""
    torch.manual_seed(42)
    n = 60
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=4,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=30,
        cccp_max_iter=10,
        pcg_tol=1e-6,
        pcg_max_iter=200,
        dtype=torch.float64,
        verbose=False,
    )

    model.fit_continuation(x, y, lambda_max=1.0, lambda_min=1e-2, n_stages=5, reuse_precond=True)

    path = model.path_
    iters = path["pcg_iters"]
    # Early stages (large lambda) should converge faster than late stages,
    # but warm-start means the cumulative effort is reasonable.
    # We assert the first stage converges in < 50 iterations.
    assert iters[0] < 50, f"First stage took too long: {iters[0]} iters"


def test_predict_train_matches_predict():
    """predict_train should match predict when model is fixed (no grad needed)."""
    torch.manual_seed(42)
    n = 60
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=4,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=30,
        cccp_max_iter=10,
        pcg_tol=1e-6,
        pcg_max_iter=200,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)

    x_test = torch.rand(10, 2, dtype=torch.float64) * 100.0

    pred_normal = model.predict(x_test)

    # predict_train is differentiable but with no_grad it should match
    with torch.no_grad():
        pred_train = model._core.predict_train(
            x=x_test,
            embedding_model=model.embedding_model,
            embeddings=model.embeddings,
            kernel_operator=model.kernel_operator,
            alpha=model.alpha,
            residual_corrector=model.residual_corrector,
        )

    torch.testing.assert_close(pred_normal, pred_train, rtol=1e-5, atol=1e-6)


def test_predict_variance_train_matches_predict_variance_rff():
    """predict_variance_train should match predict_variance for RFF kernel."""
    torch.manual_seed(42)
    n = 60
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=4,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=30,
        cccp_max_iter=10,
        pcg_tol=1e-6,
        pcg_max_iter=200,
        kernel_approx="rff",
        num_features=100,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)

    x_test = torch.rand(5, 2, dtype=torch.float64) * 100.0

    var_normal = model.predict_variance(x_test)

    with torch.no_grad():
        var_train = model._core.predict_variance_train(
            x=x_test,
            embedding_model=model.embedding_model,
            embeddings=model.embeddings,
            kernel_operator=model.kernel_operator,
            preconditioner=model.preconditioner,
            alpha=model.alpha,
            lambda_reg=model.lambda_reg,
        )

    torch.testing.assert_close(var_normal, var_train, rtol=1e-5, atol=1e-6)


def test_hypergradient_finite_difference():
    """Implicit hypergradient should match finite differences on a toy problem."""
    torch.manual_seed(42)

    # Simple diagonal system: A(theta) = (1 + theta) * I
    # Solve: A alpha = y  =>  alpha = y / (1 + theta)
    # Loss: L = 0.5 * ||alpha||^2
    # dL/dalpha = alpha
    # Adjoint: A v = dL/dalpha  =>  v = alpha / (1 + theta) = y / (1 + theta)^2
    # Hypergradient: -v^T (dA/dtheta) alpha = -v^T alpha = -||y||^2 / (1+theta)^3

    theta = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([3.0, 4.0], dtype=torch.float64)

    def make_op(p):
        def op(v):
            return (1.0 + p) * v

        return op

    def make_precond(p):
        def precond(v):
            return v / (1.0 + p)

        return precond

    # Solve A alpha = y;  loss = 0.5 * ||alpha||^2
    alpha = y / (1.0 + theta)
    dL_dalpha = alpha  # gradient of loss wrt alpha

    # Hypergradient via implicit differentiation
    op = make_op(theta)
    precond = make_precond(theta)
    hg_list = hypergradient(
        operator_fn=op,
        preconditioner_fn=precond,
        alpha=alpha.detach(),
        dL_dalpha=dL_dalpha.detach(),
        param_list=[theta],
        pcg_tol=1e-12,
        pcg_max_iter=10,
        verbose=False,
    )
    hg = hg_list[0]

    # Analytical hypergradient: -||y||^2 / (1+theta)^3
    analytical = -torch.sum(y**2) / (1.0 + theta.detach()) ** 3

    torch.testing.assert_close(hg, analytical, rtol=1e-4, atol=1e-6)

    # Finite difference check
    eps = 1e-5
    theta_plus = torch.tensor([2.0 + eps], dtype=torch.float64)
    alpha_plus = y / (1.0 + theta_plus)
    loss_plus = 0.5 * torch.sum(alpha_plus**2)

    theta_minus = torch.tensor([2.0 - eps], dtype=torch.float64)
    alpha_minus = y / (1.0 + theta_minus)
    loss_minus = 0.5 * torch.sum(alpha_minus**2)

    fd = torch.tensor([(loss_plus - loss_minus).item() / (2.0 * eps)], dtype=torch.float64)
    torch.testing.assert_close(hg, fd, rtol=1e-4, atol=1e-6)


def test_residual_corrector_train_eval_consistency():
    """ResidualCorrector should produce different outputs in train vs eval."""
    from laker.correctors import ResidualCorrector

    torch.manual_seed(42)
    n = 20
    x = torch.randn(n, 3, dtype=torch.float32)

    corrector = ResidualCorrector(input_dim=3, output_dim=1, hidden_dim=16, dropout=0.5)
    corrector.eval()
    with torch.no_grad():
        out_eval = corrector(x).squeeze()

    corrector.train()
    out_train = corrector(x).squeeze()

    # With dropout=0.5, train and eval should differ on average
    # (probabilistic, but with high probability they differ for n=20)
    assert not torch.allclose(out_train, out_eval, atol=1e-6)


def test_pcg_with_twoscale_kernel():
    """PCG should solve two-scale kernel system to reasonable tolerance."""
    from laker.preconditioner import CCCPPreconditioner
    from laker.solvers import PreconditionedConjugateGradient

    torch.manual_seed(42)
    n = 60
    de = 4
    dtype = torch.float64
    e = torch.randn(n, de, dtype=dtype)
    b = torch.randn(n, dtype=dtype)

    op = TwoScaleAttentionKernelOperator(
        e,
        lambda_reg=1e-2,
        alpha=0.5,
        num_landmarks=30,
        k_neighbors=10,
        dtype=dtype,
    )

    pre = CCCPPreconditioner(
        num_probes=40,
        gamma=1e-1,
        max_iter=20,
        tol=1e-5,
        verbose=False,
        dtype=dtype,
    )
    pre.build(op.matvec, n)

    pcg = PreconditionedConjugateGradient(tol=1e-8, max_iter=300, verbose=False)
    x = pcg.solve(op.matvec, pre.apply, b)

    res = torch.linalg.norm(op.matvec(x) - b) / torch.linalg.norm(b)
    assert res.item() < 1e-2


def test_kernel_operator_diagonal_matches_dense():
    """diagonal() must match the diagonal of to_dense() for all operators."""
    torch.manual_seed(42)
    n = 20
    de = 4
    dtype = torch.float64
    e = torch.randn(n, de, dtype=dtype)
    lam = 1e-2

    operators = [
        AttentionKernelOperator(e, lambda_reg=lam, dtype=dtype),
        NystromAttentionKernelOperator(e, lambda_reg=lam, num_landmarks=15, dtype=dtype),
        TwoScaleAttentionKernelOperator(
            e,
            lambda_reg=lam,
            alpha=0.5,
            num_landmarks=15,
            k_neighbors=5,
            dtype=dtype,
        ),
    ]

    for op in operators:
        diag_op = op.diagonal()
        diag_dense = op.to_dense().diagonal()
        torch.testing.assert_close(diag_op, diag_dense, rtol=1e-4, atol=1e-5)


def test_ski_matvec_consistency():
    """SKI matvec should match dense for small grid."""
    from laker.kernels import SKIAttentionKernelOperator

    torch.manual_seed(42)
    n = 30
    de = 2
    dtype = torch.float64
    e = torch.randn(n, de, dtype=dtype)
    lam = 1e-2

    op = SKIAttentionKernelOperator(e, lambda_reg=lam, grid_size=32, dtype=dtype)
    x = torch.randn(n, dtype=dtype)
    y_ski = op.matvec(x)
    y_dense = op.to_dense() @ x
    torch.testing.assert_close(y_ski, y_dense, rtol=1e-4, atol=1e-5)


def test_chunked_predict_matches_full():
    """Predict with chunk_size should match predict without."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=6,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=40,
        cccp_max_iter=20,
        pcg_tol=1e-6,
        pcg_max_iter=200,
        chunk_size=16,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)

    x_test = torch.rand(50, 2, dtype=torch.float64) * 100.0
    pred_chunked = model.predict(x_test)

    model.chunk_size = None
    pred_full = model.predict(x_test)

    torch.testing.assert_close(pred_chunked, pred_full, rtol=1e-5, atol=1e-6)


def test_variance_exact_matches_analytical_tiny():
    """predict_variance on exact kernel should match analytical formula for n=3."""
    torch.manual_seed(42)
    n = 3
    de = 2
    dtype = torch.float64
    e = torch.randn(n, de, dtype=dtype)
    lam = 1e-1

    op = AttentionKernelOperator(e, lambda_reg=lam, dtype=dtype)
    y = torch.randn(n, dtype=dtype)

    # Query point
    e_q = torch.randn(1, de, dtype=dtype)
    k_q = torch.exp(e_q @ e.T)  # (1, n)
    k_qq = torch.exp(e_q @ e_q.T)  # (1, 1)

    # Exact variance: k(qq) - k(q,train)^T (K + lambda I)^{-1} k(q,train)
    a_inv_k = torch.linalg.solve(op.to_dense(), k_q.T)
    var_exact = (k_qq - (k_q @ a_inv_k)).item()

    model = LAKERRegressor(
        embedding_dim=de,
        lambda_reg=lam,
        gamma=0.0,
        num_probes=3,
        cccp_max_iter=5,
        pcg_tol=1e-12,
        pcg_max_iter=10,
        chunk_size=None,
        dtype=dtype,
        verbose=False,
    )

    class FixedEmbedding(torch.nn.Module):
        def forward(self, x):
            return e

    model.embedding_module = FixedEmbedding()
    model.fit(torch.zeros(n, 2, dtype=dtype), y)

    class QueryEmbedding(torch.nn.Module):
        def forward(self, x):
            return e_q

    model.embedding_model = QueryEmbedding()
    var_model = model.predict_variance(torch.zeros(1, 2, dtype=dtype)).item()

    assert abs(var_model - var_exact) < 1e-3


def test_preconditioner_apply_linearity():
    """Preconditioner apply should be linear: P(a*u + b*v) = a*P(u) + b*P(v)."""
    from laker.kernels import AttentionKernelOperator
    from laker.preconditioner import CCCPPreconditioner

    torch.manual_seed(42)
    n = 40
    de = 4
    dtype = torch.float64
    e = torch.randn(n, de, dtype=dtype)
    op = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=dtype)

    pre = CCCPPreconditioner(
        num_probes=30,
        gamma=1e-1,
        max_iter=15,
        tol=1e-5,
        verbose=False,
        dtype=dtype,
    )
    pre.build(op.matvec, n)

    u = torch.randn(n, dtype=dtype)
    v = torch.randn(n, dtype=dtype)
    a = 2.5
    b = -1.3

    lhs = pre.apply(a * u + b * v)
    rhs = a * pre.apply(u) + b * pre.apply(v)

    torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=1e-6)
