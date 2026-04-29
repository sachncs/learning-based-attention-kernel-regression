"""Basic usage of LAKER: reproduce the paper's n=3 example and a medium-scale experiment."""

import logging

import torch

from laker.models import LAKERRegressor

logging.basicConfig(level=logging.INFO)


def paper_n3_example():
    """Reproduce the worked example from Section IV-E of the paper."""
    print("=" * 60)
    print("Paper n=3 Example")
    print("=" * 60)

    # Embeddings and observations from Eq. (53)
    e = torch.tensor(
        [[0.241, 0.444], [-0.336, 0.112], [-0.220, 0.353]],
        dtype=torch.float64,
    )
    y = torch.tensor([-66.14, -65.77, -77.30], dtype=torch.float64)

    # Exact solution for verification
    g = torch.exp(e @ e.T)
    a_mat = g + 0.1 * torch.eye(3, dtype=torch.float64)
    alpha_exact = torch.linalg.solve(a_mat, y)
    print(f"Exact alpha:       {alpha_exact.tolist()}")

    class FixedEmbedding(torch.nn.Module):
        def forward(self, x):
            return e

    model = LAKERRegressor(
        embedding_dim=2,
        lambda_reg=0.1,
        gamma=0.0,
        num_probes=3,
        cccp_max_iter=10,
        pcg_tol=1e-12,
        pcg_max_iter=10,
        chunk_size=None,
        embedding_module=FixedEmbedding(),
        dtype=torch.float64,
    )
    x_dummy = torch.zeros(3, 2, dtype=torch.float64)
    model.fit(x_dummy, y)
    print(f"LAKER alpha:       {model.alpha.tolist()}")
    print(
        f"Relative error:    {torch.norm(model.alpha - alpha_exact) / torch.norm(alpha_exact):.3e}"
    )

    # Predict at query point from Eq. (57)
    e_star = torch.tensor([[0.051, 0.452]], dtype=torch.float64)
    k_star = torch.exp(e_star @ e.T)
    pred_exact = (k_star @ alpha_exact).item()

    class QueryEmbedding(torch.nn.Module):
        def forward(self, x):
            return e_star

    model.embedding_model = QueryEmbedding()
    pred_model = model.predict(torch.zeros(1, 2, dtype=torch.float64)).item()
    print(f"Exact prediction:  {pred_exact:.2f} dBm")
    print(f"LAKER prediction:  {pred_model:.2f} dBm")
    print("Ground truth:      -67.3 dBm")


def medium_scale_benchmark(n: int = 1000) -> None:
    """Run a medium-scale benchmark similar to Section V of the paper."""
    print("\n" + "=" * 60)
    print(f"Medium-scale benchmark (n={n})")
    print("=" * 60)

    torch.manual_seed(42)
    x_train = torch.rand(n, 2) * 100.0

    # Synthetic radio field: superposition of transmitters
    tx1 = torch.tensor([30.0, 70.0])
    tx2 = torch.tensor([70.0, 30.0])
    d1 = torch.norm(x_train - tx1, dim=1)
    d2 = torch.norm(x_train - tx2, dim=1)
    y_clean = -50.0 - 20.0 * torch.log10(d1 + 1.0) - 15.0 * torch.log10(d2 + 1.0)
    y_train = y_clean + torch.randn(n) * 1.5

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=None,  # adaptive
        cccp_max_iter=100,
        cccp_tol=1e-6,
        pcg_tol=1e-10,
        pcg_max_iter=1000,
        verbose=True,
    )
    model.fit(x_train, y_train)

    # Evaluate on a dense grid
    grid = torch.linspace(0, 100, 45)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    x_test = torch.stack([xx.ravel(), yy.ravel()], dim=1)
    y_pred = model.predict(x_test)

    rmse = torch.sqrt(
        torch.mean(
            (
                y_pred
                - (
                    -50.0
                    - 20.0 * torch.log10(torch.norm(x_test - tx1, dim=1) + 1.0)
                    - 15.0 * torch.log10(torch.norm(x_test - tx2, dim=1) + 1.0)
                )
            )
            ** 2
        )
    )
    print(f"Grid RMSE (approx): {rmse.item():.4f} dB")


if __name__ == "__main__":
    paper_n3_example()
    medium_scale_benchmark(n=200)
