"""Large-scale demonstration with chunked matrix-free evaluation."""

import logging
import time

import torch

from laker.models import LAKERRegressor

logging.basicConfig(level=logging.INFO)


def large_scale_demo(n: int = 5000) -> None:
    """Fit LAKER on a large synthetic dataset using chunked evaluation."""
    print("=" * 60)
    print(f"Large-scale demo (n={n})")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_train = torch.rand(n, 2, device=device) * 100.0
    # Synthetic field with multiple transmitters
    tx = torch.tensor([[25.0, 25.0], [75.0, 75.0], [50.0, 80.0]], device=device)
    y_clean = torch.zeros(n, device=device)
    for t in tx:
        d = torch.norm(x_train - t, dim=1)
        y_clean += -40.0 - 20.0 * torch.log10(d + 1.0)
    y_train = y_clean + torch.randn(n, device=device) * 1.5

    start = time.time()
    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=None,
        cccp_max_iter=100,
        cccp_tol=1e-6,
        pcg_tol=1e-10,
        pcg_max_iter=1000,
        chunk_size=1024,
        device=device,
        dtype=torch.float32,
        verbose=True,
    )
    model.fit(x_train, y_train)
    elapsed = time.time() - start
    print(f"Total fit time: {elapsed:.2f}s")

    # Quick prediction on a small test set
    x_test = torch.rand(100, 2, device=device) * 100.0
    y_pred = model.predict(x_test)
    print(f"Test predictions shape: {y_pred.shape}")
    print(f"Test predictions mean: {y_pred.mean().item():.2f} dB")


if __name__ == "__main__":
    large_scale_demo(n=5000)
