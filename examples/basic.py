"""Basic usage of LAKER: reproduce the paper's n=3 example and a medium-scale experiment.

This module contains two self-contained examples:

1. :class:`PaperExample` — reproduces the worked example from
   Section IV-E of the LAKER paper.  It constructs a :math:`3 \\times 3`
   kernel system :math:`(K + \\lambda I)\\alpha = y` from three
   hand-crafted embeddings, solves for :math:`\\alpha` with the LAKER
   pipeline, and verifies the result against a direct
   :func:`torch.linalg.solve` reference.  A prediction at a query
   point is also verified.

2. :class:`MediumScaleBenchmark` — generates synthetic radio-field
   measurements from two transmitters in a :math:`100 \\times 100`
   domain and fits a :class:`~laker.models.LAKERRegressor` to the
   noisy observations, reporting grid RMSE.

Run this module directly to execute both examples::

    python -m examples.basic
"""

import logging
from typing import Optional

import torch

from examples.executor import ExampleExecutor
from laker.models import LAKERRegressor

logger = logging.getLogger(__name__)


class PaperExample:
    """Reproduce the worked example from Section IV-E of the paper.

    Constructs three embeddings and observations from Eq. (53), fits a
    :class:`~laker.models.LAKERRegressor` with a fixed embedding
    module, and compares the resulting coefficient vector
    :math:`\\alpha` against the direct solve
    :math:`\\alpha = (K + \\lambda I)^{-1} y`.

    Args:
        executor: Optional :class:`~examples.executor.ExampleExecutor`
            for logging.  Defaults to a new instance with the label
            ``"Paper n=3 Example"``.
    """

    def __init__(self, executor: Optional[ExampleExecutor] = None):
        self.executor = executor if executor is not None else ExampleExecutor("Paper n=3 Example")

    @classmethod
    def run_default(cls) -> None:
        """Run the paper example with default settings.

        Convenience class method that creates a :class:`PaperExample`
        instance and calls :meth:`run`.
        """
        example = cls()
        example.run()

    def run(self) -> None:
        """Run the paper example and log results.

        Executes the following steps:

        1. Constructs embeddings and observations from Eq. (53).
        2. Solves the kernel system directly via
           :func:`torch.linalg.solve` for reference.
        3. Fits a :class:`~laker.models.LAKERRegressor` with a fixed
           embedding module.
        4. Computes relative error :math:`\\|\\alpha - \\alpha_*\\| /
           \\|\\alpha_*\\|`.
        5. Predicts at the query point from Eq. (57) and compares
           against the reference prediction.
        """
        self.executor.section("Paper n=3 Example")

        # Embeddings and observations from Eq. (53)
        embeddings = torch.tensor(
            [[0.241, 0.444], [-0.336, 0.112], [-0.220, 0.353]],
            dtype=torch.float64,
        )
        observations = torch.tensor([-66.14, -65.77, -77.30], dtype=torch.float64)

        # Exact solution for verification
        gram = torch.exp(embeddings @ embeddings.T)
        system_matrix = gram + 0.1 * torch.eye(3, dtype=torch.float64)
        alpha_exact = torch.linalg.solve(system_matrix, observations)
        self.executor.log_result("Exact alpha", alpha_exact.tolist())

        class FixedEmbedding(torch.nn.Module):
            def forward(self, x):
                return embeddings

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
        dummy_locations = torch.zeros(3, 2, dtype=torch.float64)
        model.fit(dummy_locations, observations)
        self.executor.log_result("LAKER alpha", model.alpha.tolist())

        relative_error = torch.norm(model.alpha - alpha_exact) / torch.norm(alpha_exact)
        self.executor.log_metric("Relative error", relative_error.item(), fmt=".3e")

        # Predict at query point from Eq. (57)
        query_embedding = torch.tensor([[0.051, 0.452]], dtype=torch.float64)
        kernel_query = torch.exp(query_embedding @ embeddings.T)
        prediction_exact = (kernel_query @ alpha_exact).item()

        class QueryEmbedding(torch.nn.Module):
            def forward(self, x):
                return query_embedding

        model.embedding_model = QueryEmbedding()
        prediction_model = model.predict(torch.zeros(1, 2, dtype=torch.float64)).item()
        self.executor.log_metric("Exact prediction", prediction_exact, fmt=".2f")
        self.executor.log_metric("LAKER prediction", prediction_model, fmt=".2f")
        self.executor.log_result("Ground truth", "-67.3 dBm")


class MediumScaleBenchmark:
    """Run a medium-scale benchmark similar to Section V of the paper.

    Generates synthetic radio-field measurements from two transmitters
    on a :math:`100 \\times 100` domain, adds Gaussian noise, and fits
    a :class:`~laker.models.LAKERRegressor` to the data.  Grid RMSE
    is reported on a dense evaluation grid.

    Args:
        executor: Optional :class:`~examples.executor.ExampleExecutor`
            for logging.  Defaults to a new instance with the label
            ``"Medium-scale Benchmark"``.
    """

    def __init__(self, executor: Optional[ExampleExecutor] = None):
        self.executor = (
            executor if executor is not None else ExampleExecutor("Medium-scale Benchmark")
        )

    @classmethod
    def run_default(cls, n: int = 1000) -> None:
        """Run the medium-scale benchmark with default settings.

        Args:
            n: Number of training samples.
        """
        benchmark = cls()
        benchmark.run(n)

    def generate_synthetic_field(
        self, locations: torch.Tensor, transmitters: torch.Tensor
    ) -> torch.Tensor:
        """Generate a synthetic radio field from multiple transmitters.

        Computes a log-distance path-loss model for each transmitter
        and sums the contributions:

        .. math::

            s(\\mathbf{p}) = \\sum_{k} \\bigl(-50 - 20 \\log_{10}
            (\\|\\mathbf{p} - \\mathbf{t}_k\\| + 1)\\bigr)

        Args:
            locations: Tensor of shape ``(m, 2)`` with 2-D coordinates.
            transmitters: Tensor of shape ``(T, 2)`` with transmitter
                positions.

        Returns:
            Tensor of shape ``(m,)`` with the combined signal strength.
        """
        clean_signal = torch.zeros(locations.shape[0], device=locations.device)
        for transmitter in transmitters:
            distances = torch.norm(locations - transmitter, dim=1)
            clean_signal += -50.0 - 20.0 * torch.log10(distances + 1.0)
        return clean_signal

    def run(self, n: int = 1000) -> None:
        """Run the medium-scale benchmark.

        Generates ``n`` random training locations, computes noisy
        synthetic observations, fits a LAKER regressor, and evaluates
        predictions on a :math:`45 \\times 45` grid.

        Args:
            n: Number of training samples.
        """
        self.executor.section(f"Medium-scale benchmark (n={n})")

        torch.manual_seed(42)
        train_locations = torch.rand(n, 2) * 100.0

        transmitters = torch.tensor([[30.0, 70.0], [70.0, 30.0]])
        clean_signal = self.generate_synthetic_field(train_locations, transmitters)
        train_observations = clean_signal + torch.randn(n) * 1.5

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
        self.executor.time_operation(
            "LAKER fit", lambda: model.fit(train_locations, train_observations)
        )

        # Evaluate on a dense grid
        grid = torch.linspace(0, 100, 45)
        grid_x, grid_y = torch.meshgrid(grid, grid, indexing="ij")
        test_locations = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=1)
        predictions = model.predict(test_locations)

        ground_truth = self.generate_synthetic_field(test_locations, transmitters)
        rmse = torch.sqrt(torch.mean((predictions - ground_truth) ** 2))
        self.executor.log_metric("Grid RMSE (approx)", rmse.item(), fmt=".4f")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    PaperExample.run_default()
    MediumScaleBenchmark.run_default(n=200)
