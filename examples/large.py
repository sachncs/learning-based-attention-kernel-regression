"""Large-scale demonstration with chunked matrix-free evaluation.

This module demonstrates LAKER on a large synthetic dataset
(:math:`n = 5000` by default) using chunked matrix-free kernel
evaluation to control memory usage.  The attention kernel matrix
:math:`K \\in \\mathbb{R}^{n \\times n}` is never fully materialised;
instead, :math:`Kx` is computed in chunks of ``chunk_size`` rows.

Run this module directly::

    python -m examples.large
"""

import logging
from typing import Optional

import torch

from examples.executor import ExampleExecutor
from laker.models import LAKERRegressor

logger = logging.getLogger(__name__)


class LargeScaleDemo:
    """Fit LAKER on a large synthetic dataset using chunked evaluation.

    Generates noisy radio-field observations from three transmitters on
    a :math:`100 \\times 100` domain and fits a
    :class:`~laker.models.LAKERRegressor` with ``chunk_size=1024`` to
    keep memory usage sub-quadratic.  Predictions are evaluated on a
    random test set.

    Args:
        executor: Optional :class:`~examples.executor.ExampleExecutor`
            for logging.  Defaults to a new instance with the label
            ``"Large-scale Demo"``.
    """

    def __init__(self, executor: Optional[ExampleExecutor] = None):
        self.executor = executor if executor is not None else ExampleExecutor("Large-scale Demo")

    @classmethod
    def run_default(cls, n: int = 5000) -> None:
        """Run the large-scale demo with default settings.

        Args:
            n: Number of training samples.
        """
        demo = cls()
        demo.run(n)

    def generate_synthetic_field(
        self, locations: torch.Tensor, transmitters: torch.Tensor
    ) -> torch.Tensor:
        """Generate a synthetic radio field from multiple transmitters.

        Computes a log-distance path-loss model for each transmitter
        and sums the contributions:

        .. math::

            s(\\mathbf{p}) = \\sum_{k} \\bigl(-40 - 20 \\log_{10}
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
            clean_signal += -40.0 - 20.0 * torch.log10(distances + 1.0)
        return clean_signal

    def run(self, n: int = 5000) -> None:
        """Run the large-scale demonstration.

        Generates ``n`` random training locations, computes noisy
        synthetic observations from three transmitters, fits a LAKER
        regressor with chunked kernel evaluation, and evaluates
        predictions on a random test set of 100 points.

        The model uses ``chunk_size=1024`` so that the full kernel
        matrix is never materialised, keeping memory at
        :math:`O(n \\cdot \\text{chunk\\_size})` per matvec.

        Args:
            n: Number of training samples.
        """
        self.executor.section(f"Large-scale demo (n={n})")

        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.executor.log_result("Device", str(device))

        train_locations = torch.rand(n, 2, device=device) * 100.0
        transmitters = torch.tensor([[25.0, 25.0], [75.0, 75.0], [50.0, 80.0]], device=device)
        clean_signal = self.generate_synthetic_field(train_locations, transmitters)
        train_observations = clean_signal + torch.randn(n, device=device) * 1.5

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
            dtype=torch.float64,
            verbose=True,
        )
        self.executor.time_operation(
            "LAKER fit", lambda: model.fit(train_locations, train_observations)
        )

        # Quick prediction on a small test set
        test_locations = torch.rand(100, 2, device=device) * 100.0
        predictions = model.predict(test_locations)
        self.executor.log_result("Test predictions shape", str(predictions.shape))
        self.executor.log_metric("Test predictions mean", predictions.mean().item(), fmt=".2f")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    LargeScaleDemo.run_default(n=5000)
