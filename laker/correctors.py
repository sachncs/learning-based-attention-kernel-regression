"""Residual corrector models for LAKER.

A residual corrector is a tiny auxiliary MLP that learns to predict the
residual :math:`y - \\hat{y}_{\\text{laker}}` from the raw spatial
coordinates (or, optionally, embeddings). Its prediction is added to the
base LAKER regression output to compensate for local model
misspecification without destabilising the underlying solver.

Two design choices are worth highlighting:

* The hidden activation is :math:`\\tanh`, which is smooth and bounded,
  matching the regularity of attention-kernel regression solutions.
* :class:`torch.nn.Dropout` is applied at the hidden layer so the
  corrector regularises the residual rather than memorising it.
  Training uses additional weight-decay L2 (see
  :meth:`laker.training.EmbeddingTrainer.fit_residual_corrector`).

The corrector is deliberately small (single hidden layer of width 32 by
default); LAKER already captures the dominant spatial structure and the
corrector only needs to model local defects.
"""

import torch
import torch.nn as nn


class ResidualCorrector(nn.Module):
    """Tiny MLP that predicts the residual ``y - y_hat_laker``.

    The corrector operates on raw spatial coordinates (or optionally on
    embedding vectors) and adds its output to the base LAKER
    prediction. It is trained with strong L2 regularisation and early
    stopping to avoid overfitting the residual.

    Architectural rationale:

    * **Tanh activation** — smooth and bounded so the corrector cannot
      produce wildly different outputs on nearby inputs.
    * **Dropout on the hidden layer** — regularises the residual
      estimate and prevents the network from memorising
      high-frequency noise.
    * **Single hidden layer** — LAKER captures the dominant spatial
      structure; the corrector only needs to model local defects.

    Args:
        input_dim: Dimensionality of the input features (e.g. spatial
            coordinates; pass ``embedding_dim`` if feeding embeddings).
        output_dim: Dimensionality of the target. Default ``1`` for
            scalar regression (the typical LAKER use-case).
        hidden_dim: Width of the single hidden layer. Default ``32``.
        dropout: Dropout probability on the hidden layer. Default
            ``0.1``. Set to ``0`` to disable dropout.

    Raises:
        ValueError: If any of ``input_dim``, ``output_dim``, or
            ``hidden_dim`` is not strictly positive (propagated from
            :class:`torch.nn.Linear`).

    Example:
        >>> corrector = ResidualCorrector(input_dim=2, output_dim=1)
        >>> x = torch.rand(100, 2)
        >>> residual = corrector(x)
        >>> residual.shape
        torch.Size([100, 1])

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        """Initialise the residual corrector network.

        Builds a two-layer MLP with Tanh activation and a single dropout
        layer between the hidden and output projections.

        Args:
            input_dim: Dimensionality of the input features.
            output_dim: Dimensionality of the target. Default ``1``.
            hidden_dim: Width of the single hidden layer. Default
                ``32``.
            dropout: Dropout probability on the hidden layer. Default
                ``0.1``.

        Side effects:
            Registers four sub-modules (``Linear``, ``Tanh``,
            ``Dropout``, ``Linear``) under ``self.net``.

        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict residual correction for inputs ``x``.

        Args:
            x: Tensor of shape ``(m, input_dim)``. The leading
                dimension ``m`` is the batch size (any number of
                points can be scored in one call).

        Returns:
            Tensor of shape ``(m, output_dim)`` containing the
            predicted residual. The caller adds this to the base
            LAKER prediction.

        """
        return self.net(x)