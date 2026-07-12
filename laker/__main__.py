"""Command-line interface for LAKER.

The :mod:`laker` package can be invoked from a shell via the console script
entry-point declared in ``pyproject.toml`` (``laker = "laker.__main__:main"``).
The CLI exposes two subcommands:

* ``laker fit``      — load locations and measurements from disk, fit a
  :class:`~laker.models.LAKERRegressor`, and serialise the fitted model to
  a ``.pt`` file.
* ``laker predict``  — load a previously fitted model and emit predictions
  for a set of query locations.

Both subcommands accept tensor files in either PyTorch (``.pt``/``.pth``)
or NumPy (``.npy``) format; the loader dispatches on the file extension.
Logging is routed through Python's standard :mod:`logging` framework and
honours the ``--verbose`` / ``-v`` flag (``logging.DEBUG``) versus the
default ``logging.INFO`` level.

The CLI is intentionally thin: it is a thin convenience wrapper around the
:class:`~laker.models.LAKERRegressor` Python API. All algorithmic
behaviour (regularisation, kernel approximation, embedding, etc.) is
controlled by the underlying estimator and described there.
"""

import argparse
import logging
import sys

import numpy
import torch

from laker.models import LAKERRegressor

logger = logging.getLogger("laker")


def setup_logging(verbose: bool) -> None:
    """Configure root logger level for the CLI invocation.

    Args:
        verbose: If ``True``, set the root logger to ``logging.DEBUG``;
            otherwise to ``logging.INFO``. The configured format is
            ``"%(asctime)s [%(levelname)s] %(name)s: %(message)s"`` with
            a compact ``"%Y-%m-%d %H:%M:%S"`` timestamp.

    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_tensor(path: str) -> torch.Tensor:
    """Load a tensor from a ``.pt``/``.pth`` or ``.npy`` file.

    The format is inferred from the file extension. PyTorch files are
    loaded with ``weights_only=True`` for safety; NumPy files are wrapped
    in a ``torch.Tensor`` via :func:`torch.from_numpy` and therefore share
    memory with the underlying NumPy buffer.

    Args:
        path: Path to the tensor file. Must end in ``.pt``, ``.pth``, or
            ``.npy``.

    Returns:
        The loaded tensor.

    Raises:
        ValueError: If the file extension is not one of ``.pt``/``.pth``/
            ``.npy``.
        FileNotFoundError: If ``path`` does not exist.

    """
    if path.endswith(".npy"):
        return torch.from_numpy(numpy.load(path))
    if path.endswith(".pt") or path.endswith(".pth"):
        return torch.load(path, weights_only=True)
    raise ValueError(f"Unsupported file extension for {path}. " "Expected .pt, .pth, or .npy.")


def main() -> None:
    """Run the LAKER command-line interface.

    Parses command-line arguments, configures logging, and dispatches to
    :func:`cmd_fit` or :func:`cmd_predict` based on the chosen subcommand.
    If no subcommand is provided, the help text is printed and the
    process exits with a non-zero status.

    Returns:
        ``None``. Side effects only: parses ``sys.argv``, writes the
        configured log level, and dispatches to the subcommand handlers.

    """
    parser = argparse.ArgumentParser(
        description="LAKER: Learning-based Attention Kernel Regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from laker import __version__

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command")

    # Fit subcommand: mirrors LAKERRegressor's most commonly-tuned
    # hyperparameters. Defaults match the Python API defaults so users can
    # iteratively refine via the CLI without code changes.
    fit_parser = subparsers.add_parser("fit", help="Fit a LAKER model to data")
    fit_parser.add_argument("--locations", required=True, help="Path to locations .pt or .npy file")
    fit_parser.add_argument(
        "--measurements",
        required=True,
        help="Path to measurements .pt or .npy file",
    )
    fit_parser.add_argument("--output", required=True, help="Path to save fitted model")
    fit_parser.add_argument("--lambda-reg", type=float, default=1e-2, help="Regularisation lambda")
    fit_parser.add_argument("--gamma", type=float, default=1e-1, help="CCCP regularisation gamma")
    fit_parser.add_argument("--embedding-dim", type=int, default=10, help="Embedding dimension")
    fit_parser.add_argument("--num-probes", type=int, default=None, help="Number of random probes")
    fit_parser.add_argument("--device", default="cpu", help="torch device")
    # ``--dtype`` is restricted to the two float dtypes LAKER supports;
    # ``bfloat16`` is reserved for explicit embedding casts (see
    # ``LAKERRegressor.embedding_dtype``).
    fit_parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])

    # Predict subcommand: minimal surface area — just load the model and
    # query points, then emit a ``.pt`` tensor of predictions.
    pred_parser = subparsers.add_parser("predict", help="Predict using a fitted model")
    pred_parser.add_argument("--model", required=True, help="Path to fitted model .pt file")
    pred_parser.add_argument("--locations", required=True, help="Path to query locations")
    pred_parser.add_argument("--output", required=True, help="Path to save predictions")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "fit":
        cmd_fit(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_fit(args) -> None:
    """Handle the ``fit`` subcommand.

    Loads locations and measurements tensors from disk, instantiates a
    :class:`~laker.models.LAKERRegressor` with the parsed hyperparameters,
    fits it, and serialises the result to ``args.output``.

    Args:
        args: Parsed argparse namespace produced by ``main()``. The
            consumed attributes are ``locations``, ``measurements``,
            ``output``, ``lambda_reg``, ``gamma``, ``embedding_dim``,
            ``num_probes``, ``device``, and ``dtype``.

    Side effects:
        Writes a fitted-model ``.pt`` file to ``args.output`` and emits
        informational log messages via :data:`logger`.

    """
    logger.info("Loading data...")
    x = load_tensor(args.locations)
    y = load_tensor(args.measurements)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    model = LAKERRegressor(
        embedding_dim=args.embedding_dim,
        lambda_reg=args.lambda_reg,
        gamma=args.gamma,
        num_probes=args.num_probes,
        device=args.device,
        dtype=dtype,
        verbose=True,
    )
    model.fit(x, y)
    model.save(args.output)
    logger.info("Model saved to %s", args.output)


def cmd_predict(args) -> None:
    """Handle the ``predict`` subcommand.

    Loads a previously fitted model from ``args.model``, runs
    :meth:`~laker.models.LAKERRegressor.predict` on the query locations,
    and saves the resulting tensor to ``args.output``.

    Args:
        args: Parsed argparse namespace produced by ``main()``. The
            consumed attributes are ``model``, ``locations``, and
            ``output``.

    Side effects:
        Writes a predictions ``.pt`` file to ``args.output`` and emits
        informational log messages via :data:`logger`.

    """
    logger.info("Loading model...")
    model = LAKERRegressor.load(args.model)
    x = load_tensor(args.locations)
    predictions = model.predict(x)
    torch.save(predictions, args.output)
    logger.info("Predictions saved to %s", args.output)


if __name__ == "__main__":
    main()