"""Command-line interface for LAKER."""

import argparse
import logging
import sys

import numpy
import torch

from laker.models import LAKERRegressor

logger = logging.getLogger("laker")


def setup_logging(verbose: bool) -> None:
    """Configure root logger level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_tensor(path: str) -> torch.Tensor:
    """Load a tensor from a ``.pt`` or ``.npy`` file.

    Args:
        path: File path.

    Returns:
        Loaded tensor.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if path.endswith(".npy"):
        return torch.from_numpy(numpy.load(path))
    if path.endswith(".pt") or path.endswith(".pth"):
        return torch.load(path, weights_only=True)
    raise ValueError(f"Unsupported file extension for {path}. " "Expected .pt, .pth, or .npy.")


def main() -> None:
    """Run the LAKER command-line interface."""
    parser = argparse.ArgumentParser(
        description="LAKER: Learning-based Attention Kernel Regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command")

    # Fit command
    fit_parser = subparsers.add_parser("fit", help="Fit a LAKER model to data")
    fit_parser.add_argument("--locations", required=True, help="Path to locations .pt or .npy file")
    fit_parser.add_argument(
        "--measurements", required=True, help="Path to measurements .pt or .npy file"
    )
    fit_parser.add_argument("--output", required=True, help="Path to save fitted model")
    fit_parser.add_argument("--lambda-reg", type=float, default=1e-2, help="Regularisation lambda")
    fit_parser.add_argument("--gamma", type=float, default=1e-1, help="CCCP regularisation gamma")
    fit_parser.add_argument("--embedding-dim", type=int, default=10, help="Embedding dimension")
    fit_parser.add_argument("--num-probes", type=int, default=None, help="Number of random probes")
    fit_parser.add_argument("--device", default="cpu", help="torch device")
    fit_parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])

    # Predict command
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

    Args:
        args: Parsed argparse namespace with fit parameters.
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

    Args:
        args: Parsed argparse namespace with predict parameters.
    """
    logger.info("Loading model...")
    model = LAKERRegressor.load(args.model)
    x = load_tensor(args.locations)
    predictions = model.predict(x)
    torch.save(predictions, args.output)
    logger.info("Predictions saved to %s", args.output)


if __name__ == "__main__":
    main()
