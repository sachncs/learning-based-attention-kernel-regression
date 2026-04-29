"""Device and dtype management utilities."""

import logging
import os
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)

DEFAULT_DEVICE: torch.device = torch.device("cpu")
DEFAULT_DTYPE: torch.dtype = torch.float64


def get_default_device() -> torch.device:
    """Return the current default compute device."""
    return DEFAULT_DEVICE


def set_default_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Set and return the default compute device.

    If ``device`` is None, auto-select CUDA if available, else CPU.

    Args:
        device: Desired device string or torch.device instance.

    Returns:
        The resolved torch.device.
    """
    global DEFAULT_DEVICE
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    DEFAULT_DEVICE = device
    logger.info("Default device set to %s", device)
    return device


def get_default_dtype() -> torch.dtype:
    """Return the current default floating-point dtype."""
    return DEFAULT_DTYPE


def set_default_dtype(dtype: torch.dtype) -> None:
    """Set the default floating-point dtype."""
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = dtype
    logger.info("Default dtype set to %s", dtype)


def to_tensor(
    data,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Convert array-like data to a torch.Tensor on the requested device/dtype.

    Args:
        data: NumPy array, list, or existing torch.Tensor.
        device: Target device. Defaults to ``get_default_device()``.
        dtype: Target dtype. Defaults to ``get_default_dtype()``.

    Returns:
        A torch.Tensor with the specified device and dtype.
    """
    if device is None:
        device = get_default_device()
    if dtype is None:
        dtype = get_default_dtype()
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    return torch.as_tensor(data, device=device, dtype=dtype)


# Initialise from environment on import
_env_device = os.environ.get("LAKER_DEVICE")
if _env_device:
    set_default_device(_env_device)
_env_dtype = os.environ.get("LAKER_DTYPE")
if _env_dtype == "float32":
    set_default_dtype(torch.float32)
elif _env_dtype == "float64":
    set_default_dtype(torch.float64)
