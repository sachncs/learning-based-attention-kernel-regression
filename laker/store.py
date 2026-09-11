"""Model persistence: save and load LAKER models.

Public class :class:`Store` with two static methods :meth:`save` and
:meth:`load`. The serialized format is a single dictionary written with
:func:`torch.save` containing all hyperparameters, fitted tensors, and
neural-network state dicts (encoder and corrector).
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from laker.model import Laker

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


def _coerce_dtype(name: str) -> torch.dtype:
    """Map a serialised ``str(torch.dtype)`` back to a torch.dtype."""
    if name not in _DTYPE_MAP:
        raise ValueError(
            f"Store: unsupported dtype string {name!r}; expected one of {sorted(_DTYPE_MAP)}"
        )
    return _DTYPE_MAP[name]


class Store:
    """Save and load LAKER models."""

    @staticmethod
    def save(model: "Laker", path: str) -> None:
        """Serialise a fitted model to ``path``."""
        if model.coef is None:
            raise RuntimeError("Model has not been fitted. Call fit() before save().")
        state: dict[str, Any] = {
            "format": 2,
            "embed_dim": model.embed_dim,
            "lam": model.lam,
            "gamma": model.gamma,
            "num": model.num,
            "eps": model.eps,
            "base": model.base,
            "cccp_max": model.cccp_max,
            "cccp_tol": model.cccp_tol,
            "pcg_tol": model.pcg_tol,
            "pcg_max": model.pcg_max,
            "chunk": model.chunk,
            "kernel_type": model.kernel_type,
            "landmarks": model.landmarks,
            "features": model.features,
            "neighbors": model.neighbors,
            "grid_size": model.grid_size,
            "distributed": model.distributed,
            "blend": getattr(model, "blend", 0.5),
            "selection": getattr(model, "selection", "greedy"),
            "pilot": getattr(model, "pilot", 1000),
            "knots": getattr(model, "knots", 5),
            "prec_kind": getattr(model, "prec_kind", "cccp"),
            "device": (str(model.device) if model.device is not None else "cpu"),
            "dtype": str(model.dtype),
            "embed_dtype": (str(model.embed_dtype) if model.embed_dtype else None),
            "verbose": model.verbose,
            "embed": model.embed.cpu() if model.embed is not None else None,
            "coef": model.coef.cpu() if model.coef is not None else None,
            "x_train": (
                model.x_train.cpu() if getattr(model, "x_train", None) is not None else None
            ),
            "y_train": (
                model.y_train.cpu() if getattr(model, "y_train", None) is not None else None
            ),
        }
        if model.prec is not None:
            prec = model.prec
            state["precclass"] = prec.__class__.__name__
            state["precmodule"] = prec.__class__.__module__
            state["precstate"] = {k: v for k, v in vars(prec).items() if torch.is_tensor(v)}
        if model.encoder is not None:
            state["encoderstate"] = model.encoder.state_dict()
            state["encoderclass"] = model.encoder.__class__.__name__
            state["encodermodule"] = model.encoder.__class__.__module__
            if hasattr(model.encoder, "input_dim"):
                state["input_dim"] = model.encoder.input_dim
        if model.corrector is not None:
            state["corrector_state"] = model.corrector.state_dict()
            state["corrector_class"] = model.corrector.__class__.__name__
            state["corrector_module"] = model.corrector.__class__.__module__
        torch.save(state, path)

    @staticmethod
    def load(path: str) -> "Laker":
        """Deserialise a model from ``path``."""
        import os

        from laker.model import Laker

        if not os.path.exists(path):
            raise FileNotFoundError(f"Store.load: no such file {path!r}")
        state = torch.load(path, weights_only=True)
        if not isinstance(state, dict) or "format" not in state:
            raise ValueError(
                f"Store.load: {path!r} is not a LAKER model file (missing 'format' key)."
            )
        if state["format"] > 2:
            raise ValueError(
                f"Store.load: {path!r} was written by a newer LAKER version "
                f"(format={state['format']}); please upgrade the package."
            )
        for required in ("dtype", "lam", "embed_dim"):
            if required not in state:
                raise KeyError(
                    f"Store.load: {path!r} is missing required field {required!r}"
                )
        dtype = _coerce_dtype(state["dtype"])
        edt = state.get("embed_dtype")
        embed_dtype = _coerce_dtype(edt) if edt else None

        model = Laker(
            embed_dim=state["embed_dim"],
            lam=state["lam"],
            gamma=state["gamma"],
            num=state["num"],
            eps=state["eps"],
            base=state["base"],
            cccp_max=state["cccp_max"],
            cccp_tol=state["cccp_tol"],
            pcg_tol=state["pcg_tol"],
            pcg_max=state["pcg_max"],
            chunk=state.get("chunk"),
            kernel_type=state.get("kernel_type"),
            landmarks=state.get("landmarks"),
            features=state.get("features"),
            neighbors=state.get("neighbors"),
            grid_size=state.get("grid_size"),
            distributed=state.get("distributed", False),
            blend=state.get("blend", 0.5),
            selection=state.get("selection", "greedy"),
            pilot=state.get("pilot", 1000),
            knots=state.get("knots", 5),
            prec_kind=state.get("prec_kind", "cccp"),
            embed_dtype=embed_dtype,
            device=state["device"],
            dtype=dtype,
            verbose=state["verbose"],
        )

        if state.get("embed") is not None:
            model.embed = state["embed"].to(model.device)
        if state.get("coef") is not None:
            model.coef = state["coef"].to(model.device)
        if state.get("x_train") is not None:
            model.x_train = state["x_train"].to(model.device)
        if state.get("y_train") is not None:
            model.y_train = state["y_train"].to(model.device)

        if "encoderstate" in state:
            class_name = state["encoderclass"]
            module_name = state.get("encodermodule", "laker.embed")
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
            except (ImportError, AttributeError):
                logger.warning(
                    "Could not import %s.%s; falling back to embed.Position.",
                    module_name,
                    class_name,
                )
                from laker.embed import Position as cls

                class_name = "Position"

            input_dim = state.get("input_dim", 2)
            embed_dtype_v = embed_dtype if embed_dtype else dtype
            enc_cls: Callable[..., Any] = cls
            if class_name == "Position":
                model.encoder = enc_cls(
                    input_dim=input_dim,
                    dim=model.embed_dim,
                    device=model.device,
                    dtype=embed_dtype_v,
                )
            else:
                try:
                    model.encoder = enc_cls(
                        input_dim=input_dim,
                        dim=model.embed_dim,
                        device=model.device,
                        dtype=embed_dtype_v,
                    )
                except TypeError:
                    model.encoder = cls()
                    model.encoder.to(device=model.device, dtype=embed_dtype_v)
            model.encoder.load_state_dict(state["encoderstate"])

        if "corrector_state" in state:
            cn = state.get("corrector_class", "Corrector")
            mn = state.get("corrector_module", "laker.corrector")
            try:
                cm = importlib.import_module(mn)
                cc = getattr(cm, cn)
            except (ImportError, AttributeError):
                logger.warning("Could not import %s.%s; skipping corrector.", mn, cn)
                cc = None
            if cc is not None:
                input_dim = state.get("input_dim", 2)
                model.corrector = cc(
                    input_dim=input_dim, output_dim=1, hidden_dim=32, dropout=0.1
                ).to(device=model.device, dtype=dtype)
                model.corrector.load_state_dict(state["corrector_state"])

        from laker.kernel import (
            Exact,
            Fourier,
            Grid,
            Hybrid,
            Neighbors,
            Nystrom,
            Spectrum,
        )

        if model.kernel_type is None or model.kernel_type == "exact":
            model.kernel = Exact(
                embeddings=model.embed,
                lam=model.lam,
                chunk=model.chunk,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_type == "nystrom":
            model.kernel = Nystrom(
                embeddings=model.embed,
                lam=model.lam,
                num=model.landmarks,
                chunk=model.chunk,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_type == "fourier":
            model.kernel = Fourier(
                embeddings=model.embed,
                lam=model.lam,
                num=model.features,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_type == "neighbors":
            model.kernel = Neighbors(
                embeddings=model.embed,
                lam=model.lam,
                k=model.neighbors,
                chunk=model.chunk,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_type == "grid":
            model.kernel = Grid(
                embeddings=model.embed,
                lam=model.lam,
                grid_size=model.grid_size,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_type == "spectrum":
            model.kernel = Spectrum(
                embeddings=model.embed,
                lam=model.lam,
                knots=model.knots,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_type == "hybrid":
            model.kernel = Hybrid(
                embeddings=model.embed,
                lam=model.lam,
                alpha=model.blend,
                num=model.landmarks,
                k=model.neighbors,
                chunk=model.chunk,
                device=model.device,
                dtype=dtype,
            )

        if "precstate" in state and "precclass" in state and model.kernel is not None:
            try:
                pm = importlib.import_module(state["precmodule"])
                pc = getattr(pm, state["precclass"])
                prec = pc.__new__(pc)
                for key, value in state["precstate"].items():
                    if torch.is_tensor(value):
                        setattr(prec, key, value.to(model.device))
                    else:
                        setattr(prec, key, value)
                for attr in (
                    "gamma",
                    "eps",
                    "base",
                    "num",
                    "max_iter",
                    "tol",
                    "verbose",
                    "device",
                    "dtype",
                ):
                    if not hasattr(prec, attr) and hasattr(model, attr):
                        setattr(prec, attr, getattr(model, attr))
                if not hasattr(prec, "device"):
                    setattr(prec, "device", model.device)
                if not hasattr(prec, "dtype"):
                    setattr(prec, "dtype", dtype)
                model.prec = prec
            except Exception as exc:
                logger.warning(
                    "Could not restore preconditioner (%s); "
                    "variance() after load will fail until refit.",
                    exc,
                )
        return model


__all__ = ["Store"]
