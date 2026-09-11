"""Public model facade.

The single top-level export is :class:`Laker`. It owns configuration
and fitted state directly, and composes the helpers from the rest of
the package (no legacy delegation).

Naming rules:

* Hyperparameters use single-word canonical names (``lam``, ``gamma``,
  ``num``, ...).
* Fitted state uses plain names without trailing underscores
  (``coef``, ``embed``, ``kernel``, ``prec``, ``encoder``, ``inputs``,
  ``targets``, ``iters``).
* All parameters are validated against the public parameter set.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from laker.backend import Backend
from laker.check import Check
from laker.core import Core
from laker.search import Search
from laker.store import Store
from laker.stream import Stream
from laker.train import Trainer

if TYPE_CHECKING:
    from laker.solve import Report

logger = logging.getLogger(__name__)


class Laker:
    """Sklearn-compatible LAKER estimator.

    Single public type for the LAKER pipeline. Wraps a stateless
    :class:`~laker.core.Core` that runs the embed → kernel → preconditioner
    → solve → predict stages.

    Args:
        embed_dim: Dimension of the embedding space.
        lam: Ridge weight ``lambda``.
        gamma: Kernel bandwidth for the CCCP preconditioner.
        kernel: ``"exact"``, ``"nystrom"``, ``"fourier"``, ``"neighbors"``,
            ``"grid"``, ``"spectrum"``, ``"hybrid"``.
        landmarks: Nyström landmark count.
        features: RFF feature count.
        neighbors: k-NN sparsity count.
        grid_size: SKI grid resolution.
        blend: Hybrid blend weight in ``[0, 1]``.
        selection: ``"greedy"`` or ``"leverage"`` landmark selection.
        pilot: Leverage-score pilot size.
        knots: Spectral kernel spline knots.
        distributed: Use multi-device distributed kernel.
        num: Random probe count for preconditioner construction.
        eps: Numerical stability constant.
        base: Base spectral norm bound for CCCP.
        prec_kind: ``"cccp"`` or ``"adaptive"``.
        cccp_max: Maximum CCCP iterations.
        cccp_tol: CCCP convergence tolerance.
        pcg_tol: PCG relative residual tolerance.
        pcg_max: Maximum PCG iterations.
        chunk: Tile size for chunked kernel evaluation.
        encoder: Optional pre-built embedding module.
        embed_dtype: Dtype for embedding computation.
        device: Target torch device.
        dtype: Floating-point dtype for the solver.
        verbose: Whether to log diagnostics.
        warm: Carry forward fitted state across ``fit`` calls.

    Attributes:
        coef: Solution vector, shape ``(n,)``.
        embed: Training embeddings, shape ``(n, D)``.
        kernel: Fitted kernel operator.
        prec: Fitted preconditioner.
        encoder: Fitted embedding module.
        inputs: Training locations, shape ``(n, d)``.
        targets: Training observations, shape ``(n,)``.
        iters: Iterations used by the last PCG solve.
        report: ``PCG.Report`` from the last fit's solve, ``None`` before fit.
    """

    PARAMS = (
        "embed_dim",
        "lam",
        "gamma",
        "kernel",
        "landmarks",
        "features",
        "neighbors",
        "grid_size",
        "blend",
        "selection",
        "pilot",
        "knots",
        "distributed",
        "num",
        "eps",
        "base",
        "prec_kind",
        "cccp_max",
        "cccp_tol",
        "pcg_tol",
        "pcg_max",
        "chunk",
        "encoder",
        "embed_dtype",
        "device",
        "dtype",
        "verbose",
        "warm",
    )

    def __init__(
        self,
        embed_dim: int = 10,
        lam: float = 1e-2,
        gamma: float = 1e-1,
        kernel_type: str = "exact",
        landmarks: Optional[int] = None,
        features: Optional[int] = None,
        neighbors: Optional[int] = None,
        grid_size: Optional[int] = None,
        blend: float = 0.5,
        selection: str = "greedy",
        pilot: int = 1000,
        knots: int = 5,
        distributed: bool = False,
        num: Optional[int] = None,
        eps: float = 1e-8,
        base: float = 0.05,
        prec_kind: str = "cccp",
        cccp_max: int = 200,
        cccp_tol: float = 1e-6,
        pcg_tol: float = 1e-6,
        pcg_max: int = 1000,
        chunk: Optional[int] = None,
        encoder: Optional[nn.Module] = None,
        embed_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = os.environ.get("LAKER_VERBOSE", "1") == "1",
        warm: bool = False,
    ) -> None:
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if not 0 <= base <= 1:
            raise ValueError(f"base must be in [0, 1], got {base}")
        if cccp_max <= 0:
            raise ValueError(f"cccp_max must be positive, got {cccp_max}")
        if cccp_tol <= 0:
            raise ValueError(f"cccp_tol must be positive, got {cccp_tol}")
        if pcg_tol <= 0:
            raise ValueError(f"pcg_tol must be positive, got {pcg_tol}")
        if pcg_max <= 0:
            raise ValueError(f"pcg_max must be positive, got {pcg_max}")
        if neighbors is not None and neighbors <= 0:
            raise ValueError(f"neighbors must be positive, got {neighbors}")
        if grid_size is not None and grid_size < 2:
            raise ValueError(f"grid_size must be at least 2, got {grid_size}")
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"blend must be in [0, 1], got {blend}")
        if selection not in ("greedy", "leverage"):
            raise ValueError(f"selection must be 'greedy' or 'leverage', got {selection!r}")
        if pilot <= 0:
            raise ValueError(f"pilot must be positive, got {pilot}")
        if knots <= 0:
            raise ValueError(f"knots must be positive, got {knots}")
        if prec_kind not in ("cccp", "adaptive"):
            raise ValueError(f"prec_kind must be 'cccp' or 'adaptive', got {prec_kind!r}")
        if kernel_type not in (
            "exact",
            "nystrom",
            "fourier",
            "neighbors",
            "grid",
            "spectrum",
            "hybrid",
        ):
            raise ValueError(
                "kernel_type must be one of 'exact', 'nystrom', 'fourier', "
                "'neighbors', 'grid', 'spectrum', 'hybrid'; "
                f"got {kernel_type!r}"
            )

        self.embed_dim = embed_dim
        self.lam = lam
        self.gamma = gamma
        self.kernel_type = kernel_type
        self.landmarks = landmarks
        self.features = features
        self.neighbors = neighbors
        self.grid_size = grid_size
        self.blend = blend
        self.selection = selection
        self.pilot = pilot
        self.knots = knots
        self.distributed = distributed
        self.num = num
        self.eps = eps
        self.base = base
        self.prec_kind = prec_kind
        self.cccp_max = cccp_max
        self.cccp_tol = cccp_tol
        self.pcg_tol = pcg_tol
        self.pcg_max = pcg_max
        self.chunk = chunk
        self.embed_dtype = embed_dtype
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.warm = warm
        self.init_encoder = encoder

        self.core = Core(
            embed_dim=embed_dim,
            lam=lam,
            gamma=gamma,
            num=num,
            eps=eps,
            base=base,
            cccp_max=cccp_max,
            cccp_tol=cccp_tol,
            pcg_tol=pcg_tol,
            pcg_max=pcg_max,
            chunk=chunk,
            encoder=encoder,
            kernel_type=kernel_type,
            landmarks=landmarks,
            features=features,
            neighbors=neighbors,
            grid_size=grid_size,
            distributed=distributed,
            blend=blend,
            selection=selection,
            pilot=pilot,
            knots=knots,
            prec_kind=prec_kind,
            embed_dtype=embed_dtype,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )
        self.stream = Stream(self.core)
        self.searcher = Search(self.core)
        self.train = Trainer(self.core)

        self.coef: Optional[torch.Tensor] = None
        self.embed: Optional[torch.Tensor] = None
        self.kernel: Any = None
        self.prec: Any = None
        self.encoder: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.iters: Optional[int] = None
        self.report: Optional["Report"] = None
        self.corrector: Optional[nn.Module] = None
        self.x_train: Optional[torch.Tensor] = None
        self.y_train: Optional[torch.Tensor] = None
        self.partial_count: int = 0
        self.regpath: Optional[dict] = None

    def __repr__(self) -> str:
        fitted = "fitted" if self.coef is not None else "not fitted"
        return f"Laker(embed_dim={self.embed_dim}, lam={self.lam}, {fitted})"

    # ------------------------------------------------------------------
    # Hyperparameter contract
    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "lam": self.lam,
            "gamma": self.gamma,
            "kernel_type": self.kernel_type,
            "landmarks": self.landmarks,
            "features": self.features,
            "neighbors": self.neighbors,
            "grid_size": self.grid_size,
            "blend": self.blend,
            "selection": self.selection,
            "pilot": self.pilot,
            "knots": self.knots,
            "distributed": self.distributed,
            "num": self.num,
            "eps": self.eps,
            "base": self.base,
            "prec_kind": self.prec_kind,
            "cccp_max": self.cccp_max,
            "cccp_tol": self.cccp_tol,
            "pcg_tol": self.pcg_tol,
            "pcg_max": self.pcg_max,
            "chunk": self.chunk,
            "encoder": getattr(self, "init_encoder", None),
            "embed_dtype": self.embed_dtype,
            "device": self.device,
            "dtype": self.dtype,
            "verbose": self.verbose,
            "warm": self.warm,
        }

    def set_params(self, **params: Any) -> "Laker":
        unknown = set(params) - set(self.PARAMS)
        if unknown:
            raise ValueError(
                f"Invalid parameter(s) for Laker: {sorted(unknown)}. Valid: {list(self.PARAMS)}."
            )
        coerced = dict(params)
        if isinstance(coerced.get("device"), str):
            coerced["device"] = torch.device(coerced["device"])
        for name in ("dtype", "embed_dtype"):
            v = coerced.get(name)
            if isinstance(v, str):
                if "float64" in v:
                    coerced[name] = torch.float64
                elif "bfloat16" in v:
                    coerced[name] = torch.bfloat16
                elif "float16" in v:
                    coerced[name] = torch.float16
                else:
                    coerced[name] = torch.float32
        for key, value in coerced.items():
            setattr(self, key, value)
        self.core = Core(
            embed_dim=self.embed_dim,
            lam=self.lam,
            gamma=self.gamma,
            num=self.num,
            eps=self.eps,
            base=self.base,
            cccp_max=self.cccp_max,
            cccp_tol=self.cccp_tol,
            pcg_tol=self.pcg_tol,
            pcg_max=self.pcg_max,
            chunk=self.chunk,
            encoder=self.init_encoder,
            kernel_type=self.kernel_type,
            landmarks=self.landmarks,
            features=self.features,
            neighbors=self.neighbors,
            grid_size=self.grid_size,
            distributed=self.distributed,
            blend=self.blend,
            selection=self.selection,
            pilot=self.pilot,
            knots=self.knots,
            prec_kind=self.prec_kind,
            embed_dtype=self.embed_dtype,
            device=self.device,
            dtype=self.dtype,
            verbose=self.verbose,
        )
        self.stream = Stream(self.core)
        self.searcher = Search(self.core)
        self.train = Trainer(self.core)
        return self

    def __sklearn_clone__(self) -> "Laker":
        return Laker(**self.get_params())

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------
    def fit(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        x0: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> "Laker":
        device = self.core.device
        dtype = self.core.dtype
        x = Check.x(Backend.tensor(x, device=device, dtype=dtype))
        y = Check.y(Backend.tensor(y, device=device, dtype=dtype))

        if not self.warm and self.coef is not None:
            params = self.get_params()
            fresh = Laker(**params)
            return fresh.fit(x, y, x0=x0, seed=seed)

        embed, enc = self.core.embed(x)
        kernel_op = self.core.build_kernel(embed)
        prec = self.core.build_prec(
            kernel_op.matvec, embed.shape[0], diag=kernel_op.diag(), seed=seed
        )
        coef, iters = self.core.solve(kernel_op, prec, y, x0=x0)

        self.embed = embed
        self.kernel = kernel_op
        self.prec = prec
        self.coef = coef
        self.encoder = enc
        self.inputs = x
        self.targets = y
        self.iters = iters
        self.report = getattr(self.core, "last_report", None)
        self.x_train = x
        self.y_train = y
        self.partial_count = 0
        return self

    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if self.coef is None or self.embed is None or self.encoder is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        x = Check.x(Backend.tensor(x, device=self.core.device, dtype=self.core.dtype))
        if hasattr(self.encoder, "input_dim") and x.shape[1] != self.encoder.input_dim:
            raise ValueError(
                f"x has {x.shape[1]} features but model expects {self.encoder.input_dim}"
            )
        return self.core.predict(
            x, self.encoder, self.embed, self.kernel, self.coef, self.corrector
        )

    def variance(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if self.coef is None or self.embed is None or self.prec is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        x = Backend.tensor(x, device=self.core.device, dtype=self.core.dtype)
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        return self.core.variance(
            x, self.encoder, self.embed, self.kernel, self.prec, self.coef, self.lam
        )

    def score(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """Coefficient of determination ``R^2``."""
        y_true = Check.y(Backend.tensor(y, device=self.core.device, dtype=self.core.dtype))
        y_pred = self.predict(x)
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"shape mismatch: predict={tuple(y_pred.shape)} vs y={tuple(y_true.shape)}"
            )
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

    def condition(self) -> float:
        if self.prec is None or self.kernel is None:
            raise RuntimeError("Model has not been fitted.")
        return self.core.condition(self.kernel, self.prec)

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------
    def search(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        val: float = 0.2,
        lam_grid: Optional[list[float]] = None,
        gamma_grid: Optional[list[float]] = None,
        num_grid: Optional[list[int]] = None,
        warm: bool = True,
        seed: Optional[int] = None,
    ) -> "Laker":
        return self.searcher.grid(
            self,
            x,
            y,
            val=val,
            lam_grid=lam_grid,
            gamma_grid=gamma_grid,
            num_grid=num_grid,
            warm=warm,
            seed=seed,
        )

    def bayes(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        val: float = 0.2,
        n_calls: int = 15,
        n_init: int = 5,
        lam_bounds: tuple[float, float] = (1e-4, 1.0),
        gamma_bounds: tuple[float, float] = (0.0, 2.0),
        num_bounds: tuple[int, int] = (20, 300),
        seed: Optional[int] = None,
    ) -> "Laker":
        return self.searcher.bayes(
            self,
            x,
            y,
            val=val,
            n_calls=n_calls,
            n_init=n_init,
            lam_bounds=lam_bounds,
            gamma_bounds=gamma_bounds,
            num_bounds=num_bounds,
            seed=seed,
        )

    def update(
        self,
        x_new: Union[torch.Tensor, np.ndarray],
        y_new: Union[torch.Tensor, np.ndarray],
        forget: float = 1.0,
        threshold: int = 100,
        seed: Optional[int] = None,
        autofit: bool = True,
    ) -> "Laker":
        return self.stream.update(
            self,
            x_new,
            y_new,
            forget=forget,
            threshold=threshold,
            seed=seed,
            autofit=autofit,
        )

    def path(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        grid: list[float],
        reuse: bool = True,
    ) -> dict:
        return self.stream.path(self, x, y, grid=grid, reuse=reuse)

    def continuation(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        stages: int = 5,
        reuse: bool = True,
    ) -> "Laker":
        return self.stream.continuation(self, x, y, lo=lo, hi=hi, stages=stages, reuse=reuse)

    def learn(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        lr: float = 1e-3,
        epochs: int = 50,
        rebuild: int = 10,
        patience: int = 5,
    ) -> "Laker":
        x = Check.x(Backend.tensor(x, device=self.core.device, dtype=self.core.dtype))
        y = Check.y(Backend.tensor(y, device=self.core.device, dtype=self.core.dtype))
        return self.train.learn(
            self, x, y, lr=lr, epochs=epochs, rebuild=rebuild, patience=patience
        )

    def correct(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        val: float = 0.2,
        epochs: int = 200,
        patience: int = 10,
        weight_decay: float = 1e-2,
        lr: float = 1e-3,
        seed: Optional[int] = None,
    ) -> "Laker":
        x = Check.x(Backend.tensor(x, device=self.core.device, dtype=self.core.dtype))
        y = Check.y(Backend.tensor(y, device=self.core.device, dtype=self.core.dtype))
        return self.train.correct(
            self,
            x,
            y,
            val=val,
            epochs=epochs,
            patience=patience,
            weight_decay=weight_decay,
            lr=lr,
            seed=seed,
        )

    def bilevel(
        self,
        x_train: Union[torch.Tensor, np.ndarray],
        y_train: Union[torch.Tensor, np.ndarray],
        x_val: Union[torch.Tensor, np.ndarray],
        y_val: Union[torch.Tensor, np.ndarray],
        lr: float = 1e-3,
        epochs: int = 20,
        patience: int = 5,
    ) -> "Laker":
        x_train = Check.x(Backend.tensor(x_train, device=self.core.device, dtype=self.core.dtype))
        y_train = Check.y(Backend.tensor(y_train, device=self.core.device, dtype=self.core.dtype))
        x_val = Check.x(Backend.tensor(x_val, device=self.core.device, dtype=self.core.dtype))
        y_val = Check.y(Backend.tensor(y_val, device=self.core.device, dtype=self.core.dtype))
        return self.train.bilevel(
            self, x_train, y_train, x_val, y_val, lr=lr, epochs=epochs, patience=patience
        )

    def calibrate(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        lr: float = 1e-3,
        epochs: int = 50,
        beta: float = 0.1,
        subset: float = 0.2,
        patience: int = 5,
        seed: Optional[int] = None,
    ) -> "Laker":
        x = Check.x(Backend.tensor(x, device=self.core.device, dtype=self.core.dtype))
        y = Check.y(Backend.tensor(y, device=self.core.device, dtype=self.core.dtype))
        return self.train.calibrate(
            self,
            x,
            y,
            lr=lr,
            epochs=epochs,
            beta=beta,
            subset=subset,
            patience=patience,
            seed=seed,
        )

    def tune(
        self,
        x_train: Union[torch.Tensor, np.ndarray],
        y_train: Union[torch.Tensor, np.ndarray],
        x_val: Union[torch.Tensor, np.ndarray],
        y_val: Union[torch.Tensor, np.ndarray],
        lr: float = 1e-3,
        epochs: int = 20,
        patience: int = 5,
    ) -> "Laker":
        return self.bilevel(x_train, y_train, x_val, y_val, lr=lr, epochs=epochs, patience=patience)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        Store.save(self, path)

    @classmethod
    def load(cls, path: str) -> "Laker":
        return Store.load(path)


__all__ = ["Laker"]
