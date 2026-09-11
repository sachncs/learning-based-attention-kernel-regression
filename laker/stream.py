"""Streaming updates, regularisation paths, and continuation schedules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from laker.core import Core
    from laker.model import Laker

logger = logging.getLogger(__name__)


class Stream:
    """Incremental updates and continuation-path fitting for a fitted model."""

    def __init__(self, core: "Core") -> None:
        self.core = core

    def update(
        self,
        model: "Laker",
        x_new: torch.Tensor,
        y_new: torch.Tensor,
        forget: float = 1.0,
        threshold: int = 100,
        seed: Optional[int] = None,
        autofit: bool = True,
    ) -> "Laker":
        """Append new observations and re-solve with a warm start.

        Args:
            model: Fitted :class:`Laker`.
            x_new: New inputs of shape ``(m, d)``.
            y_new: New targets of shape ``(m,)``.
            forget: Scalar in ``[0, 1]`` scaling the previous ``alpha``.
            threshold: Max cumulative new points before triggering a refit.
            seed: Optional seed for the preconditioner's random probes.
            autofit: When ``True`` (default) and the threshold is exceeded,
                automatically concatenate all known data and refit. When
                ``False``, raise ``RuntimeError`` instead.
        """
        from laker.backend import Backend
        from laker.check import Check
        from laker.math import Math

        if model.coef is None or model.embed is None:
            raise RuntimeError("Model has not been fitted. Call fit() before update().")

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            Math.seed_set(int(seed))
            Backend.seed(int(seed))

        x_new = Check.x(Check.tensor(x_new, device=model.device, dtype=model.dtype), "x_new")
        y_new = Check.y(Check.tensor(y_new, device=model.device, dtype=model.dtype), "y_new")

        m = x_new.shape[0]
        total = model.partial_count + m

        if total >= threshold:
            if not autofit:
                model.partial_count = 0
                raise RuntimeError(
                    "update threshold exceeded. Concatenate all data and call fit() "
                    "for a full refit."
                )
            all_x = (
                torch.cat([model.x_train, x_new], dim=0)
                if model.x_train is not None
                else x_new
            )
            all_y = (
                torch.cat([model.y_train, y_new])
                if model.y_train is not None
                else y_new
            )
            warm_was = model.warm
            model.warm = True
            try:
                model.fit(all_x, all_y, seed=seed)
            finally:
                model.warm = warm_was
            return model

        model.partial_count = total

        new_emb = model.encoder(x_new.to(dtype=self.core.embed_dtype))
        if self.core.embed_dtype != self.core.dtype:
            new_emb = new_emb.to(dtype=self.core.dtype)

        old_n = model.embed.shape[0]
        model.embed = torch.cat([model.embed, new_emb], dim=0)

        kernel = self.core.build_kernel(model.embed, lam=model.lam)
        model.kernel = kernel

        old_alpha = model.coef * forget
        y_old = model.y_train
        if y_old is None:
            y_old = torch.zeros(old_n, device=self.core.device, dtype=self.core.dtype)
        y_ext = torch.cat([y_old, y_new])

        x0 = torch.cat([old_alpha, torch.zeros(m, device=self.core.device, dtype=self.core.dtype)])

        with torch.no_grad():
            prec = self.core.build_prec(
                kernel.matvec,
                model.embed.shape[0],
                diag=kernel.diag(),
                seed=seed,
            )
            model.prec = prec
            model.coef, model.iters = self.core.solve(kernel, prec, y_ext, x0=x0)
        model.y_train = y_ext
        if model.x_train is not None:
            model.x_train = torch.cat([model.x_train, x_new], dim=0)
        else:
            model.x_train = x_new

        if self.core.verbose:
            logger.info(
                "update: added %d points, total=%d, iters=%d",
                m,
                model.embed.shape[0],
                model.iters,
            )
        return model

    def path(
        self,
        model: "Laker",
        x: torch.Tensor,
        y: torch.Tensor,
        grid: list[float],
        reuse: bool = True,
    ) -> dict:
        """Fit a regularisation path over a sequence of ``lam`` values.

        Each solve warm-starts from the previous one, sorted from
        largest to smallest ``lam``.

        Returns a dict with ``"lam"``, ``"coef"``, ``"iters"``, ``"rel"``.
        """
        from laker.check import Check

        x = Check.x(Check.tensor(x, device=model.device, dtype=model.dtype))
        y = Check.y(Check.tensor(y, device=model.device, dtype=model.dtype))
        if not grid:
            raise ValueError("grid must not be empty")

        embed, enc = self.core.embed(x)
        n = embed.shape[0]
        sorted_grid = sorted(grid, reverse=True)

        coefs, iterslist, rels = [], [], []
        x0 = None
        prec = None

        for lam_value in sorted_grid:
            kernel = self.core.build_kernel(embed, lam=lam_value)
            if prec is None or not reuse:
                prec = self.core.build_prec(
                    kernel.matvec,
                    n,
                    gamma=self.core.gamma,
                    num=self.core.num,
                    diag=kernel.diag(),
                )
            coef, it = self.core.solve(kernel, prec, y, x0=x0)
            coefs.append(coef)
            iterslist.append(it)
            rels.append(
                torch.linalg.norm(kernel.matvec(coef) - y).item() / torch.linalg.norm(y).item()
            )
            x0 = coef.clone()

        model.embed = embed
        model.encoder = enc
        kernel = self.core.build_kernel(embed, lam=sorted_grid[-1])
        model.kernel = kernel
        model.prec = self.core.build_prec(
            kernel.matvec,
            n,
            gamma=self.core.gamma,
            num=self.core.num,
            diag=kernel.diag(),
        )
        model.coef = coefs[-1]
        model.iters = iterslist[-1]
        model.y_train = y
        path = {"lam": sorted_grid, "coef": coefs, "iters": iterslist, "rel": rels}
        model.regpath = path
        return path

    def continuation(
        self,
        model: "Laker",
        x: torch.Tensor,
        y: torch.Tensor,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        stages: int = 5,
        reuse: bool = True,
    ) -> "Laker":
        """Geometric schedule of ``lam`` values from ``hi`` down to ``lo``."""
        if hi is None:
            hi = 10.0 * self.core.lam
        if lo is None:
            lo = self.core.lam
        if stages < 1:
            raise ValueError(f"stages must be positive, got {stages}")
        if hi <= 0 or lo <= 0:
            raise ValueError("lo and hi must be positive")

        ratio = (lo / hi) ** (1.0 / max(1, stages - 1))
        schedule = [hi * (ratio**k) for k in range(stages)]
        schedule[-1] = lo

        path = self.path(model, x, y, grid=schedule, reuse=reuse)
        model.lam = float(lo)
        model.iters = path["iters"][-1]
        return model


__all__ = ["Stream"]
