"""Hyperparameter search: grid search and Bayesian optimisation.

Public class :class:`Search` providing two strategies for selecting the
key LAKER hyperparameters (``lam``, ``gamma``, ``num``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from laker.math import GP

if TYPE_CHECKING:
    from laker.core import Core
    from laker.model import Laker

logger = logging.getLogger(__name__)


class Search:
    """Validation-based hyperparameter search.

    * :meth:`grid` — exhaustive grid over ``lam``, ``gamma``, ``num``.
    * :meth:`bayes` — Bayesian optimisation with GP surrogate.
    """

    def __init__(self, core: "Core") -> None:
        self.core = core

    def grid(
        self,
        model: "Laker",
        x: torch.Tensor,
        y: torch.Tensor,
        val: float = 0.2,
        lam_grid: Optional[list[float]] = None,
        gamma_grid: Optional[list[float]] = None,
        num_grid: Optional[list[int]] = None,
        warm: bool = True,
        seed: Optional[int] = None,
    ) -> "Laker":
        """Grid search over ``lam x gamma x num`` with optional warm-starting."""
        from laker.check import Check

        x = Check.x(Check.tensor(x, device=model.device, dtype=model.dtype))
        y = Check.y(Check.tensor(y, device=model.device, dtype=model.dtype))

        n = x.shape[0]
        n_val = max(1, int(n * val))
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed) if seed is not None else int(torch.initial_seed()))
        perm = torch.randperm(n, generator=gen, device=x.device)
        train_idx = perm[n_val:]
        val_idx = perm[:n_val]

        if lam_grid is None:
            lam_grid = [1e-3, 1e-2, 1e-1]
        if gamma_grid is None:
            gamma_grid = [0.0, 1e-1, 1.0]
        if num_grid is None:
            num_grid = [50, 100, 200]

        embed, enc = self.core.embed(x)
        model.encoder = enc
        train_embed = embed[train_idx]

        best_rmse = float("inf")
        best_params = {}
        best_alpha: Optional[torch.Tensor] = None

        if self.core.verbose:
            logger.info(
                "Grid search: %d lam x %d gamma x %d num = %d configs",
                len(lam_grid),
                len(gamma_grid),
                len(num_grid),
                len(lam_grid) * len(gamma_grid) * len(num_grid),
            )

        x0 = None
        for lam_v in lam_grid:
            for gamma_v in gamma_grid:
                for num_v in num_grid:
                    try:
                        kernel = self.core.build_kernel(train_embed, lam=lam_v)
                        prec = self.core.build_prec(
                            kernel.matvec,
                            train_embed.shape[0],
                            gamma=gamma_v,
                            num=num_v,
                            diag=kernel.diag(),
                            seed=seed,
                        )
                        alpha, _ = self.core.solve(
                            kernel, prec, y[train_idx], x0=x0 if warm else None
                        )
                        x_val_emb = x[val_idx].to(dtype=self.core.embed_dtype)
                        with torch.no_grad():
                            val_embed = model.encoder(x_val_emb)
                        if self.core.embed_dtype != self.core.dtype:
                            val_embed = val_embed.to(dtype=self.core.dtype)
                        k_val = kernel.eval(val_embed, train_embed)
                        y_pred = k_val @ alpha
                        rmse = torch.sqrt(torch.mean((y_pred - y[val_idx]) ** 2)).item()
                    except (RuntimeError, ValueError) as exc:
                        rmse = float("inf")
                        if self.core.verbose:
                            logger.warning(
                                "Trial failed: lam=%.3e gamma=%.3e num=%d (%s)",
                                lam_v,
                                gamma_v,
                                num_v,
                                exc,
                            )

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {"lam": lam_v, "gamma": gamma_v, "num": num_v}
                        best_alpha = alpha
                        if self.core.verbose:
                            logger.info(
                                "New best: lam=%.3e gamma=%.3e num=%d rmse=%.4f",
                                lam_v,
                                gamma_v,
                                num_v,
                                rmse,
                            )
                    if warm and best_alpha is not None:
                        x0 = best_alpha.clone()

        if not best_params:
            raise RuntimeError(
                "Grid search failed: all parameter combinations diverged. "
                "Try widening lam_grid, increasing pcg_max_iter, or dtype=float64."
            )

        model.lam = float(best_params["lam"])
        model.gamma = float(best_params["gamma"])
        model.num = int(best_params["num"])
        return model.fit(x, y, seed=seed)

    def bayes(
        self,
        model: "Laker",
        x: torch.Tensor,
        y: torch.Tensor,
        val: float = 0.2,
        n_calls: int = 15,
        n_init: int = 5,
        lam_bounds: tuple[float, float] = (1e-4, 1.0),
        gamma_bounds: tuple[float, float] = (0.0, 2.0),
        num_bounds: tuple[int, int] = (20, 300),
        seed: Optional[int] = None,
    ) -> "Laker":
        """Bayesian optimisation over ``(lam, gamma, num)``."""
        from laker.check import Check

        x = Check.x(Check.tensor(x, device=model.device, dtype=model.dtype))
        y = Check.y(Check.tensor(y, device=model.device, dtype=model.dtype))

        n = x.shape[0]
        n_val = max(1, int(n * val))
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed) if seed is not None else int(torch.initial_seed()))
        perm = torch.randperm(n, generator=gen, device=x.device)
        train_idx = perm[n_val:]
        val_idx = perm[:n_val]

        embed, enc = self.core.embed(x)
        model.encoder = enc
        train_embed = embed[train_idx]

        bounds = np.array(
            [lam_bounds, gamma_bounds, num_bounds],
            dtype=np.float64,
        )

        def lh_sample(s: int, rng: np.random.Generator) -> np.ndarray:
            d = bounds.shape[0]
            samples = np.zeros((s, d), dtype=np.float64)
            for i in range(d):
                perm_np = rng.permutation(s)
                samples[:, i] = (perm_np + rng.uniform(size=s)) / s
            return samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

        bayes_rng = np.random.default_rng(
            int(seed) if seed is not None else int(torch.initial_seed())
        )

        x_obs = []
        y_obs = []

        for _ in range(n_init):
            point = lh_sample(1, bayes_rng)[0]
            lam_v = float(point[0])
            gamma_v = float(point[1])
            num_v = int(round(float(point[2])))
            rmse = self.eval(
                model,
                train_embed,
                x[val_idx],
                y,
                train_idx,
                val_idx,
                lam_v,
                gamma_v,
                num_v,
                seed,
            )
            x_obs.append(point)
            y_obs.append(rmse)

        gp = GP(bounds, log=[0, 1])
        best_rmse = min(y_obs)
        best_arr = x_obs[int(np.argmin(y_obs))]
        best_params = {
            "lam": float(best_arr[0]),
            "gamma": float(best_arr[1]),
            "num": int(round(float(best_arr[2]))),
        }

        for _ in range(n_calls - n_init):
            gp.fit(np.vstack(x_obs), np.array(y_obs, dtype=np.float64))
            candidates = lh_sample(500, bayes_rng)
            ei = gp.improve(candidates)
            next_point = candidates[int(np.argmax(ei))]

            lam_v = float(next_point[0])
            gamma_v = float(next_point[1])
            num_v = int(round(float(next_point[2])))
            rmse = self.eval(
                model,
                train_embed,
                x[val_idx],
                y,
                train_idx,
                val_idx,
                lam_v,
                gamma_v,
                num_v,
                seed,
            )
            x_obs.append(next_point)
            y_obs.append(rmse)

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {"lam": lam_v, "gamma": gamma_v, "num": num_v}

        model.lam = float(best_params["lam"])
        model.gamma = float(best_params["gamma"])
        model.num = int(best_params["num"])
        return model.fit(x, y, seed=seed)

    def eval(
        self,
        model: "Laker",
        train_embed: torch.Tensor,
        x_val: torch.Tensor,
        y: torch.Tensor,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        lam_v: float,
        gamma_v: float,
        num_v: int,
        seed: Optional[int],
    ) -> float:
        try:
            kernel = self.core.build_kernel(train_embed, lam=lam_v)
            prec = self.core.build_prec(
                kernel.matvec,
                train_embed.shape[0],
                gamma=gamma_v,
                num=num_v,
                diag=kernel.diag(),
                seed=seed,
            )
            alpha, _ = self.core.solve(kernel, prec, y[train_idx])

            x_val_emb = x_val.to(dtype=self.core.embed_dtype)
            with torch.no_grad():
                val_embed = model.encoder(x_val_emb)
            if self.core.embed_dtype != self.core.dtype:
                val_embed = val_embed.to(dtype=self.core.dtype)
            k_val = kernel.eval(val_embed, train_embed)
            y_pred = k_val @ alpha
            return torch.sqrt(torch.mean((y_pred - y[val_idx]) ** 2)).item()
        except (RuntimeError, ValueError):
            logger.debug("bayes trial failed", exc_info=True)
            return float("inf")


__all__ = ["Search"]
