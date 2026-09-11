"""Core LAKER pipeline: embeddings, kernels, preconditioners, solves, predictions.

Single public class :class:`Core`. Stateless with respect to fitted
data; it stores only hyperparameters and device/dtype configuration.
Fitted state lives on the :class:`~laker.model.Laker` facade and is
passed as arguments to each method.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union, cast

import torch
import torch.nn as nn

from laker.backend import Backend
from laker.distributed import Distributed
from laker.embed import Position
from laker.kernel import (
    Exact,
    Fourier,
    Grid,
    Hybrid,
    Kernel,
    Neighbors,
    Nystrom,
    Spectrum,
    exp_safe,
)
from laker.prec import CCCP, Adaptive
from laker.solve import PCG

logger = logging.getLogger(__name__)


class Core:
    """LAKER pipeline: embeddings → kernel → preconditioner → solve → predict."""

    def __init__(
        self,
        embed_dim: int = 10,
        lam: float = 1e-2,
        gamma: float = 1e-1,
        num: Optional[int] = None,
        eps: float = 1e-8,
        base: float = 0.05,
        cccp_max: int = 200,
        cccp_tol: float = 1e-6,
        pcg_tol: float = 1e-6,
        pcg_max: int = 1000,
        chunk: Optional[int] = None,
        encoder: Optional[nn.Module] = None,
        kernel_type: Optional[str] = None,
        landmarks: Optional[int] = None,
        features: Optional[int] = None,
        neighbors: Optional[int] = None,
        grid_size: Optional[int] = None,
        distributed: bool = False,
        blend: float = 0.5,
        selection: str = "greedy",
        pilot: int = 1000,
        knots: int = 5,
        prec_kind: str = "cccp",
        embed_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = True,
    ) -> None:
        self.embed_dim = embed_dim
        self.lam = lam
        self.gamma = gamma
        self.num = num
        self.eps = eps
        self.base = base
        self.cccp_max = cccp_max
        self.cccp_tol = cccp_tol
        self.pcg_tol = pcg_tol
        self.pcg_max = pcg_max
        self.chunk = None if Backend.chunk_off else chunk
        self.encoder = encoder
        self.kernel_type = kernel_type
        self.landmarks = landmarks
        self.features = features
        self.neighbors = neighbors
        self.grid_size = grid_size
        self.distributed = distributed
        self.blend = blend
        self.selection = selection
        self.pilot = pilot
        self.knots = knots
        self.prec_kind = prec_kind
        self.verbose = verbose

        if device is None:
            device = Backend.device
        elif isinstance(device, str):
            device = torch.device(device)
        if dtype is None:
            dtype = Backend.dtype
        self.device = device
        self.dtype = dtype
        if embed_dtype is None:
            self.embed_dtype = dtype
        else:
            self.embed_dtype = embed_dtype

    def embed(self, x: torch.Tensor) -> tuple[torch.Tensor, nn.Module]:
        """Compute embeddings for ``x`` of shape ``(n, d)``."""
        n = x.shape[0]
        input_dim = x.shape[1]
        if self.verbose:
            logger.info("Fitting LAKER on n=%d, dx=%d", n, input_dim)

        embedded_input = x.to(dtype=self.embed_dtype)
        if self.encoder is not None:
            enc = self.encoder.to(self.device)
            with torch.no_grad():
                embed = enc(embedded_input)
        else:
            enc = Position(
                input_dim=input_dim,
                dim=self.embed_dim,
                device=self.device,
                dtype=self.embed_dtype,
            )
            with torch.no_grad():
                embed = enc(embedded_input)

        if self.embed_dtype != self.dtype:
            embed = embed.to(dtype=self.dtype)
            if self.verbose:
                logger.info(
                    "Mixed-precision: embed=%s solver=%s",
                    self.embed_dtype,
                    self.dtype,
                )
        return embed, enc

    def build_kernel(self, embed: torch.Tensor, lam: Optional[float] = None) -> Kernel:
        """Build a kernel operator for the given embeddings."""
        n = embed.shape[0]
        lam_value = float(lam) if lam is not None else self.lam
        chunk = self.chunk
        if chunk is None and n > 5000 and not Backend.chunk_off:
            chunk = max(1024, min(n // 10, 8192))
            if self.verbose:
                logger.info("Auto-selected chunk=%d for n=%d", chunk, n)

        op: Kernel
        if self.distributed and (self.kernel_type is None or self.kernel_type == "exact"):
            op = Distributed(embeddings=embed, lam=lam_value, master=self.device, dtype=self.dtype)
            if self.verbose:
                logger.info(
                    "Distributed kernel on %d device(s)", len(cast(Distributed, op).devices)
                )
        elif self.kernel_type is None or self.kernel_type == "exact":
            op = Exact(
                embeddings=embed, lam=lam_value, chunk=chunk, device=self.device, dtype=self.dtype
            )
        elif self.kernel_type == "nystrom":
            op = Nystrom(
                embeddings=embed,
                lam=lam_value,
                num=self.landmarks,
                method=self.selection,
                pilot=self.pilot,
                chunk=chunk,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info(
                    "Nystrom with m=%d landmarks (%s)",
                    cast(Nystrom, op).num_landmarks,
                    self.selection,
                )
        elif self.kernel_type == "fourier":
            op = Fourier(
                embeddings=embed,
                lam=lam_value,
                num=self.features,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info("Fourier with r=%d features", cast(Fourier, op).num)
        elif self.kernel_type == "neighbors":
            op = Neighbors(
                embeddings=embed,
                lam=lam_value,
                k=self.neighbors,
                chunk=chunk,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info("Neighbors with k=%d", cast(Neighbors, op).num_neighbors)
        elif self.kernel_type == "grid":
            op = Grid(
                embeddings=embed,
                lam=lam_value,
                grid_size=self.grid_size,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info("Grid with %d points", cast(Grid, op).points.shape[0])
        elif self.kernel_type == "hybrid":
            op = Hybrid(
                embeddings=embed,
                lam=lam_value,
                alpha=self.blend,
                num=self.landmarks,
                k=self.neighbors,
                chunk=chunk,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info("Hybrid with alpha=%.2f", cast(Hybrid, op).alpha)
        elif self.kernel_type == "spectrum":
            op = Spectrum(
                embeddings=embed,
                lam=lam_value,
                knots=self.knots,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info("Spectrum with %d knots", cast(Spectrum, op).shaper.knots)
        else:
            raise ValueError(f"Unknown kernel={self.kernel_type}")
        return op

    def prec(
        self,
        op: Callable[[torch.Tensor], torch.Tensor],
        n: int,
        gamma: Optional[float] = None,
        num: Optional[int] = None,
        seed: Optional[int] = None,
        diag: Optional[torch.Tensor] = None,
    ) -> Union[CCCP, Adaptive]:
        """Build the preconditioner for the given operator.

        Note: this method is also exposed as :meth:`Core.build_prec` to
        avoid name collision with the ``prec_kind`` config attribute.
        """
        return self.build_prec(op, n, gamma=gamma, num=num, seed=seed, diag=diag)

    def build_prec(
        self,
        op: Callable[[torch.Tensor], torch.Tensor],
        n: int,
        gamma: Optional[float] = None,
        num: Optional[int] = None,
        seed: Optional[int] = None,
        diag: Optional[torch.Tensor] = None,
    ) -> Union[CCCP, Adaptive]:
        """Build the preconditioner for the given operator."""
        if self.prec_kind == "adaptive":
            prec = Adaptive(
                gamma=gamma if gamma is not None else self.gamma,
                num=num if num is not None else self.num,
                eps=self.eps,
                base=self.base,
                max_iter=self.cccp_max,
                tol=self.cccp_tol,
                verbose=self.verbose,
                device=self.device,
                dtype=self.dtype,
            )
            prec.build(op, n, diag=diag, seed=seed)
            return prec

        cccp = CCCP(
            num=num if num is not None else self.num,
            gamma=gamma if gamma is not None else self.gamma,
            eps=self.eps,
            base=self.base,
            max_iter=self.cccp_max,
            tol=self.cccp_tol,
            verbose=self.verbose,
            device=self.device,
            dtype=self.dtype,
        )
        cccp.build(op, n, seed=seed)
        return cccp

    def solve(
        self,
        kernel_op: Kernel,
        prec_op: Union[CCCP, Adaptive],
        rhs: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int]:
        """Solve ``(K + lambda I) alpha = rhs`` with PCG."""
        pcg = PCG(tol=self.pcg_tol, max_iter=self.pcg_max, verbose=self.verbose)
        with Backend.autocast():
            alpha, status = pcg.solve(
                op=kernel_op.matvec,
                prec=prec_op.apply,
                rhs=rhs,
                x0=x0,
            )
        self.last_report = status
        if self.verbose:
            rel = (
                torch.linalg.norm(kernel_op.matvec(alpha) - rhs).item()
                / torch.linalg.norm(rhs).item()
            )
            logger.info("PCG iters=%d rel_res=%.3e", pcg.iterations, rel)
        return alpha, pcg.iterations

    def predict(
        self,
        x: torch.Tensor,
        enc: nn.Module,
        embed: torch.Tensor,
        kernel_op: Kernel,
        alpha: torch.Tensor,
        corrector: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Reconstruct the radio field at query locations."""
        with torch.no_grad(), Backend.autocast():
            embedded_input = x.to(dtype=self.embed_dtype)
            query_embed = enc(embedded_input)
            if self.embed_dtype != self.dtype:
                query_embed = query_embed.to(dtype=self.dtype)
            m = query_embed.shape[0]
            n = embed.shape[0]

            chunk_size = self.chunk
            if chunk_size is None and max(m, n) > 5000 and not Backend.chunk_off:
                chunk_size = max(1024, min(max(m, n) // 10, 8192))

            element_size = 4 if self.dtype == torch.float32 else 8
            mem_per_chunk = (
                (chunk_size or m) * n * element_size if chunk_size else m * n * element_size
            )
            if chunk_size is None or mem_per_chunk <= Backend.chunk:
                k_q = kernel_op.eval(query_embed, embed, chunk=chunk_size)
                out = k_q @ alpha
            elif self.kernel_type is not None:
                k_q = kernel_op.eval(query_embed, embed, chunk=chunk_size)
                out = k_q @ alpha
            else:
                out = torch.empty(m, device=self.device, dtype=self.dtype)
                for i in range(0, m, chunk_size):
                    i_end = min(i + chunk_size, m)
                    accum = torch.zeros(i_end - i, device=self.device, dtype=self.dtype)
                    e_i = query_embed[i:i_end]
                    for j in range(0, n, chunk_size):
                        j_end = min(j + chunk_size, n)
                        gb = e_i @ embed[j:j_end].T
                        exp_safe(gb, out=gb)
                        accum.addmv_(gb, alpha[j:j_end])
                    out[i:i_end] = accum

            if corrector is not None:
                corrector.eval()
                with torch.no_grad():
                    out = out + corrector(x).squeeze()
            return out

    def variance(
        self,
        x: torch.Tensor,
        enc: nn.Module,
        embed: torch.Tensor,
        kernel_op: Kernel,
        prec_op: Union[CCCP, Adaptive],
        alpha: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Predictive posterior variance at query locations."""
        with torch.no_grad(), Backend.autocast():
            embedded_input = x.to(dtype=self.embed_dtype)
            query_embed = enc(embedded_input)
            if self.embed_dtype != self.dtype:
                query_embed = query_embed.to(dtype=self.dtype)
            m = query_embed.shape[0]
            n = embed.shape[0]

            if self.kernel_type == "fourier" and hasattr(kernel_op, "phi"):
                ko = cast(Fourier, kernel_op)
                proj = query_embed @ ko.freq
                phi_q = torch.cat(
                    [torch.cos(proj + ko.phase), torch.sin(proj + ko.phase)],
                    dim=1,
                ) / (ko.num**0.5)
                a = ko.phi.T @ ko.phi
                a_reg = a + lam * torch.eye(a.shape[0], device=self.device, dtype=self.dtype)
                chol = torch.linalg.cholesky(a_reg)
                m_solve = torch.cholesky_solve(
                    torch.eye(a.shape[0], device=self.device, dtype=self.dtype), chol
                )
                var = lam * torch.sum(phi_q @ m_solve * phi_q, dim=1)
                return var.clamp(min=0.0)

            chunk_size = self.chunk
            if chunk_size is None and not Backend.chunk_off:
                element_size = 4 if self.dtype == torch.float32 else 8
                mem_needed = m * n * element_size
                if mem_needed > Backend.chunk:
                    chunk_size = max(1024, min(n // 10, 8192))

            var = torch.empty(m, device=self.device, dtype=self.dtype)
            pcg = PCG(tol=self.pcg_tol, max_iter=self.pcg_max, verbose=False)

            if chunk_size is None or m <= chunk_size:
                k_tq = kernel_op.eval(embed, query_embed)
                if k_tq.is_sparse:
                    k_tq = k_tq.to_dense()
                v, status = pcg.solve(op=kernel_op.matvec, prec=prec_op.apply, rhs=k_tq)
                if not status.converged:
                    logger.warning(
                        "variance PCG did not converge (reason=%s, residual=%.3e); "
                        "returning NaN.",
                        status.reason,
                        status.residual,
                    )
                    return torch.full_like(var, float("nan"))
                k_diag_mat = kernel_op.eval(query_embed, query_embed)
                if k_diag_mat.is_sparse:
                    k_diag_mat = k_diag_mat.to_dense()
                k_diag = k_diag_mat.diagonal()
                var[:] = k_diag - torch.sum(k_tq * v, dim=0)
            else:
                for start in range(0, m, chunk_size):
                    end = min(start + chunk_size, m)
                    q_c = query_embed[start:end]
                    k_tc = kernel_op.eval(embed, q_c)
                    if k_tc.is_sparse:
                        k_tc = k_tc.to_dense()
                    v_c, status_c = pcg.solve(
                        op=kernel_op.matvec, prec=prec_op.apply, rhs=k_tc
                    )
                    if not status_c.converged:
                        logger.warning(
                            "variance PCG did not converge on chunk [%d:%d] "
                            "(reason=%s, residual=%.3e); returning NaN.",
                            start,
                            end,
                            status_c.reason,
                            status_c.residual,
                        )
                        var[start:end] = float("nan")
                        continue
                    k_diag_mat = kernel_op.eval(q_c, q_c)
                    if k_diag_mat.is_sparse:
                        k_diag_mat = k_diag_mat.to_dense()
                    k_diag_c = k_diag_mat.diagonal()
                    var[start:end] = k_diag_c - torch.sum(k_tc * v_c, dim=0)

            return var.clamp(min=0.0)

    def predict_train(
        self,
        x: torch.Tensor,
        enc: nn.Module,
        embed: torch.Tensor,
        kernel_op: Kernel,
        alpha: torch.Tensor,
        corrector: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Differentiable version of :meth:`predict` (no ``torch.no_grad``)."""
        with Backend.autocast():
            embedded_input = x.to(dtype=self.embed_dtype)
            query_embed = enc(embedded_input)
            if self.embed_dtype != self.dtype:
                query_embed = query_embed.to(dtype=self.dtype)
            m = query_embed.shape[0]
            n = embed.shape[0]

            chunk_size = self.chunk
            if chunk_size is None and max(m, n) > 5000 and not Backend.chunk_off:
                chunk_size = max(1024, min(max(m, n) // 10, 8192))

            element_size = 4 if self.dtype == torch.float32 else 8
            mem_per_chunk = (
                (chunk_size or m) * n * element_size if chunk_size else m * n * element_size
            )
            if chunk_size is None or mem_per_chunk <= Backend.chunk:
                k_q = kernel_op.eval(query_embed, embed, chunk=chunk_size)
                out = k_q @ alpha
            elif self.kernel_type is not None:
                k_q = kernel_op.eval(query_embed, embed, chunk=chunk_size)
                out = k_q @ alpha
            else:
                out = torch.empty(m, device=self.device, dtype=self.dtype)
                for i in range(0, m, chunk_size):
                    i_end = min(i + chunk_size, m)
                    accum = torch.zeros(i_end - i, device=self.device, dtype=self.dtype)
                    e_i = query_embed[i:i_end]
                    for j in range(0, n, chunk_size):
                        j_end = min(j + chunk_size, n)
                        gb = e_i @ embed[j:j_end].T
                        exp_safe(gb, out=gb)
                        accum.addmv_(gb, alpha[j:j_end])
                    out[i:i_end] = accum

        if corrector is not None:
            corrector.eval()
            out = out + corrector(x).squeeze()
        return out

    def predict_var_train(
        self,
        x: torch.Tensor,
        enc: nn.Module,
        embed: torch.Tensor,
        kernel_op: Kernel,
        prec_op: Union[CCCP, Adaptive],
        alpha: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Differentiable variance proxy (exact for Fourier; soft-min otherwise)."""
        with Backend.autocast():
            embedded_input = x.to(dtype=self.embed_dtype)
            query_embed = enc(embedded_input)
            if self.embed_dtype != self.dtype:
                query_embed = query_embed.to(dtype=self.dtype)

            if self.kernel_type == "fourier" and hasattr(kernel_op, "phi"):
                ko = cast(Fourier, kernel_op)
                proj = query_embed @ ko.freq
                phi_q = torch.cat(
                    [torch.cos(proj + ko.phase), torch.sin(proj + ko.phase)], dim=1
                ) / (ko.num**0.5)
                a = ko.phi.T @ ko.phi
                a_reg = a + lam * torch.eye(a.shape[0], device=self.device, dtype=self.dtype)
                chol = torch.linalg.cholesky(a_reg)
                m_solve = torch.cholesky_solve(
                    torch.eye(a.shape[0], device=self.device, dtype=self.dtype), chol
                )
                var = lam * torch.sum(phi_q @ m_solve * phi_q, dim=1)
                return var.clamp(min=0.0)

            dists = torch.cdist(query_embed, embed) ** 2
            weights = torch.softmax(-dists, dim=1)
            mean_dist = (weights * dists).sum(dim=1)
            return (lam + mean_dist).clamp(min=0.0)

    def condition(self, kernel_op: Kernel, prec_op: Union[CCCP, Adaptive]) -> float:
        """Estimate ``kappa(P^{-1} K)`` via power iteration."""
        n = kernel_op.size
        apply_op = kernel_op.matvec
        apply_prec = prec_op.apply

        def precond(v: torch.Tensor) -> torch.Tensor:
            return apply_prec(apply_op(v))

        v = torch.randn(n, device=self.device, dtype=self.dtype)
        v = v / torch.linalg.norm(v)
        for _ in range(10):
            v = precond(v)
            v = v / torch.linalg.norm(v)
        lam_max = torch.dot(v, precond(v)).item()

        v = torch.randn(n, device=self.device, dtype=self.dtype)
        v = v / torch.linalg.norm(v)
        pcg = PCG(tol=1e-6, max_iter=50, verbose=False)
        for _ in range(5):
            v, _ = pcg.solve(op=precond, prec=lambda x: x, rhs=v)
            v = v / torch.linalg.norm(v)
        lam_min = max(
            torch.dot(v, precond(v)).item(),
            torch.finfo(self.dtype).eps,
        )

        return lam_max / lam_min


__all__ = ["Core"]
