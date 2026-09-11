"""Multi-GPU distributed attention kernel operator.

Shards embeddings across available CUDA devices and computes matvecs in
parallel. Falls back gracefully to single-device execution when only
one GPU is available (or none).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from laker.kernel import Exact, exp_safe

logger = logging.getLogger(__name__)


class Distributed:
    """Wrapper that distributes a dense attention kernel across multiple GPUs.

    Embeddings are split evenly among devices. Each GPU computes its
    local chunk of the matvec; results are gathered back to the master
    device.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lam: float = 1e-2,
        master: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if dtype is None:
            dtype = embeddings.dtype
        if master is None:
            master = embeddings.device

        self.size = embeddings.shape[0]
        self.dim = embeddings.shape[1]
        self.lam = float(lam)
        self.master = master
        self.device = master
        self.dtype = dtype
        self.shape = (self.size, self.size)
        self.skip = True

        if torch.cuda.is_available():
            self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        else:
            self.devices = [master]

        if len(self.devices) == 1:
            self.single = True
            self.local_op = Exact(
                embeddings=embeddings.to(device=master, dtype=dtype),
                lam=lam,
                chunk=None,
                device=master,
                dtype=dtype,
            )
            return

        self.single = False
        self.shard(embeddings.to(dtype=dtype))

    def shard(self, embeddings: torch.Tensor) -> None:
        n = embeddings.shape[0]
        nd = len(self.devices)
        sizes = [n // nd] * nd
        for i in range(n % nd):
            sizes[i] += 1
        self.sizes = sizes
        self.ops = []
        self.shards = []
        self.slices = []
        start = 0
        for device, sz in zip(self.devices, sizes):
            end = start + sz
            local = embeddings[start:end].to(device=device)
            op = Exact(
                embeddings=local,
                lam=self.lam,
                chunk=None,
                device=device,
                dtype=self.dtype,
            )
            self.ops.append(op)
            self.shards.append(local)
            self.slices.append((start, end))
            start = end

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        if self.single:
            return self.local_op.matvec(x)

        x_m = x.to(self.master)
        outs = []
        chunk_size = 8192
        for op, shard, (start, end) in zip(self.ops, self.shards, self.slices):
            device = shard.device
            d_t = x_m.to(device)
            local_out = self.lam * d_t[start:end]
            for j in range(0, self.size, chunk_size):
                j_end = min(j + chunk_size, self.size)
                if j >= start and j_end <= end:
                    remote = shard[(j - start):(j_end - start)]
                else:
                    remote_full = torch.cat(self.shards, dim=0)
                    remote = remote_full[j:j_end].to(device)
                gb = shard @ remote.T
                exp_safe(gb, out=gb, skip=False)
                if d_t.dim() == 1:
                    local_out.addmv_(gb, d_t[j:j_end])
                else:
                    local_out.addmm_(gb, d_t[j:j_end])
            outs.append(local_out.to(self.master))

        return torch.cat(outs, dim=0)

    def diag(self) -> torch.Tensor:
        if self.single:
            return self.local_op.diag()
        diags = [op.diag().to(self.master) for op in self.ops]
        return torch.cat(diags)

    def dense(self) -> torch.Tensor:
        if self.single:
            return self.local_op.dense()
        full = torch.cat([shard.to(self.master) for shard in self.shards], dim=0)
        gram = full @ full.T
        exp_safe(gram, out=gram, skip=False)
        gram.diagonal().add_(self.lam)
        return gram

    def eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk: Optional[int] = None,
    ) -> torch.Tensor:
        if self.single:
            return self.local_op.eval(x, y, chunk=chunk)
        full = (
            torch.cat([shard.to(self.master) for shard in self.shards], dim=0)
            if y is None
            else y
        )
        gram = x @ full.T
        torch.exp(gram, out=gram)
        return gram


__all__ = ["Distributed"]
