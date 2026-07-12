"""Multi-GPU distributed attention kernel operator.

Shards embeddings across available CUDA devices and computes matvecs
in parallel. Falls back gracefully to single-device execution when only
one GPU is available (or none).

This is **data parallelism**, not model parallelism: each GPU holds
its own shard of the embeddings, but every matvec still needs the
*full* embedding matrix on the GPU that owns the local output chunk.
The peak memory per device is therefore

.. math::

    O(n_{\\text{local}} \\cdot d_e) + O(n \\cdot d_e)

(the local shard plus a replicated full copy of the embeddings). True
model parallelism (all-reduce over partial contributions) is **not**
implemented; see ``README.md`` limitations section 8.

The wrapper has two execution modes selected at construction time:

* **Single-device**: when only one CUDA device is detected, the wrapper
  delegates everything to an inner
  :class:`~laker.kernels.AttentionKernelOperator`. No sharding, no
  device transfers.
* **Multi-device**: embeddings are split into ``num_dev`` contiguous
  chunks. Each device hosts one chunk and computes its local slice of
  the matvec against the *gathered* full embedding matrix.
"""

import logging
from typing import Optional

import torch

from laker.kernels import AttentionKernelOperator, exp_safe

logger = logging.getLogger(__name__)


class DistributedAttentionKernelOperator:
    """Wrapper that distributes a dense attention kernel across multiple GPUs.

    Embeddings are split evenly among devices. Each GPU computes its
    local chunk of the matvec; results are gathered back to the master
    device.

    Args:
        embeddings: Full embeddings of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation.
        master_device: Device where the input/output vectors live
            (typically CPU or ``cuda:0``). Defaults to
            ``embeddings.device``.
        dtype: :class:`torch.dtype`.

    Note:
        This is **data parallelism** (shard vectors, replicate full
        embeddings across devices), not model parallelism. Peak memory
        per device is ``O(n_local * embedding_dim)`` for the local
        embedding shard plus ``O(n)`` for the full vector. The current
        implementation gathers the full embedding matrix to each device
        on every matvec; true model parallelism is not yet implemented.

    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
        master_device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialise the distributed kernel operator.

        Detection of available CUDA devices happens here. If only one
        CUDA device (or none) is detected, the wrapper constructs a
        single inner :class:`AttentionKernelOperator` and delegates
        every call to it. Otherwise, embeddings are sharded across the
        available devices.

        Args:
            embeddings: Full embedding matrix of shape ``(n, d)``.
            lambda_reg: Tikhonov regularisation parameter.
            master_device: Device where input/output vectors live.
            dtype: Floating-point dtype.

        Side effects:
            Allocates one local :class:`AttentionKernelOperator` per
            CUDA device in single-device mode, or one per device in
            multi-device mode. The full embeddings are moved to
            ``master_device`` and copied once per shard.

        """
        if dtype is None:
            dtype = embeddings.dtype

        if master_device is None:
            master_device = embeddings.device

        self.n = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.lambda_reg = float(lambda_reg)
        self.master_device = master_device
        self.device = master_device
        self.dtype = dtype
        self.shape = (self.n, self.n)
        # ``skip_clamp`` is recorded for diagnostic/inspection
        # purposes (e.g. ``LAKERCore`` may print it). The wrapper does
        # not use this flag directly — the per-device
        # ``AttentionKernelOperator`` instances each compute their own
        # ``skip_clamp`` from their local embedding shard.
        self.skip_clamp = True

        # Detect CUDA devices; if none are available the wrapper
        # collapses to single-device execution on ``master_device``.
        if torch.cuda.is_available():
            self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        else:
            self.devices = [master_device]

        if len(self.devices) == 1:
            # Single-device fast path: build a regular
            # ``AttentionKernelOperator`` and delegate everything to
            # it. ``self.operators`` is left empty so the multi-device
            # branches can short-circuit via ``self.single_device``.
            self.single_device = True
            self.local_op = AttentionKernelOperator(
                embeddings=embeddings.to(device=master_device, dtype=dtype),
                lambda_reg=lambda_reg,
                chunk_size=None,
                device=master_device,
                dtype=dtype,
            )
            return

        self.single_device = False
        # Shard embeddings across devices; see ``shard_embeddings``.
        self.shard_embeddings(embeddings.to(dtype=dtype))

    def shard_embeddings(self, embeddings: torch.Tensor) -> None:
        """Split embeddings evenly among available devices.

        The split is as balanced as possible: each shard has size
        ``n // num_dev`` and the first ``n % num_dev`` shards receive
        one extra embedding. This guarantees ``sum(shard_sizes) == n``
        regardless of how ``n`` divides ``num_dev``.

        Each shard is moved to its target device and wrapped in a
        local :class:`AttentionKernelOperator`.

        Args:
            embeddings: Full embedding matrix of shape ``(n, d)`` in
                ``self.dtype``.

        Side effects:
            Populates ``self.chunk_sizes`` and ``self.operators`` (one
            per device).

        """
        n = embeddings.shape[0]
        num_dev = len(self.devices)
        chunk_sizes = [n // num_dev] * num_dev
        # Distribute the remainder evenly across the first ``n % num_dev``
        # shards so the total is exactly ``n``.
        for i in range(n % num_dev):
            chunk_sizes[i] += 1

        self.chunk_sizes = chunk_sizes
        self.operators = []
        start = 0
        for device, chunk_size_local in zip(self.devices, chunk_sizes):
            end = start + chunk_size_local
            local_embeddings = embeddings[start:end].to(device=device)
            operator = AttentionKernelOperator(
                embeddings=local_embeddings,
                lambda_reg=self.lambda_reg,
                chunk_size=None,
                device=device,
                dtype=self.dtype,
            )
            self.operators.append(operator)
            start = end

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(\\lambda I + G)`` to vector(s) ``x``.

        Automatically handles device movement and gathers results back
        to ``master_device``.

        In multi-device mode the implementation:

        1. Moves ``x`` to ``master_device``.
        2. Gathers all per-device embedding shards into a full
           ``(n, embedding_dim)`` tensor on ``master_device``.
        3. For each device: copies both the input slice and the full
           embeddings to that device, computes the local matvec chunk
           with 1-D chunking over the reduction dimension (peak memory
           ``O(chunk_size_local * embedding_dim)``), and returns the
           chunk to ``master_device``.
        4. Concatenates the chunks into the final output.

        Args:
            x: Tensor of shape ``(n,)`` (single RHS) or ``(n, k)``
                (batch). Lives on any device; will be moved to
                ``master_device``.

        Returns:
            ``(\\lambda I + G) x`` of the same shape as ``x``, on
            ``master_device``.

        """
        if self.single_device:
            return self.local_op.matvec(x)

        # Multi-device path. ``x_master`` is the canonical copy of the
        # RHS on ``master_device``; ``full_embed`` is the gathered copy
        # of every per-device embedding shard. Both are replicated to
        # each device before the local matvec.
        x_master = x.to(self.master_device)
        full_embed = torch.cat(
            [operator.embeddings.to(self.master_device) for operator in self.operators],
            dim=0,
        )

        outputs = []
        start = 0
        for operator in self.operators:
            device = operator.embeddings.device
            local_embeddings = operator.embeddings
            local_size = local_embeddings.shape[0]
            end = start + local_size

            # Move the input vector and full embeddings to the device
            # so the local matvec is fully on-device (no cross-device
            # references).
            device_tensor = x_master.to(device)
            full_embeddings_device = full_embed.to(device)

            # 1-D chunking over the reduction dimension keeps peak
            # memory bounded at ``chunk_size_local * embedding_dim``
            # instead of the full ``n * embedding_dim``. ``8192`` was
            # chosen empirically to keep per-block memory under
            # ``~64 MB`` at ``float64`` (8192 * 8 bytes/element * 2
            # for gram + accumulator) while still amortising kernel
            # launch overhead.
            chunk_size_local = 8192
            local_out = self.lambda_reg * device_tensor
            for j_start in range(0, self.n, chunk_size_local):
                j_end = min(j_start + chunk_size_local, self.n)
                gram_block = local_embeddings @ full_embeddings_device[j_start:j_end].T
                exp_safe(gram_block, out=gram_block, skip_clamp=False)
                if device_tensor.dim() == 1:
                    local_out[start:end].addmv_(gram_block, device_tensor[j_start:j_end])
                else:
                    local_out[start:end].addmm_(gram_block, device_tensor[j_start:j_end])

            outputs.append(local_out[start:end].to(self.master_device))
            start = end

        return torch.cat(outputs, dim=0)

    def diagonal(self) -> torch.Tensor:
        """Return diagonal of ``\\lambda I + G``.

        Multi-device: each shard computes its local diagonal, results
        are gathered to ``master_device`` and concatenated. Single-device:
        delegates to the inner operator.

        Returns:
            Diagonal tensor of shape ``(n,)`` on ``master_device``.

        """
        if self.single_device:
            return self.local_op.diagonal()
        diags = []
        for operator in self.operators:
            diags.append(operator.diagonal().to(self.master_device))
        return torch.cat(diags)

    def to_dense(self) -> torch.Tensor:
        """Materialise the full dense matrix on ``master_device``.

        Multi-device: gather all per-device embeddings to master,
        compute ``exp(E E^T)`` in one shot, and add ``\\lambda I``.
        Single-device: delegates to the inner operator.

        Returns:
            Dense ``(n, n)`` matrix on ``master_device``.

        """
        if self.single_device:
            return self.local_op.to_dense()
        # Gather all embeddings to master device.
        full_embed = torch.cat(
            [operator.embeddings.to(self.master_device) for operator in self.operators],
            dim=0,
        )
        gram = full_embed @ full_embed.T
        exp_safe(gram, out=gram, skip_clamp=False)
        gram.diagonal().add_(self.lambda_reg)
        return gram

    def kernel_eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate exact kernel between queries and training points.

        Multi-device: gather per-device embeddings into a full matrix,
        then evaluate ``exp(x @ full_y^T)`` in one shot. Single-device:
        delegates to the inner operator.

        Args:
            x: Query embeddings of shape ``(n_queries, d)``.
            y: Optional training embeddings. If ``None``, the gathered
                full embeddings are used.
            chunk_size: Unused at this level; preserved for API
                compatibility with :class:`laker.kernels.KernelOperator`.

        Returns:
            Dense kernel matrix of shape ``(n_queries, n)``.

        """
        if self.single_device:
            return self.local_op.kernel_eval(x, y, chunk_size=chunk_size)
        # Gather all training embeddings, or use the caller's ``y``.
        full_y = (
            torch.cat(
                [operator.embeddings.to(self.master_device) for operator in self.operators],
                dim=0,
            )
            if y is None
            else y
        )
        gram = x @ full_y.T
        torch.exp(gram, out=gram)
        return gram
