"""Model persistence: save and load LAKER models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from laker.models import LAKERRegressor

logger = logging.getLogger(__name__)


class ModelPersistence:
    """Handles serialization and deserialization of LAKER models."""

    @staticmethod
    def save(regressor: "LAKERRegressor", path: str) -> None:
        """Serialize the fitted model to disk."""
        if regressor.alpha is None:
            raise RuntimeError("Model has not been fitted. Call fit() before saving.")
        state: dict[str, Any] = {
            "embedding_dim": regressor.embedding_dim,
            "lambda_reg": regressor.lambda_reg,
            "gamma": regressor.gamma,
            "num_probes": regressor.num_probes,
            "epsilon": regressor.epsilon,
            "base_rho": regressor.base_rho,
            "cccp_max_iter": regressor.cccp_max_iter,
            "cccp_tol": regressor.cccp_tol,
            "pcg_tol": regressor.pcg_tol,
            "pcg_max_iter": regressor.pcg_max_iter,
            "chunk_size": regressor.chunk_size,
            "kernel_approx": regressor.kernel_approx,
            "num_landmarks": regressor.num_landmarks,
            "num_features": regressor.num_features,
            "k_neighbors": regressor.k_neighbors,
            "grid_size": regressor.grid_size,
            "distributed": regressor.distributed,
            "device": str(regressor.device),
            "dtype": str(regressor.dtype),
            "embedding_dtype": (
                str(regressor.embedding_dtype) if regressor.embedding_dtype else None
            ),
            "verbose": regressor.verbose,
            "embeddings": regressor.embeddings,
            "alpha": regressor.alpha,
        }
        if regressor.embedding_model is not None:
            state["embedding_model_state"] = regressor.embedding_model.state_dict()
            state["embedding_model_class"] = regressor.embedding_model.__class__.__name__
            state["embedding_model_module"] = regressor.embedding_model.__class__.__module__
            if hasattr(regressor.embedding_model, "input_dim"):
                state["input_dim"] = regressor.embedding_model.input_dim
        if regressor.residual_corrector is not None:
            state["residual_corrector_state"] = regressor.residual_corrector.state_dict()
            state["residual_corrector_class"] = regressor.residual_corrector.__class__.__name__
            state["residual_corrector_module"] = regressor.residual_corrector.__class__.__module__
        torch.save(state, path)

    @staticmethod
    def load(path: str) -> "LAKERRegressor":
        """Deserialize a model from disk."""
        state = torch.load(path, weights_only=False)
        dtype = torch.float32 if "float32" in state["dtype"] else torch.float64
        embedding_dtype = (
            torch.float32
            if state.get("embedding_dtype") and "float32" in state["embedding_dtype"]
            else (
                torch.float64
                if state.get("embedding_dtype") and "float64" in state["embedding_dtype"]
                else None
            )
        )

        from laker.models import LAKERRegressor

        model = LAKERRegressor(
            embedding_dim=state["embedding_dim"],
            lambda_reg=state["lambda_reg"],
            gamma=state["gamma"],
            num_probes=state["num_probes"],
            epsilon=state["epsilon"],
            base_rho=state["base_rho"],
            cccp_max_iter=state["cccp_max_iter"],
            cccp_tol=state["cccp_tol"],
            pcg_tol=state["pcg_tol"],
            pcg_max_iter=state["pcg_max_iter"],
            chunk_size=state.get("chunk_size"),
            kernel_approx=state.get("kernel_approx"),
            num_landmarks=state.get("num_landmarks"),
            num_features=state.get("num_features"),
            k_neighbors=state.get("k_neighbors"),
            grid_size=state.get("grid_size"),
            distributed=state.get("distributed", False),
            embedding_dtype=embedding_dtype,
            device=state["device"],
            dtype=dtype,
            verbose=state["verbose"],
        )
        model.embeddings = state["embeddings"].to(model.device)
        model.alpha = state["alpha"].to(model.device)

        if "embedding_model_state" in state:
            class_name = state["embedding_model_class"]
            module_name = state.get("embedding_model_module", "laker.embeddings")
            try:
                import importlib

                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
            except (ImportError, AttributeError):
                logger.warning(
                    "Could not import %s.%s for loading; falling back to PositionEmbedding. "
                    "Save/load of custom embedding modules requires the module to be importable.",
                    module_name,
                    class_name,
                )
                from laker.embeddings import PositionEmbedding as cls

                class_name = "PositionEmbedding"

            input_dim = state.get("input_dim", 2)
            embed_dtype = embedding_dtype if embedding_dtype else dtype
            embed_cls: Callable[..., Any] = cls
            if class_name == "PositionEmbedding":
                model.embedding_model = embed_cls(
                    input_dim=input_dim,
                    embedding_dim=model.embedding_dim,
                    device=model.device,
                    dtype=embed_dtype,
                )
            else:
                try:
                    model.embedding_model = embed_cls(
                        input_dim=input_dim,
                        embedding_dim=model.embedding_dim,
                        device=model.device,
                        dtype=embed_dtype,
                    )
                except TypeError:
                    model.embedding_model = cls()
                    model.embedding_model.to(device=model.device, dtype=embed_dtype)
            model.embedding_model.load_state_dict(state["embedding_model_state"])

        if "residual_corrector_state" in state:
            corr_class_name = state.get("residual_corrector_class", "ResidualCorrector")
            corr_module_name = state.get("residual_corrector_module", "laker.correctors")
            try:
                import importlib

                corr_module = importlib.import_module(corr_module_name)
                corr_cls = getattr(corr_module, corr_class_name)
            except (ImportError, AttributeError):
                logger.warning(
                    "Could not import %s.%s for loading residual corrector; skipping.",
                    corr_module_name,
                    corr_class_name,
                )
                corr_cls = None

            if corr_cls is not None:
                input_dim = state.get("input_dim", 2)
                model.residual_corrector = corr_cls(
                    input_dim=input_dim,
                    output_dim=1,
                    hidden_dim=32,
                    dropout=0.1,
                ).to(model.device)
                model.residual_corrector.load_state_dict(state["residual_corrector_state"])

        from laker.kernels import (
            AttentionKernelOperator,
            NystromAttentionKernelOperator,
            RandomFeatureAttentionKernelOperator,
            SKIAttentionKernelOperator,
            SparseKNNAttentionKernelOperator,
        )

        if model.kernel_approx is None:
            model.kernel_operator = AttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                chunk_size=model.chunk_size,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_approx == "nystrom":
            model.kernel_operator = NystromAttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                num_landmarks=model.num_landmarks,
                chunk_size=model.chunk_size,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_approx == "rff":
            model.kernel_operator = RandomFeatureAttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                num_features=model.num_features,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_approx == "knn":
            model.kernel_operator = SparseKNNAttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                k_neighbors=model.k_neighbors,
                chunk_size=model.chunk_size,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_approx == "ski":
            model.kernel_operator = SKIAttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                grid_size=model.grid_size,
                device=model.device,
                dtype=dtype,
            )
        return model
