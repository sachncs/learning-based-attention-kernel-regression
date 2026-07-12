"""Implicit differentiation through PCG fixed-point for hypergradients.

In LAKER's bilevel optimisation framework, the inner problem is to solve

.. math::
    A(\\theta) \\alpha = y

where :math:`A(\\theta) = K(\\theta) + \\lambda I` depends on
learnable hyperparameters :math:`\\theta` (e.g. regularisation strength
or embedding weights). The outer problem minimises a validation loss
:math:`\\mathcal{L}(\\alpha(\\theta))`.

Rather than differentiating through every PCG iteration (expensive and
memory-intensive), the **adjoint method** (implicit differentiation)
computes the hypergradient in two steps:

1. **Adjoint solve.** Solve :math:`A v = \\nabla_\\alpha \\mathcal{L}`
   for the adjoint vector :math:`v` using PCG with the same operator and
   preconditioner.

2. **Parameter gradient.** For each hyperparameter :math:`\\theta_k`,

   .. math::
       \\frac{d\\mathcal{L}}{d\\theta_k}
       = -v^\\top \\frac{\\partial A}{\\partial \\theta_k} \\alpha

   where the partial derivative is computed efficiently via
   :func:`torch.autograd.grad` on the scalar
   :math:`v^\\top A(\\theta) \\alpha`.

This approach costs only one additional PCG solve (the adjoint) plus
cheap per-parameter dot products, making it far more efficient than
unrolling through the full CG trajectory.
"""

from typing import Callable, List

import torch

from laker.solvers import PreconditionedConjugateGradient


def hypergradient(
    operator_fn: Callable[[torch.Tensor], torch.Tensor],
    preconditioner_fn: Callable[[torch.Tensor], torch.Tensor],
    alpha: torch.Tensor,
    dL_dalpha: torch.Tensor,
    param_list: List[torch.Tensor],
    pcg_tol: float = 1e-6,
    pcg_max_iter: int = 500,
    verbose: bool = False,
) -> List[torch.Tensor]:
    r"""Compute hypergradients via the adjoint method (implicit differentiation).

    Given the fixed-point solution :math:`\alpha^*` satisfying
    :math:`A(\theta) \alpha = y` for some operator
    :math:`A(\theta) = K(\theta) + \lambda I` that depends on learnable
    parameters :math:`\theta`, and a scalar loss
    :math:`\mathcal{L}(\alpha)`, this function computes

    .. math::
        \frac{d\mathcal{L}}{d\theta_k}
        = -v^\top \frac{\partial A}{\partial \theta_k} \alpha^*

    for each parameter in ``param_list``, where :math:`v` is the adjoint
    vector satisfying :math:`A v = \nabla_\alpha \mathcal{L}`.

    The computation proceeds in two stages:

    1. **Adjoint solve.** Solve :math:`A v = \nabla_\alpha \mathcal{L}`
       for :math:`v` using preconditioned conjugate gradient (PCG) with
       the same operator and preconditioner used for the forward solve.
    2. **Parameter gradients.** For each :math:`\theta_k` in
       ``param_list``, compute the scalar
       :math:`s = v^\top A(\theta) \alpha` in a differentiable context,
       then use :func:`torch.autograd.grad` to obtain
       :math:`\partial s / \partial \theta_k`. The hypergradient is the
       negation of this partial derivative.

    Args:
        operator_fn: Callable that applies :math:`A(\theta)` to a vector
            or batch of vectors.
        preconditioner_fn: Callable that applies the preconditioner
            :math:`P` to a vector or batch of vectors.
        alpha: Fixed-point solution :math:`\alpha^*` of shape ``(n,)``
            or ``(n, k)``. This tensor should be **detached** from the
            computation graph (the adjoint solve is independent).
        dL_dalpha: Gradient of the outer loss with respect to
            :math:`\alpha`, of the same shape as ``alpha``.
        param_list: List of parameter tensors :math:`\theta_k}` to
            differentiate with respect to. Parameters without
            ``requires_grad`` receive a zero hypergradient.
        pcg_tol: Relative tolerance for the adjoint PCG solve.
        pcg_max_iter: Maximum number of PCG iterations for the adjoint
            solve.
        verbose: Whether to log the adjoint solve progress.

    Returns:
        List of hypergradient tensors, one per element of ``param_list``
        and each of the same shape as the corresponding parameter.
        Parameters without ``requires_grad`` receive a zero tensor.

    .. math::
        \nabla_\theta \mathcal{L}
        = \bigl[\,-v^\top (\partial A / \partial \theta_1)\, \alpha^*,\;
           \ldots,\;
           -v^\top (\partial A / \partial \theta_m)\, \alpha^*\bigr]

    """
    # Adjoint solve: A v = dL_dalpha
    pcg = PreconditionedConjugateGradient(
        tol=pcg_tol,
        max_iter=pcg_max_iter,
        verbose=verbose,
    )
    v = pcg.solve(
        operator=operator_fn,
        preconditioner=preconditioner_fn,
        rhs=dL_dalpha,
    )

    # For each parameter theta, compute -v^T (dA/dtheta) alpha
    # via torch.autograd.grad on the scalar v^T A(alpha) alpha.
    # We recompute A(alpha) in a differentiable context.
    alpha_const = alpha.detach()
    v_const = v.detach()

    with torch.enable_grad():
        # Re-enable grad for the operator evaluation
        alpha_for_grad = alpha_const.clone().requires_grad_(True)
        A_alpha = operator_fn(alpha_for_grad)
        scalar = torch.dot(v_const, A_alpha)

    hypergrads = []
    for param in param_list:
        if param.requires_grad:
            # d(scalar)/d(param) = v^T (dA/dparam) alpha
            # We want -v^T (dA/dparam) alpha, so negate.
            grad = torch.autograd.grad(
                scalar,
                param,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad is None:
                grad = torch.zeros_like(param)
            hypergrads.append(-grad)
        else:
            hypergrads.append(torch.zeros_like(param))

    return hypergrads
