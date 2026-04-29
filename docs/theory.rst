Mathematical Background
=======================

This page explains the mathematics behind LAKER in enough depth that you
can reason about hyperparameters, diagnose convergence, and adapt the method
to new problems.

Problem Formulation
-------------------

LAKER solves the *regularised attention kernel regression* problem

.. math::

   \min_{\alpha \in \mathbb{R}^n}
   \;\|\,G\alpha - y\,\|_2^2
   \;+\; \lambda \,\alpha^{\!\top} G \alpha
   \qquad (1)

where

* :math:`y \in \mathbb{R}^n` are noisy measurements,
* :math:`\lambda > 0` is a Tikhonov regularisation parameter,
* :math:`G \in \mathbb{R}^{n \times n}` is the **exponential attention kernel**

.. math::

   G_{ij} = \exp\!\bigl(\langle e_i, e_j \rangle\bigr)
   \qquad (2)

induced by learned embeddings :math:`E = [e_1,\dots,e_n]^{\!\top} \in \mathbb{R}^{n \times d_e}`.

The first term in (1) is a data-fitting loss; the second is a smoothness
penalty that prefers :math:`\alpha` lying in the span of the kernel.
Differentiating (1) yields the linear system

.. math::

   (\lambda I + G)\,\alpha = y
   \qquad (3)

which is what LAKER actually solves.

Why an Exponential Attention Kernel?
--------------------------------------

In spectrum cartography the received signal strength at location :math:`x_i`
is a smooth function of spatial coordinates.  A Gaussian (RBF) kernel
could be used, but it requires a carefully tuned length-scale.  The
*attention kernel* (2) is parameterised by the embedding inner products
:math:`\langle e_i, e_j \rangle`.  Because the embeddings are themselves
learned (or at least data-dependent, via ``PositionEmbedding``), the kernel
adaptively reshapes its “similarity landscape” to the geometry of the
measurements.  The exponential guarantees strict positive definiteness and
gives rapid decay for dissimilar points, which is exactly the inductive bias
needed for radio-map reconstruction.

The Ill-Conditioning Challenge
------------------------------

For :math:`n \gtrsim 10\,000` the matrix :math:`\lambda I + G` is
dense, :math:`O(n^2)` in memory and :math:`O(n^3)` to factorise.
Worse, the condition number

.. math::

   \kappa(\lambda I + G) = \frac{\lambda + \lambda_{\max}(G)}
                                {\lambda + \lambda_{\min}(G)}

can easily exceed :math:`10^8`, because the spectrum of an exponential
kernel typically has a handful of very large eigenvalues (corresponding to
global structure) and a long tail of tiny eigenvalues (high-frequency
variations).  A standard Conjugate Gradient (CG) solver therefore needs
hundreds or thousands of iterations, each costing an :math:`O(n^2)` matvec.

LAKER attacks both problems simultaneously:

1. **Matrix-free matvecs** – the kernel is never formed explicitly;
   chunked dot-products keep memory at :math:`O(n \cdot \text{chunk\_size})`.
2. **Learned data-dependent preconditioner** – a cheaply-computable
   :math:`P \approx (\lambda I + G)^{-1}` is learned via random probes so
   that the preconditioned system :math:`P(\lambda I + G)` has a compressed
   spectrum and CG converges in *tens* of iterations, essentially
   independently of :math:`n`.

The CCCP Preconditioner (Algorithm 1)
-------------------------------------

The paper proposes learning a preconditioner by treating the kernel matrix as
a covariance and estimating its inverse square-root :math:`\Sigma^{-1/2}`
from a small number of random probe directions.

Random probes
~~~~~~~~~~~~~

Draw :math:`N_r` random directions :math:`R \in \mathbb{R}^{n \times N_r}`
and apply the operator:

.. math::

   U = (\lambda I + G)\,R

Each column :math:`u_k` is a “probe response”.  The responses are
normalised to unit length, giving :math:`\bar{U}`, and an economy QR
factorisation yields an orthonormal basis :math:`Q \in \mathbb{R}^{n \times
N_r}`:

.. math::

   \bar{U} = Q R_{\mathrm{qr}}

The key insight is that the *span* of the probe responses captures the
dominant eigenspaces of the operator, because random vectors have non-negligible
overlap with *every* eigenvector.  By working entirely in the :math:`Q`-basis
we reduce the effective dimension from :math:`n` to :math:`N_r`.

Maximum-likelihood objective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The paper derives a regularised log-likelihood for :math:`\Sigma` given
the probe responses.  In the :math:`Q`-basis the covariance has a
*low-rank-plus-isotropic* structure:

.. math::

   \Sigma = a I + Q C Q^{\!\top}
   \qquad (4)

where :math:`a > 0` is the isotropic coefficient and :math:`C \in
\mathbb{R}^{N_r \times N_r}` is the low-rank correction.  The objective is
non-convex, so the authors apply the **Convex-Concave Procedure (CCCP)**.

CCCP iteration
~~~~~~~~~~~~~~

At each CCCP step we form the surrogate

.. math::

   F_{\gamma}
   = \frac{1}{1 + \gamma/n}
   \Bigl(\sum_{k=1}^{N_r} w_k \,\bar{u}_k \bar{u}_k^{\!\top}
         + \gamma I\Bigr)

where :math:`\gamma` is a regularisation parameter and the weights
:math:`w_k` depend on the *previous* iterate’s estimate of :math:`\Sigma`.
The CCCP update then solves a tractable quadratic program whose closed-form
solution is the new :math:`\Sigma`.

Shrinkage regularisation
~~~~~~~~~~~~~~~~~~~~~~~~

Because :math:`N_r \ll n` the probe sample is *undersampled*.  A pure
maximum-likelihood estimate would overfit to the probe directions and give
an ill-conditioned preconditioner.  LAKER therefore applies **isotropic
shrinkage**:

.. math::

   \Sigma_{\rho} = (1 - \rho)\,F_{\gamma} + \rho\,I

where the shrinkage parameter :math:`\rho` is *adaptive*: it increases
automatically when the undersampling ratio :math:`N_r/n` is small or when
:math:`\gamma` is large, providing a smooth interpolation between the
learned low-rank structure and a safe identity preconditioner.

Trace normalisation
~~~~~~~~~~~~~~~~~~~

Shrinkage biases the eigenvalues toward 1, but it also changes the mean
eigenvalue.  To keep the preconditioner neutral (i.e. not rescale the
overall magnitude of the linear system), LAKER applies **trace
normalisation** after each CCCP step:

.. math::

   \Sigma \leftarrow \frac{n}{\operatorname{tr}(\Sigma)} \,\Sigma

This ensures :math:`\operatorname{tr}(\Sigma) = n`, so the mean
eigenvalue remains 1 and the preconditioner does not artificially inflate or
deflate the residual norm monitored by PCG.

The Factored :math:`O(N_r^3)` Representation
--------------------------------------------

A naive implementation of CCCP would require :math:`O(n^3)` work per
iteration (full eigendecompositions of an :math:`n \times n` matrix).
LAKER exploits the fixed random-probe structure to reduce this to
:math:`O(N_r^3)`, independent of :math:`n`.

Because :math:`\Sigma` always has the form (4), every matrix operation
inside CCCP can be rewritten in the :math:`N_r`-dimensional :math:`Q`-basis:

* Inverting :math:`\Sigma` becomes inverting the :math:`N_r \times N_r`
  matrix :math:`M = a I + C`.
* Eigenvalue computations are performed on :math:`M`, not on the full
  :math:`n \times n` matrix.
* The preconditioner apply :math:`P = \Sigma^{-1/2}` decomposes as

  .. math::

     P x = a^{-1/2} x
           + Q\,V\,\bigl(\lambda_i^{-1/2} - a^{-1/2}\bigr)\,V^{\!\top}\,Q^{\!\top} x

  where :math:`M = V \operatorname{diag}(\lambda_i) V^{\!\top}`.
  This is an :math:`O(n N_r)` operation, again independent of the
  condition number.

In practice :math:`N_r \approx 2\sqrt{n}` (the adaptive heuristic used when
``num_probes=None``), so for :math:`n = 100\,000` we have :math:`N_r \approx
630` and the cubic term is negligible.

Preconditioned Conjugate Gradient
---------------------------------

Once :math:`P` is built, LAKER solves (3) with standard PCG.  The
preconditioned system is

.. math::

   P(\lambda I + G)\,\alpha = P y

Because :math:`P \approx (\lambda I + G)^{-1/2}`, the spectrum of the
preconditioned operator is clustered around 1.  The paper reports condition
number reductions of **up to three orders of magnitude**, which translates
directly into PCG convergence in **20–40 iterations** instead of several
thousand, even for :math:`n = 10^5`.

Complexity Summary
------------------

Let :math:`n` be the number of measurements and :math:`N_r` the number of
random probes.

+-----------------------------+--------------------------+----------------------+
| Step                        | Time                     | Memory               |
+=============================+==========================+======================+
| Embedding (forward)         | :math:`O(n d_e^2)`       | :math:`O(n d_e)`     |
+-----------------------------+--------------------------+----------------------+
| Kernel matvec (chunked)     | :math:`O(n^2 d_e)`       | :math:`O(n \cdot`    |
|                             |                          | ``chunk_size`` :math:`)` |
+-----------------------------+--------------------------+----------------------+
| CCCP preconditioner build   | :math:`O(N_r^3 + n N_r)` | :math:`O(n N_r)`     |
| (per iteration)             |                          |                      |
+-----------------------------+--------------------------+----------------------+
| PCG solve                   | :math:`O(\text{iters} \cdot n^2 d_e)` | :math:`O(n)` |
|                             |                          |                      |
+-----------------------------+--------------------------+----------------------+

Because :math:`\text{iters}` is :math:`O(1)` thanks to the preconditioner,
the dominant cost is the matvecs inside PCG, i.e. :math:`O(n^2 d_e)` total
for the solve.  The preconditioner learning itself is a small additive
overhead.

Practical Take-aways
--------------------

* **Start with the defaults.**  ``gamma=0.1``, ``num_probes=None``, and
  ``base_rho=0.05` are the values reported in the paper and work well for
  :math:`n \in [10^3, 10^5]`.
* **If PCG takes > 100 iterations**, increase ``num_probes`` or decrease
  ``gamma`` (less regularisation lets the preconditioner learn more
  structure).
* **If the preconditioner build is slow**, you can decrease ``num_probes``
  manually at the cost of slightly more PCG iterations.
* **For :math:`n < 5\,000`** you can set ``chunk_size=None`` to form the
  full kernel explicitly; for larger problems the auto-selected chunk size
  prevents out-of-memory errors.
* **Use ``float64``** when the condition number is very high
  (:math:`\kappa \gtrsim 10^{10}`); otherwise ``float32`` is usually fine
  and twice as fast on modern GPUs.
