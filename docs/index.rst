LAKER Documentation
===================

**LAKER** (Learning-based Attention Kernel Regression) is a production-ready
Python package for large-scale spectrum cartography and radio map reconstruction.

.. note::
   This repository is an independent implementation of the LAKER algorithm.
   I am not an author of the original paper.

It solves regularised attention kernel regression problems of the form

.. math::

   \min_\alpha \|G \alpha - y\|_2^2 + \lambda \alpha^\top G \alpha

where :math:`G = \exp(E E^\top)` is an exponential attention kernel.
The key innovation is a learned data-dependent preconditioner obtained via
a shrinkage-regularised Convex-Concave Procedure (CCCP), which reduces
condition numbers by up to three orders of magnitude and enables
near size-independent Preconditioned Conjugate Gradient (PCG) convergence.

Based on `Accelerating Regularized Attention Kernel Regression for Spectrum Cartography <https://arxiv.org/html/2604.25138v1>`_ (Tao & Tan, 2026).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   theory
   api
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
