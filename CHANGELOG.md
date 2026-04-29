# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-29

### Added
- Initial release of LAKER (Learning-based Attention Kernel Regression).
- `LAKERRegressor`: sklearn-compatible estimator for attention kernel regression.
- `AttentionKernelOperator`: matrix-free exponential attention kernel with chunked evaluation.
- `CCCPPreconditioner`: learned data-dependent preconditioner via shrinkage-regularised CCCP with factored `O(N_r^3)` implementation.
- `PreconditionedConjugateGradient`: standard PCG solver with convergence monitoring.
- `PositionEmbedding`: deterministic position-driven embedding module.
- Baseline solvers: `GradientDescent` and `JacobiPreconditioner`.
- Synthetic radio-field generators and visualisation utilities.
- Comprehensive test suite with 14+ tests.
- Documentation and usage examples.

[Unreleased]: https://github.com/convexsoft/kernelSC/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/convexsoft/kernelSC/releases/tag/v0.1.0
