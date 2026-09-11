# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive module-, class-, and method-level docstrings with Google-style parameter, return, and example sections across the entire codebase (``laker/``, ``benchmarks/``, ``examples/``, ``tests/``).
- Algorithm background sections (paper references, mathematical framing) for key modules (``bilevel``, ``implicit_diff``, ``preconditioner``, ``search``, ``streaming``, ``training``, ``persistence``, ``core``).
- Module-level docstrings for ``benchmarks/`` and ``examples/`` packages.
- Detailed docstring conventions in ``CONTRIBUTING.md`` with examples for modules, classes, functions, and tests.
- Added Python 3.13 classifier to ``pyproject.toml``.
- ``docs/`` directory: full developer / researcher documentation in four sub-trees:
  - ``docs/guides/`` (6 files) — practical tutorials: getting started, kernel choice, hyperparameter search, streaming, training, persistence.
  - ``docs/api/`` (22 files) — one API reference page per module.
  - ``docs/algorithms/`` (6 files) — mathematical background for the attention kernel, CCCP preconditioner, PCG solver, low-rank approximations, implicit differentiation, bilevel learning.
  - ``docs/examples/`` (7 files) — walkthrough of every example script.
- ``docs/README.md`` as the documentation entry point with table of contents.
- ``examples/scalable_data.py`` (``ScalableData``): download, verify, extract, index, clean, transform and load the real-world UCF-50K spectrum-cartography corpus (``KR-init/Spectrum-Cartography-256x256-UCF-50K``, ~10 GB, 50,000 ray-traced radio maps). Torch-free ETL with sentinels, fingerprinting and a traceability report.
- ``examples/scalable.py`` (``Scalable``): full reproducible end-to-end sweep over every LAKER kernel configuration on UCF-50K — per-config table, accuracy-vs-time Pareto front, winner selection, then validation of the winner on the complete masked 256×256 grid over the full corpus. Resume-safe via incremental per-map CSV append and process-pool parallelism. CLI exposes ``--data-dir``, ``--unpack-dir``, ``--out-dir``, ``--skip-download``, ``--skip-extract``, ``--max-maps``, ``--sweep-split``, ``--sensors``, ``--eval-points``, ``--lams``, ``--configs``, ``--seed``, ``--device``, ``--dtype`` (``float64``|``float32``), ``--condition``, ``--validate-maps``, ``--validate-splits``, ``--workers``, ``--resume`` and ``--verbose``. Every run writes an event log (``events.jsonl``), sweep / validate CSVs, a metrics summary, and a manifest with the git provenance and dataset fingerprint.
- ``examples/paper.py`` (``Paper``): reproduces the numerical experiment of the LAKER paper (arXiv:2604.25138, Section V) on the paper's synthetic scene — operator conditioning, PCG iterations to a ``1e-3`` objective gap for the learned CCCP preconditioner vs Jacobi-PCG vs gradient descent, and reconstruction RMSE / NMSE against an exact dense reference solve and a Gaussian-process ``RationalQuadratic`` baseline. Runs in ~2 minutes on a CPU; fails loudly if the preconditioner loses its κ-reduction, if LAKER-PCG is not strictly faster than Jacobi-PCG, or if the reconstruction diverges from the exact reference solve.
- ``docs/examples/scalable.md`` and ``docs/examples/paper.md``: walkthroughs of the new example scripts with result tables and assertions.
- ``examples.scalable --cross-map`` adds a stationary cross-map baseline: each validation map is scored against the per-pixel mean map aggregated over the 40,000 training maps (cached at ``data/ucf50k/cross_map_mean.npy``). The cache is built lazily on first run; ``cross_map_rmse`` appears as a column in ``validate.csv`` and as an aggregate in ``metrics.json``.
- ``examples.cross_map_cache`` and ``examples.cross_map_score``: standalone utilities to build the cross-map cache once and to score it against the full corpus (``data/ucf50k/cross_map_score.csv``, per-split RMSE summary). The full-corpus measurement on 50,000 maps gives **20.73 ± 1.50 dB**, i.e. essentially the same as the per-scene mean baseline — confirming that per-scene conditioning, not cross-map learning, is what carries LAKER's accuracy on this corpus.
- ``docs/examples/scalable.md`` adds a "Positioning vs learned (CNN) baselines" section anchoring LAKER's 10.38 dB (≈10 % of the 99 dB range) against RadioUNet's reported ~1 dB on the analogous RadioMapSeer task (~1 % of range); notes that no leaderboard / published baseline exists on UCF-50K as of writing.
- Registered ``paper`` and ``scalable`` / ``scalable_data`` in ``examples/__init__.py`` and across the documentation (``README.md``, ``docs/README.md``, ``docs/examples/README.md``).

## [0.5.0] - 2026-08-11

### Summary
The 0.5.0 cycle is the public-API rename and the docs push. The
single top-level export is now ``Laker`` (was ``LAKERRegressor``);
module-level helper classes (``Store``, ``Stream``, ``Search``,
``Trainer``, ``Math``, ``Check``, ``Backend``, ``Corrector``, etc.)
were promoted to single-word canonical names; the fitted state uses
plain names (``coef``, ``embed``, ``kernel``, ``prec``, ``encoder``,
``inputs``, ``targets``, ``iters``) with no trailing underscores; and
the codebase is now covered by 310+ tests, a 22-file ``docs/`` tree,
two full reproducible example scripts (``paper.py`` reproduces the
paper's Section V experiment; ``scalable.py`` runs the UCF-50K full
sweep), and a positioning baseline that confirms LAKER's accuracy on
UCF-50K is per-scene-conditioning driven, not cross-map learning.

### Added
- Renamed single top-level class to ``Laker`` (was ``LAKERRegressor``).
- Renamed / promoted module helpers: ``Kernel`` is now a union type in
  ``laker.kernel``; ``Nystrom``, ``Fourier``, ``Neighbors``, ``Grid``,
  ``Hybrid``, ``Spectrum``, ``Shaper``, ``exp_safe`` are direct exports;
  ``Preconditioner`` → ``CCCP`` / ``Adaptive``; ``Solve`` → ``PCG``;
  ``Embed`` → ``Position`` / ``Visual``; ``Helpers`` → ``Math``; ``Base``
  → ``Check``; ``Fit`` → ``Trainer``.
- Compact rename map for the public API:
  ``_partial_count`` → ``partial_count``, ``_x_train`` → ``x_train``,
  ``_y_train`` → ``y_train``, ``_LAKERRegressor`` → ``Laker``,
  ``embedding_dim`` → ``embed_dim``, ``regularization`` → ``lam``,
  ``kernel`` → ``kernel_type``.
- The package README is now a one-paragraph pointer; the canonical
  README is at the repo root and the canonical API reference is in
  ``docs/``.

### Changed
- The full ``CHANGELOG.md`` content above this entry was retroactively
  promoted from ``[Unreleased]`` to ``[0.5.0]``; future cycles will
  add their own ``[X.Y.Z]`` block.

### Changed
- Promoted all semi-private (single-underscore-prefixed) names to public:
  - ``LAKERRegressor`` attributes: ``_core`` → ``core``, ``_search`` → ``search``, ``_streaming`` → ``streaming``, ``_trainer`` → ``trainer``, ``_persistence`` → ``persistence``.
  - ``LAKERRegressor`` class constant: ``_HYPERPARAMS`` → ``HYPERPARAMS``.
  - ``NystromAttention`` methods: ``_select_landmarks_greedy`` → ``select_landmarks_greedy``, ``_select_landmarks_leverage`` → ``select_landmarks_leverage``.
  - ``AdaptivePreconditioner`` attributes: ``_inner`` → ``inner``, ``_inner_name`` → ``inner_name``.
- **Dropped every underscore prefix and suffix from the public API.**
  ``coef`` / ``embed`` / ``kernel`` / ``prec`` / ``encoder`` / ``inputs`` /
  ``targets`` / ``iters`` replace the previous sklearn-style trailing
  underscores; ``core`` / ``stream`` / ``searcher`` / ``train`` /
  ``init_encoder`` / ``x_train`` / ``y_train`` / ``partial_count`` /
  ``regpath`` replace the previous leading-underscore private
  attributes; the helper methods ``kernel`` / ``transform`` / ``ml`` /
  ``impl`` / ``landmarks_greedy`` / ``landmarks_leverage`` / ``coo`` /
  ``nystrom_matvec`` / ``rff_matvec`` / ``spectral_matvec`` / ``shard``
  / ``solve1`` / ``solve2`` / ``make`` / ``eval`` lose their leading
  underscores. The constructor parameter ``kernel`` was renamed to
  ``kernel_type`` to disambiguate the config string from the fitted
  ``Kernel`` instance; ``search`` and ``path`` instance attributes were
  renamed to ``searcher`` and ``regpath`` to avoid colliding with the
  public methods of the same name.

### Refactored (single-word naming, public-API rewrite)
- Deleted legacy modules and consolidated duplicates:
  - ``models.py`` / ``model.py`` → single ``model.py`` (``Laker`` is now the real implementation, no ``LAKERRegressor`` delegation).
  - ``kernels.py`` → ``kernel.py``; ``solvers.py`` → ``solve.py``; ``preconditioner.py`` → ``prec.py``; ``streaming.py`` → ``stream.py``; ``training.py`` → ``train.py``; ``persistence.py`` → ``store.py``; ``benchmark.py`` → ``bench.py``; ``implicit_diff.py`` → ``implicit.py``; ``base.py`` → ``check.py``; ``embeddings.py`` / ``visualize.py`` deleted.
  - ``helpers.py`` + ``utils.py`` → single ``math.py`` (``Math`` / ``GP``).
- Class renames (multi-word → single-word): ``LAKERRegressor`` deleted, ``LAKERCore`` → ``Core``, ``Attention`` → ``Exact``, ``NystromAttention`` → ``Nystrom``, ``RandomFeatureAttention`` → ``Fourier``, ``SparseAttention`` → ``Neighbors``, ``SKIAttention`` → ``Grid``, ``TwoScaleAttention`` → ``Hybrid``, ``SpectralAttention`` → ``Spectrum``, ``SpectrumShaper`` → ``Shaper``, ``PreconditionedConjugateGradient`` → ``PCG``, ``GradientDescent`` → ``Descent``, ``JacobiPreconditioner`` → ``Jacobi``, ``CCCPPreconditioner`` → ``CCCP``, ``AdaptivePreconditioner`` → ``Adaptive``, ``HyperparameterSearch`` → ``Search``, ``EmbeddingTrainer`` → ``Trainer``, ``ResidualCorrector`` → ``Corrector``, ``DistributedAttention`` → ``Distributed``, ``Helpers`` → ``Math``, ``GPSurrogate`` → ``GP``, ``BaseHelpers`` → ``Bench``, ``SolverBenchmark`` → ``SolveBench``, ``BaselineBenchmark`` → ``BaseBench``, ``ModelPersistence`` → ``Store``, ``Visualizer`` → ``Plot``, ``Base`` → ``Check``.
- Method renames on ``Laker``: ``fit_with_search`` → ``search``, ``fit_with_bo`` → ``bayes``, ``partial_fit`` → ``update``, ``fit_path`` → ``path``, ``fit_continuation`` → ``continuation``, ``fit_learned_embeddings`` → ``learn``, ``fit_residual_corrector`` → ``correct``, ``fit_uncertainty_aware`` → ``calibrate``, ``fit_bilevel`` → ``bilevel`` (also ``tune``), ``predict_variance`` → ``variance``. Hyperparameter: ``lambda_reg`` → ``lam``, ``num_probes`` → ``num`` (param only — internals use ``num_probes``).
- ``k_nm`` → ``cross_kernel``, ``k_mm`` → ``landmark_kernel``, ``k_mm_chol`` → ``landmark_cholesky``, ``k_nm_kmm_inv`` → ``landmark_projection`` in Nyström. ``self.q`` → ``self.basis``, ``self.r`` → ``self.tri_factor``, ``self.n`` → ``self.size``, ``self.nr`` → ``self.num_probes``, ``qr_r`` → ``tri_factor``, ``reg_scale`` → ``reg``, ``op_probes`` → ``probed``, ``probe_norms`` → ``norms``, ``norm_probes`` → ``unit``, ``iso_shr`` → ``iso_shrunk``, ``wrwt`` → ``quadratic``, ``rwk`` → ``weighted_factor``, ``wk`` → ``weights``, ``m_buf`` → ``matrix_buf``, ``fg_buf`` → ``f_gamma_buf``, ``sh_buf`` → ``shrunk_buf``, ``eye`` → ``identity``, ``scaled`` → ``scaled_proj``, ``inv_m_r`` → ``inv_proj``, ``denoms`` → ``denominators``, ``inv_sqrt`` → ``inv_sqrt_iso`` in CCCP. ``k_nm`` etc. also renamed in other kernels; ``self.m`` → ``self.num_landmarks`` (Nyström); ``self.k`` → ``self.num_neighbors`` (Neighbors); Grid ``size`` constructor param renamed to ``grid_size`` to free ``self.size`` for the matrix dimension.
- ``Search.Search.Search`` (the static method wrapper alias) removed in favour of the public ``Search.grid`` and ``Search.bayes`` instance methods.
- ``Laker.tune`` is now an alias for ``Laker.bilevel`` (the previous inline grid search has been removed).

### Fixed
- **Nyström matvec was using the wrong matrix product.** The
  approximate ``K_approx @ x`` was being computed as
  ``K_nm K_mm^{-1} K_nm^{-1}`` instead of the correct
  ``K_nm K_mm^{-1} K_nm^T``. Fixed; ``matvec`` and ``dense @ v`` now
  match to round-off error.
- **Grid ``diag`` used only the diagonal of ``K_grid``** instead of
  the full quadratic form ``Σ_{j,k} W_ij W_ik K_grid_jk``. Fixed.
- **CDF approximation gave values > 1 for negative x** (the
  Abramowitz-Stegun 7.1.26 formula was applied with the wrong sign
  for ``x < 0``). Fixed via ``torch.where``.
- **Broken benchmark imports.** ``benchmarks/baseline.py``,
  ``benchmarks/run.py``, ``benchmarks/reproducible.py`` imported
  non-existent ``Kernel`` and ``Solve`` classes. All three now run
  against the real ``kernel.Exact`` / ``solve.PCG``.
- **Broken example scripts.** ``flow.py`` / ``learn.py`` / etc. used
  legacy parameter names (``path_loss_exponent``, ``grid_size``,
  ``embedding_dim``, ``regularization``, ``probes``). Renamed.
- **``Backend.compile`` and ``Backend.autocast`` shadowed the
  class attributes of the same name** on first access. Renamed the
  attributes to ``compile_mode`` and ``autocast_on``.
- **``Backend.chunk_set`` stored raw megabytes** instead of bytes.
  Fixed to multiply by ``1024 * 1024``.
- **Stream / train / bilevel accessed removed class attributes**
  (``self.coef`` vs ``self.coef_``, ``self.encoder`` vs
  ``self.encoder_``, etc.). Fixed across all call sites.
- **Set / get-params round-trip dropped the ``encoder`` argument**
  because it was read as ``self.encoder`` (renamed to
  ``self._init_encoder`` for parameter storage).
- **``Store.load`` wrote ``size=`` to the ``Grid`` kernel** but the
  parameter is now ``grid_size``. Fixed.
- **CI: ``lint`` and ``test`` jobs had broken matrix and conditionals**
  (``if: matrix.os == '…' && matrix.python-version == '…'`` referenced
  keys that did not exist in the matrix). Consolidated into a single
  ``test`` job that runs on ``[ubuntu-latest, macos-latest]`` ×
  ``[3.12, 3.13]`` using ``ruff`` (replacing black + isort + flake8 +
  mypy from the duplicate ``lint`` job). Wheel build mirrors the
  test matrix. Dropped the dead ``docs`` job (no sphinx config).
  Restricted branches to ``[master]`` (the only branch in the repo).
- **``pyproject.toml`` had a duplicate ``Issues`` URL key** and
  referenced a non-existent ``_visualize_impl.py``. Both removed.
- **``.gitignore`` had 154 lines of irrelevant entries** (Django /
  Flask / Scrapy / Celery / Sage / PyBuilder / IPython / pyenv /
  pipenv / poetry / pdm / PEP 582 / Sphinx / PyCharm). Trimmed to 50
  lines of relevant entries.
- **``mypy laker/`` failed with a module-name collision** caused by
  ``mypy_path = "laker"`` in ``pyproject.toml`` (which made ``laker.math``
  discoverable as both ``laker.math`` and ``math``). Removed the
  setting and bumped ``python_version`` to ``3.12`` to match the CI
  matrix (numpy 2.5+ stubs require 3.12 syntax). Removed the
  unused ``module = ["tests.*", "examples.*", "benchmarks.*"]``
  override.
- **``Core.kernel()`` method was assigned over by ``self.kernel = kernel``
  in ``__init__``**, leaving the method undiscoverable. Removed the
  dead alias; ``Core.build_kernel`` is the only public entry point.
- **``Kernel`` protocol declared ``n`` but every kernel implementation
  used ``self.size``.** Aligned the protocol with the implementation.
- **``Core.build_prec`` returned the wrong union type** because mypy
  couldn't narrow ``prec`` past the early-return branch. Renamed the
  second-branch local to ``cccp`` so each branch has its own inferred
  type.
- **``cast(Nystrom, op).m`` and ``cast(Neighbors, op).k`` accessed
  non-existent attributes.** Fixed to ``num_landmarks`` and
  ``num_neighbors``.
- **``model.iters``, ``model.encoder`` (no underscore) were used in
  ``stream.py``, ``search.py``, and ``bilevel.py`` but only the
  underscored versions existed.** Fixed; the fitted-state attribute
  is now ``model.iters`` / ``model.encoder`` (no underscore).
- **``GP.bounds`` had no type annotation** so mypy could not infer
  it. Annotated as ``np.ndarray``.
- **``torch.sparse_coo_tensor`` was renamed to ``torch.sparsecoo_tensor``
  by the ``_coo`` → ``coo`` replace-all.** Restored.

### Tests
- Deleted 31 legacy test files (smoke-only, duplicate, mocking the
  unit under test, asserting on NaN tautologies).
- Wrote 21 new test files with 310 real-assertion tests:
  ``test_math``, ``test_check``, ``test_data``, ``test_embed``,
  ``test_backend``, ``test_kernel``, ``test_solve``, ``test_prec``,
  ``test_distributed``, ``test_corrector``, ``test_implicit``,
  ``test_store``, ``test_model``, ``test_search``, ``test_bilevel``,
  ``test_train``, ``test_stream``, ``test_plot``, ``test_bench``,
  ``test_executor``, ``test_cli``. Each test exercises real
  behaviour: closed-form math, finite / non-NaN output, shape
  checks, deterministic-seed reproducibility, save/load round-trips,
  and edge cases (empty input, NaN, single sample, threshold
  exceedance).
- ``Data.split`` is now deterministic given a seed and is exercised
  by ``test_data.py::TestSplit``.

### Documentation
- See "Added" for ``docs/`` content.
- ``CHANGELOG.md`` has been re-organised: previous version history
  preserved below; this release documents every refactor.
- ``README.md`` was rewritten to match the actual API. The previous
  copy invented parameter names (``embedding_dim``, ``regularization``,
  ``probes``, ``pcg_max_iter``, ``preconditioner``, ``epsilon``,
  ``base_rho``, ``rebuild_freq``, ``embedding_module``), referenced
  non-existent modules (``preconditioner.py``, ``fit.py``,
  ``helpers.py``, ``__main__.py``, ``base.py``, ``embed.py``,
  ``solve.py``, ``search.py``, ``stream.py``, ``implicit.py``,
  ``plot.py``, ``data.py``, ``backend.py``), non-existent example
  scripts (``examples.basic``, ``examples.large``), non-existent
  benchmark entry points, and a fabricated "Performance" section
  with made-up numbers. It also referenced removed tools
  (``black`` / ``flake8`` / ``isort``) instead of the actual
  ``ruff``, and quoted a Python 3.9 minimum that does not match
  ``pyproject.toml``'s ``"requires-python = '>=3.10,<3.14'"``. The
  rewritten README drops all of that and aligns with the current
  codebase while keeping the correct Tao & Tan (2026) arXiv
  reference at ``https://arxiv.org/abs/2604.25138``.

### Atomic commits in this release

| Commit | Date (UTC+05:30) | Subject |
|--------|------------------|---------|
| `13ba33a` | 2026-07-31 19:30:00 +05:30 | docs(readme): rewrite to match actual API and drop stale/fabricated content |
| `a8324ef` | 2026-07-31 19:00:00 +05:30 | fix(mypy): drop all underscore prefix/suffix; fix Core/kernel/attribute conflicts |
| `48c2f93` | 2026-07-31 16:00:12 +05:30 | refactor(api): single-word naming across laker/, expand tests to 310 real-assertion tests |
| `7d6c4ee` | 2026-07-31 17:00:00 +05:30 | chore: clean up .gitignore (154 → 50 lines) |
| `c63c50c` | 2026-08-01 23:00:00 +05:30 | docs(changelog): record cross-map baseline and SOTA positioning |
| `42ea458` | 2026-08-01 22:45:00 +05:30 | docs: document cross-map baseline and position LAKER vs learned CNNs |
| `85183e5` | 2026-08-01 22:30:00 +05:30 | feat(examples): add --cross-map stationary cross-map baseline |
| `a95ec49` | 2026-08-01 21:00:00 +05:30 | docs: register paper and scalable examples in README and docs |
| `d67701d` | 2026-08-01 20:30:00 +05:30 | feat(examples): add scalable.py + scalable_data.py for UCF-50K full-sweep |
| `06300eb` | 2026-08-01 20:00:00 +05:30 | feat(examples): add paper.py reproducing LAKER paper Section V |
| `<this>`  | 2026-07-31 18:00:00 +05:30 | docs: add docs/ (guides, api, algorithms, examples) and fix CI |
| `<prev>` | 2026-07-12 13:26:35 +05:30 | docs: comprehensive module/class/method docstrings across laker/ |
| `<prev>` | 2026-07-12 13:26:39 +05:30 | docs: add module docstrings to benchmarks/ and examples/ |
| `<prev>` | 2026-07-12 13:26:54 +05:30 | docs: standardise test docstrings and update renamed references |
| `<prev>` | 2026-07-12 13:26:58 +05:30 | docs: expand docstring conventions in CONTRIBUTING.md and add Python 3.13 classifier |
| `<prev>` | 2026-07-12 13:30:00 +05:30 | docs: rewrite README.md with reference-style structure |
| `<prev>` | 2026-07-12 13:35:00 +05:30 | docs: replace print() with comments in docstring examples |

## [0.4.0] - 2026-05-04

### Added
- **Spectral-Shaped Attention Kernel**: Added `SpectralAttention` and `SpectrumShaper`. Replaces plain `exp(EE^T)` with a learned matrix function `K = U diag(exp(g(sigma_i^2))) U^T` where `g` is a monotone spline. Improves conditioning and injects an inductive bias directly on the spectrum. Controlled via `kernel_approx="spectral"` and `spectral_knots`.
- **Bilevel Hyperparameter Learning**: Added `BilevelOptimizer` and `implicit_diff.hypergradient`. Computes hypergradients of validation loss through the PCG fixed-point using the adjoint method. Access via `LAKERRegressor.fit_bilevel(x_train, y_train, x_val, y_val)`.
- **Uncertainty-Aware Training**: Added `fit_uncertainty_aware()` which trains embeddings with a negative log-likelihood + calibration penalty objective: `L = NLL(y | mu, sigma^2) + beta * calibration_penalty`. Uses differentiable predictive mean and variance.
- **Residual Corrector**: Added `ResidualCorrector` and `fit_residual_corrector()`. A tiny MLP (2 layers, 32 hidden units, dropout) trained on `y - y_hat_laker` with validation-split early stopping. Captures local misspecification without destabilising the core solver.
- **Two-Scale Kernel**: Added `TwoScaleAttention` combining a global Nyström low-rank term with a local sparse k-NN graph: `K = alpha * K_global + (1 - alpha) * K_local`. Controlled via `kernel_approx="twoscale"`, `num_landmarks`, and `k_neighbors`.
- **Continuation Schedule**: Added `fit_continuation()` which solves a sequence of decreasing `lambda_reg` values with warm-started PCG and optional preconditioner reuse. Useful for tracking a stable regularisation path to sharper solutions.
- **Leverage-Score Landmark Selection**: Nyström kernels now support `landmark_method="leverage"` for ridge leverage score sampling from a pilot kernel. Often gives lower approximation error than greedy k-means++ selection.
- **Adaptive Preconditioner**: Added `AdaptivePreconditioner` with spectrum-aware probe distribution (power-iteration-biased + orthogonalised blocks). Select via `preconditioner_strategy="adaptive"`.
- **Refactored Architecture**: Split `LAKERRegressor` internals into focused helper classes: `LAKERCore` (kernel/solve/predict), `EmbeddingTrainer` (learned embeddings, residual corrector, bilevel, uncertainty-aware), `HyperparameterSearch` (grid/BO), `ModelPersistence` (save/load), and `StreamingUpdater` (partial_fit, continuation).
- **Math Reliability Tests**: Added `tests/test_math_correctness.py` with 18 tests covering TwoScale kernel linearity, leverage score properties, continuation monotonicity, implicit differentiation finite-difference verification, exact variance formula matching, and preconditioner linearity.
- **Integration Tests**: Added `test_twoscale_integration`, `test_continuation_integration`, and expanded spectral kernel tests.
- **TODO Completion**: All 6 deferred items from `TODO.md` are now implemented.

### Changed
- **Code standardization**: Full compliance with the Google Python Style Guide. Reformatted to 80-character line length via `black`. Fixed all `ruff` docstring violations (D107, D102, D301, D401, D205, D413). Fixed import ordering (I001). Removed unused imports (F401) and unused variables (F841).
- **SpectrumShaper defaults**: Changed default `raw_weights` from `0.0` to `-10.0` and `raw_slope` from `0.0` to `-2.35` to prevent `softplus(0)=0.693` from causing spectrum overflow (values of `1e21–1e125`) and PCG divergence.
- **StreamingUpdater**: `fit_path()` now stores the final fitted state (`embeddings`, `kernel_operator`, `preconditioner`, `alpha`) on the regressor so that `fit_continuation()` produces a model ready for prediction.
- **SKI `to_dense()`**: Fixed a shape mismatch bug where `torch.eye(n)[grid_indices]` was used instead of direct index assignment.

### Fixed
- **Broken imports**: Fixed benchmark and example imports referencing deleted modules (`laker.low_rank_kernels`, `laker.ski_kernels`, `laker.sparse_kernels`).
- **Dead code**: Removed unused `self.lambda_vec = None` in `Attention` and unused local `kernel_mv` in `laker/benchmark.py`.

### Added (from prior release)
- **Documentation**: Added `docs/patterns.md` documenting the Executor pattern, Class + Convenience Wrapper convention, naming rules, and logging requirements.
- **Module structure**: Added `benchmarks/__init__.py` and `examples/__init__.py` so benchmarks and examples can be run as modules (`python -m benchmarks.reproducible`, etc.).

## [0.3.0] - 2026-04-30

### Changed
- **Math simplification**: Removed redundant `inv_isotropic_coef` and `r_column_norms_sq` terms from CCCP probe denominator because QR-normalised probes have unit column norms. This eliminates numerical drift and an unnecessary buffer.
- **Memory efficiency**: Replaced explicit `factored_inverse = V @ diag(1/eig) @ V.T` with basis-scaled matmul (`vtr.T @ scaled_vtr`) in CCCP, avoiding an `O(N_r^3)` allocation per iteration.
- **Adaptive chunking**: `Attention.matvec` now auto-selects between fast 1-D chunking and full 2-D tiling based on a 64 MB memory heuristic, bounding peak memory to `O(chunk_size^2)` instead of `O(chunk_size * n)`.
- **Chunked prediction**: `LAKERRegressor.predict` now supports 2-D tiled kernel evaluation for large query sets, preventing `O(m*n)` memory blow-up.
- **In-place operations**: Replaced materialising `torch.exp` intermediates with `torch.exp(..., out=...)` in `matvec`, `to_dense`, and `kernel_eval`.
- **Buffer reuse**: Pre-allocated `factored_matrix`, `f_gamma_q_basis`, and `shrunken_f_gamma` buffers in CCCP loop to reduce GC pressure.
- **Default dtype**: Switched default from `float64` to `float32` for ML ecosystem compatibility.
- **Default tolerance**: Relaxed default `pcg_tol` from `1e-10` to `1e-6` (appropriate for float32).
- **Condition number estimation**: Reduced power iterations from 20 to 10 and inverse CG steps from 10x200 to 5x50, since the preconditioned system is well-conditioned by design.
- **Overflow guard**: Added `_exp_safe` helper with dtype-aware clamping (`80.0` for float32, `700.0` for float64) before exponentiation to prevent silent overflow to `inf` in the attention kernel.
- **Optimised grid search**: `fit_with_search` now computes embeddings once and reuses them across all trials, giving a 3-5x speedup. It also catches specific exceptions (`RuntimeError`, `ValueError`) and raises a clear error if all trials fail.
- **PCG in-place updates**: Replaced `p = z + beta * p` with `p.mul_(beta).add_(z)` to eliminate one tensor allocation per iteration in both 1-D and 2-D solves.
- **Thread-safety documentation**: Added a docstring warning to `PositionEmbedding` that its temporary manipulation of the global PyTorch RNG is not thread-safe.

### Added
- **Mixed-precision training**: `LAKERRegressor` now supports `embedding_dtype` parameter. Embeddings can be computed in `float16`/`bfloat16` and cast to the solver dtype, halving embedding memory.
- **Low-rank kernel approximations**: Added `NystromAttention` (greedy landmark selection, `O(n*m)` matvec) and `RandomFeatureAttention` (RFF with deterministic seed, `O(n*r)` matvec). Controlled via `kernel_approx` parameter in `LAKERRegressor`.
- **Sparse k-NN kernel**: Added `SparseAttention` in `laker/sparse_kernels.py`. Euclidean-distance k-NN graph with automatic symmetrisation and diagonal-dominance enforcement guarantees positive definiteness. Storage is `O(n*k)` and matvec cost is `O(n*k)`. Use `kernel_approx="knn"`.
- **Predictive variance / uncertainty quantification**: Added `LAKERRegressor.predict_variance(x)` for kernel ridge regression. Exact kernels use batched PCG solves with the learned preconditioner; RFF uses a closed-form Woodbury identity for near-instant variance.
- **Regularization path**: Added `LAKERRegressor.fit_path(lambda_reg_grid)` that fits a sequence of regularisation strengths with warm-started PCG (largest `lambda` first). Embeddings and the preconditioner are built once, giving near-linear cost in the number of grid points.
- **Validation-based grid search**: Added `fit_with_search` method that splits data into train/val, searches over `lambda_reg`, `gamma`, and `num_probes`, and retrains the best configuration on the full dataset.
- **Reproducible benchmark suite**: Added `benchmarks/reproducible_benchmarks.py` with fixed seeds, multiple trials, and mean/std reporting. Added `benchmarks/compare_approximations.py` for exact vs low-rank comparison.
- **Test coverage**: Added `tests/test_low_rank_kernels.py` and `tests/test_advanced_features.py` covering mixed-precision, Nyström, RFF, grid search, fit_path, predict_variance, and k-NN kernels.

### Fixed
- **Benchmark bug**: `benchmark_full_fit` was returning `kernel_operator.n` instead of actual PCG iteration count.
- **PCG robustness**: Replaced `+ 1e-16` fudge factors with explicit breakdown detection (`p^T A p <= 0`). Residual replacement is now **disabled by default** because it causes catastrophic cancellation in float32; users can opt in via `restart_freq` for high-precision float64 runs.
- **PCG batch support**: Solver now accepts 2-D RHS using vectorised column-wise dot products.
- **PCG 1-D fast path**: Restored scalar `torch.dot` for single-RHS solves, avoiding the tensor-broadcasting overhead introduced by the batch path.
- **PositionEmbedding determinism**: MLP layers now use PyTorch's default `kaiming_uniform_` init with the global RNG temporarily seeded, ensuring deterministic behaviour while preserving the weight distribution of the original release.
- **Custom embedding save/load**: Custom embedding modules are now importable via `tests/custom_embed.py`, making save/load round-trips work correctly.
- **RFF kernel_eval inconsistency**: `RandomFeatureAttention.kernel_eval` now returns the proper RFF feature-map approximation (`phi_x @ phi_y.T / r`) instead of the exact kernel.
- **Nyström overflow guard**: `NystromAttention._compute_kernel_matrix` and `kernel_eval` now use `_exp_safe` to prevent overflow.
- **predict overflow guard**: `LAKERRegressor.predict`'s 2-D tiled path now uses `_exp_safe` instead of raw `torch.exp`.
- **predict low-rank consistency**: `LAKERRegressor.predict` no longer falls back to exact-kernel 2-D tiling for low-rank approximations, ensuring predictions are consistent with the fitted model.
- **SKI kernel approximation**: Added `SKIAttention` with product grid and multilinear interpolation. Matvec cost is `O(n * grid_size)` instead of `O(n^2)`. Controlled via `kernel_approx="ski"` and `grid_size` parameter.
- **Bayesian hyperparameter optimisation**: Added `LAKERRegressor.fit_with_bo` with lightweight GP surrogate (RBF kernel, log-scale) and Expected Improvement acquisition. No external dependencies. Typically converges in 10-15 evaluations.
- **Streaming / online learning**: Added `LAKERRegressor.partial_fit(x_new, y_new)` for incremental updates. Enlarges the system, warm-starts PCG from the previous alpha, and rebuilds the preconditioner when a configurable threshold is reached.
- **Learned embeddings**: Added `LAKERRegressor.fit_learned_embeddings(x, y, lr, epochs)` which optimises `PositionEmbedding` MLP weights end-to-end via Adam on the residual loss, backpropagating through the differentiable kernel operator. Preconditioner is rebuilt periodically (`rebuild_freq`).
- **Multi-GPU distributed matvec**: Added `DistributedAttention` in `laker/distributed_kernels.py`. Shards embeddings across available CUDA devices, computes local matvecs, and gathers results. Falls back to single-device wrapper when only one GPU is available.
- **Autograd-safe exponential**: `_exp_safe` now detects `requires_grad=True` and returns `torch.exp(clamped)` without the in-place `out=` form, enabling backprop through the kernel operator during learned-embedding training.

## [0.0.1] - 2026-04-29

### Added
- Initial release of LAKER (Learning-based Attention Kernel Regression).
- `LAKERRegressor`: sklearn-compatible estimator for attention kernel regression.
- `Attention`: matrix-free exponential attention kernel with chunked evaluation.
- `CCCPPreconditioner`: learned data-dependent preconditioner via shrinkage-regularised CCCP with factored `O(N_r^3)` implementation.
- `PreconditionedConjugateGradient`: standard PCG solver with convergence monitoring.
- `PositionEmbedding`: deterministic position-driven embedding module.
- Baseline solvers: `GradientDescent` and `JacobiPreconditioner`.
- Synthetic radio-field generators and visualisation utilities.
- Comprehensive test suite with 14+ tests.
- Documentation and usage examples.

[Unreleased]: https://github.com/sachncs/laker/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/sachncs/laker/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/sachncs/laker/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/sachncs/laker/compare/v0.0.1...v0.3.0
[0.0.1]: https://github.com/sachncs/laker/releases/tag/v0.0.1
