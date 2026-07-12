# Contributing to LAKER

Thank you for your interest in contributing to LAKER! We welcome bug reports,
feature requests, documentation improvements, and code contributions.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/laker.git
   cd laker
   ```
3. Install in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names with conventional prefixes:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feat/` | New feature | `feat/add-batch-predict` |
| `fix/` | Bug fix | `fix/pcg-convergence-nan` |
| `docs/` | Documentation | `docs/update-api-reference` |
| `refactor/` | Code restructuring | `refactor/split-solver-module` |
| `test/` | Adding tests | `test/add-nystrom-edge-cases` |
| `chore/` | Maintenance | `chore/update-dependencies` |

### Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/).
Use the following format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Code style changes (formatting, missing semi-colons, etc.) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Build process, CI, or auxiliary tool changes |

**Examples:**

```
feat(kernels): add batch prediction for multiple query sets
fix(pcg): handle NaN in residual computation
docs(readme): add benchmarks section
refactor(preconditioner): extract CCCP into standalone class
test(kernels): add edge cases for Nyström approximation
chore(ci): add Python 3.13 to test matrix
```

### Code Style

We use the following tools to maintain code quality:

- **Black** for formatting: `black laker/ tests/ examples/ benchmarks/`
- **isort** for import sorting: `isort laker/ tests/ examples/ benchmarks/`
- **flake8** for linting: `flake8 laker/ tests/ examples/ benchmarks/`
- **mypy** for type checking: `mypy laker/`

All of these are configured in `pyproject.toml` and `.flake8`.

Run all checks before submitting:

```bash
black --check laker/ tests/ examples/ benchmarks/
isort --check-only laker/ tests/ examples/ benchmarks/
flake8 laker/ tests/ examples/ benchmarks/
mypy laker/
```

Or auto-fix formatting:

```bash
black laker/ tests/ examples/ benchmarks/
isort laker/ tests/ examples/ benchmarks/
```

### Testing

Run the test suite with pytest:

```bash
pytest tests/ -v
```

For coverage reporting:

```bash
pytest tests/ --cov=laker --cov-report=html
open htmlcov/index.html
```

### Docstrings

All public and private classes, functions, and modules must have Google-style
docstrings. Every module (`.py` file) must start with a module-level docstring
explaining its purpose and relationship to the rest of the package.

**Module docstrings** should:
- Start with a one-line summary of the module's purpose.
- Describe the public API surface (classes, functions, constants).
- Reference the mathematical notation used (e.g. ``:math:`G = \exp(E E^T)```).
- List submodules and their roles when the module is a package entry point.

**Class docstrings** should:
- Start with a one-line summary.
- Include an `Attributes:` section for public state.
- Include an `Args:` section for constructor parameters.
- Reference the relevant paper sections or equations where applicable.

**Function/method docstrings** should:
- Start with a one-line summary.
- Include `Args:` for all parameters (including `self` is not needed).
- Include `Returns:` for non-`None` return values.
- Include `Raises:` for explicitly raised exceptions.
- Include `Side effects:` when the function mutates state or logs.
- Include `Example:` blocks for public API methods.
- Use `:math:` directives for mathematical notation.

**Test docstrings** should:
- Every test function must have a one-line docstring describing what it tests.
- Test classes must have a docstring describing the group of tests.
- Test modules must have a module-level docstring.

**Example convention:**

```python
def solve(operator, rhs, tol=1e-10):
    """Solve ``A x = b`` using preconditioned conjugate gradient.

    The solver iterates until ``||b - A x|| / ||b|| <= tol`` or
    ``max_iter`` is reached.

    Args:
        operator: Callable applying ``A`` to a vector of shape ``(n,)``.
        rhs: Right-hand side vector of shape ``(n,)``.
        tol: Relative residual tolerance. Default ``1e-10``.

    Returns:
        Solution tensor of shape ``(n,)``.

    Raises:
        RuntimeError: If the iteration encounters negative curvature.

    Example:
        >>> x = solve(A, b, tol=1e-8)
        >>> torch.linalg.norm(A(x) - b) / torch.linalg.norm(b)
        tensor(1.2345e-08)
    """
```

### Pull Request Process

1. Create a new branch for your feature or bugfix from `main`.
2. Make your changes and ensure all tests pass.
3. Update documentation and the changelog if applicable.
4. Open a pull request with a clear description of the changes.
5. Link the related issue if one exists.
6. Wait for CI to pass and request a review.

### Coding Standards

- Keep functions focused and under 50 lines where possible.
- Use type hints for all public function signatures.
- Prefer composition over inheritance.
- Use `logging` instead of `print` for diagnostic output.
- Never commit secrets, API keys, or credentials.
- Follow existing code patterns in the module you are modifying.

## Reporting Bugs

When reporting bugs, please include:

1. A minimal code snippet that reproduces the issue.
2. The expected vs actual behavior.
3. Your environment (OS, Python version, PyTorch version, CUDA version if applicable).
4. The full traceback if an exception occurred.

## Submitting Changes

1. Create a new branch for your feature or bugfix.
2. Make your changes and ensure tests pass.
3. Update documentation and the changelog if applicable.
4. Open a pull request with a clear description of the changes.

## Code of Conduct

Be respectful and constructive in all interactions. We are committed to providing
a welcoming and inspiring community for everyone. See
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.
