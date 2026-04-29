# Contributing to LAKER

Thank you for your interest in contributing to LAKER! We welcome bug reports,
feature requests, documentation improvements, and code contributions.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/kernelSC.git
   cd kernelSC
   ```
3. Install in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Code Style

We use the following tools to maintain code quality:

- **Black** for formatting: `black laker/ tests/ examples/`
- **isort** for import sorting: `isort laker/ tests/ examples/`
- **flake8** for linting: `flake8 laker/ tests/ examples/`
- **mypy** for type checking: `mypy laker/`

All of these are configured in `pyproject.toml`.

### Testing

Run the test suite with pytest:

```bash
pytest tests/ -v
```

For coverage reporting:

```bash
pytest tests/ --cov=laker --cov-report=html
```

### Docstrings

All public classes and functions must have Google-style docstrings with:
- A one-line summary.
- A longer description if needed.
- `Args:` section for parameters.
- `Returns:` section for return values.
- `Examples:` section where applicable.

### Commit Messages

- Use the present tense ("Add feature" not "Added feature").
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
- Limit the first line to 72 characters or less.
- Reference issues and pull requests liberally after the first line.

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
a welcoming and inspiring community for everyone.
