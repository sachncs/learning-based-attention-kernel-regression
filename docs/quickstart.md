# Quick Start

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.23.0

## Installation

Install from source in editable mode with development dependencies:

```bash
git clone https://github.com/sachncs/laker.git
cd laker
pip install -e ".[dev]"
```

## Basic Usage

Fit a model to sparse wireless measurements and reconstruct a radio map:

```python
import torch
from laker import LAKERRegressor

# 1000 sensor locations in a 100 x 100 m^2 area
x_train = torch.rand(1000, 2) * 100.0
y_train = torch.randn(1000)

model = LAKERRegressor(
    embedding_dim=10,
    lambda_reg=1e-2,
    gamma=1e-1,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
model.fit(x_train, y_train)

# Predict on a dense grid
x_test = torch.rand(2000, 2) * 100.0
y_pred = model.predict(x_test)
```

## Save and Load

Models can be serialised to disk and restored for later inference:

```python
model.save("laker_model.pt")
loaded = LAKERRegressor.load("laker_model.pt")
y_pred = loaded.predict(x_test)
```

## Kernel Approximations

For very large $n$, use a low-rank or sparse approximation via the `kernel_approx` argument:

```python
# Nyström low-rank approximation
model = LAKERRegressor(
    kernel_approx="nystrom",
    num_landmarks=200,
    embedding_dim=10,
)

# Random Fourier Features
model = LAKERRegressor(
    kernel_approx="rff",
    num_features=400,
    embedding_dim=10,
)

# Sparse k-NN
model = LAKERRegressor(
    kernel_approx="knn",
    k_neighbors=50,
    embedding_dim=10,
)

# Structured Kernel Interpolation (SKI)
model = LAKERRegressor(
    kernel_approx="ski",
    grid_size=1024,
    embedding_dim=10,
)
```

Each approximation trades accuracy for speed and memory. See [Theory](theory.md) for guidance on choosing an approximation.

### Spectral-shaped kernel

```python
model = LAKERRegressor(
    embedding_dim=10,
    kernel_approx="spectral",
    spectral_knots=5,
    dtype=torch.float64,
)
model.fit(x_train, y_train)
```

### Two-scale kernel

```python
model = LAKERRegressor(
    embedding_dim=10,
    kernel_approx="twoscale",
    num_landmarks=100,
    k_neighbors=30,
)
model.fit(x_train, y_train)
```

### Bilevel hyperparameter learning

```python
model = LAKERRegressor(embedding_dim=10, dtype=torch.float64)
model.fit(x_train, y_train)
model.fit_bilevel(
    x_train, y_train, x_val, y_val,
    lr=1e-3, epochs=20, patience=5,
)
```

### Uncertainty-aware training

```python
model = LAKERRegressor(embedding_dim=10, dtype=torch.float64)
model.fit(x_train, y_train)
model.fit_uncertainty_aware(
    x_train, y_train,
    lr=1e-3, epochs=50, beta=0.1, patience=5,
)
var = model.predict_variance(x_test)
```

### Residual corrector

```python
model = LAKERRegressor(embedding_dim=10)
model.fit(x_train, y_train)
model.fit_residual_corrector(
    x_train, y_train, epochs=200, patience=10,
)
```

### Continuation schedule

```python
model = LAKERRegressor(embedding_dim=10, dtype=torch.float64)
model.fit_continuation(
    x_train, y_train,
    lambda_max=1.0, lambda_min=1e-2, n_stages=5,
)
```

## CLI Usage

For batch workflows you can use the bundled command-line tool:

```bash
laker fit --locations x_train.pt --measurements y_train.pt --output model.pt
laker predict --model model.pt --locations x_test.pt --output y_pred.pt
```

## Hyperparameter Search

LAKER includes two built-in hyperparameter tuning strategies:

### Grid Search

```python
model = LAKERRegressor(embedding_dim=10, verbose=True)
model.fit_with_search(x_train, y_train)
```

Splits the data into train/validation, tries combinations of `lambda_reg`, `gamma`, and `num_probes`, and refits on the full dataset with the best configuration.

### Bayesian Optimisation

```python
model = LAKERRegressor(embedding_dim=10, verbose=True)
model.fit_with_bo(x_train, y_train, n_calls=15)
```

Uses a lightweight Gaussian Process surrogate with Expected Improvement acquisition. Typically requires 10–15 evaluations versus 27 for a full $3 \times 3 \times 3$ grid search.

## Next Steps

- Learn the [mathematics](theory.md) behind the CCCP preconditioner.
- Browse the [API reference](api.md) for detailed class documentation.
- Run the [worked examples](examples.md) to reproduce paper results.
