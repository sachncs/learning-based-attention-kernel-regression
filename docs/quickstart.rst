Quick Start
===========

Requirements
------------

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.23.0

Installation
------------

Install from source in editable mode with development dependencies::

   git clone https://github.com/convexsoft/kernelSC.git
   cd kernelSC
   pip install -e ".[dev]"

Basic Usage
-----------

Fit a model to sparse wireless measurements and reconstruct a radio map::

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

Save and load::

   model.save("laker_model.pt")
   loaded = LAKERRegressor.load("laker_model.pt")

CLI Usage
---------

For batch workflows you can use the bundled command-line tool::

   laker fit --locations x_train.pt --measurements y_train.pt --output model.pt
   laker predict --model model.pt --locations x_test.pt --output y_pred.pt
