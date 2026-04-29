Examples
========

Worked Example from the Paper
-----------------------------

Reproduce the :math:`n = 3` example from Section IV-E of the paper::

   import torch
   from laker import LAKERRegressor

   e = torch.tensor(
       [[0.241, 0.444], [-0.336, 0.112], [-0.220, 0.353]],
       dtype=torch.float64,
   )
   y = torch.tensor([-66.14, -65.77, -77.30], dtype=torch.float64)

   class FixedEmbedding(torch.nn.Module):
       def forward(self, x):
           return e

   model = LAKERRegressor(
       embedding_dim=2,
       lambda_reg=0.1,
       gamma=0.0,
       embedding_module=FixedEmbedding(),
       dtype=torch.float64,
   )
   model.fit(torch.zeros(3, 2), y)

   print(model.alpha)  # [0.828, 5.404, -65.384]

Benchmarking
------------

Run a head-to-head comparison::

   from laker import benchmark_laker_vs_baselines
   from laker.data import generate_radio_field

   x = torch.rand(500, 2) * 100.0
   tx = torch.tensor([[30.0, 70.0], [70.0, 30.0]])
   pwr = torch.tensor([-40.0, -45.0])
   y_clean, y = generate_radio_field(x, tx, pwr)

   # Need embeddings first
   from laker import PositionEmbedding
   emb = PositionEmbedding(2, 10)
   e = emb(x)

   results = benchmark_laker_vs_baselines(e, y)
   for r in results:
       print(f"{r.name:20s}  iters={r.iterations:4d}  time={r.solve_time_seconds:.3f}s")
