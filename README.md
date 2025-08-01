# RL-Based Hyperparameter Tuner

End‑to‑end prototype that trains a ResNet‑18 on CIFAR‑10 while a PPO agent dynamically adjusts learning rate.
Metrics are logged to PostgreSQL via PySpark and visualized with a Grafana dashboard.
A JAX `pmap` harness benchmarks an upper‑bound inference ceiling on multi‑GPU HPC nodes.
