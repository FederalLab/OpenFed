# Optim

## FederatedOptimizer

:class:`FederatedOptimizer` wrapper an :class:`torch.optim.Optimizer`, and provide some necessary functions for federated learning. The simplest way to generate an federated optimizer is to use this wrapper like:

```python
sgd = optim.SGD(...)
# For aggregator
fed_sgd = FederatedOptimizer(sgd, role=openfed.aggregator)
# For collaborator
fed_sgd = FederatedOptimizer(sgd, role=openfed.collaborator)
```

FederatedOptimizer usually has different behaviors when it plays different roles.
It has two special functions, namely :func:`acg_step` and :func:`round`.

- `acg_step`: If you want to calculate some statistic metric of dataset with the downloaded model, you can implement here. This function will be called before the training phase.
- `round`: If you need to calculate some statistic metric of dataset with the trained model, you can implement here. This function will be called after the training phase.
