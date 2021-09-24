# Api

:class:``API`` provides a simple wrapper of aggregator logistic.
After define an :class:``API``, you can use it in the backend:

```python
api.start()
api.join()
```

or run it on the main process:

```python
api.run()
```

When it runs on backend, you have to acquire the :attr:`openfed.federated.openfed_lock` before start your main process distributed training.
The :attr:`openfed.federated.openfed_lock` will lock the data-transfer operation at openfed, but has no influence on message-transfer.
Since openfed share the same module with torch to build communication between two process, we have to use this lock to control the data transfer operation.
