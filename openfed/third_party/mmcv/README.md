# mmcv

An OpenFed extra module for open-mmlab's projects, such as mmdetection, mmdetection3d ...

## Usage

1. Define the config file

```python
# For collaborator
runner = dict(
    type='EpochBasedRunner',
    max_epochs=36,
    role='collaborator',
    address_file='/path/to/address.json',
    constructor='OpenFedRunnerConstructor',
)

# For aggregator
runner = dict(
    type='EpochBasedRunner',
    max_epochs=36,
    role='aggregator',
    address_file='/path/to/address.json',
    constructor='OpenFedRunnerConstructor',
)
```

1. Import `openfed` at the start point of program.
