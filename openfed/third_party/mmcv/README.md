# mmcv

An OpenFed extra module for open-mmlab's projects, such as mmdetection, mmdetection3d ...

## Usage

1. Define the config file

```python
# runtime settings
runner = dict(type='OpenFedRunner',
              max_epochs=36,
              runner_cfg=dict(
                  type='EpochBasedRunner'
              ),
              openfed_cfg=dict(
                  type='OpenFed',
                  role='openfed_leader', # openfed_follower
                  address_file='path/to/address.json', # use python -m openfed.tools.helper to get the address file
                  fed_optim_cfg=dict(type='fedavg', lr=1.0),
                  hook_cfg_list=[
                      dict(type='Aggregate', 
                           count=dict(train=-1), # the count to trigger the aggregate operation
                           checkpoint: str = None, # save aggregated model
                           max_version: int = -1, # terminate if max version is achieved
                           )
                  ],
              )
)
```

1. Import `openfed` at the start point of program.
