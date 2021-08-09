import warnings

try:
    import mmcv
    ENABLE_MMCV = True
except ImportError:
    ENABLE_MMCV = False

if ENABLE_MMCV:
    from .runner import *
else:
    warnings.warn(
        "Run without `MMCV` support. If you want to use `MMCV`, please build it from https://github.com/open-mmlab/mmcv.")
