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
        "Run without MMCV support. If you want to using MMCV, please install build mmcv: https://github.com/open-mmlab/mmcv.")
