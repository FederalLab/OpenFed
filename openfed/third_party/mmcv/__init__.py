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
        "`openfed.third_party.mmcv` module requires `mmcv` package, but not found. "
        "You can install it from https://github.com/open-mmlab/mmcv.")
