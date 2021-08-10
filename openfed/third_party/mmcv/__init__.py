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
        "`openfed.mmcv` module requires `mmcv` package, but not found. "
        "Disable `openfed.mmcv`.")
