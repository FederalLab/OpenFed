# Copyright (c) FederalLab. All rights reserved.
from typing import Any, Optional


class DefaultMaintainer(object):
    r"""Records global maintainer for auto-register.
    """
    _default_maintainer: Optional[Any] = None
