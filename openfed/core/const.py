# @Author            : FederalLab
# @Date              : 2021-09-25 16:51:15
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:51:15
# Copyright (c) FederalLab. All rights reserved.
from typing import Any, Optional


class DefaultMaintainer(object):
    r'''Records global maintainer for auto-register.
    '''
    _default_maintainer: Optional[Any] = None
