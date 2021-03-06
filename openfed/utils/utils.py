# @Author            : FederalLab
# @Date              : 2021-09-25 16:54:58
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:54:58
# Copyright (c) FederalLab. All rights reserved.
import random
from datetime import datetime

import numpy.random as np_random
import torch


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def seed_everything(seed: int = 0):
    torch.manual_seed(seed)
    np_random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore


class _FMT:
    r"""Sets ``True`` if you want to show a colorful output on the screen."""
    color = False

    @property
    def openfed_title(self):
        return '\033[0;34m<OpenFed>\033[0m' if self.color else '<OpenFed>'

    @property
    def openfed_class_fmt(self):
        return self.openfed_title + (
            ' \033[0;35m{class_name}\033[0m\n{description}\n'
            if self.color else ' {class_name}\n{description}\n')


FMT = _FMT()
