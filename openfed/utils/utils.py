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


openfed_title = '\033[0;34m<OpenFed>\033[0m'
openfed_class_fmt = openfed_title + \
    " \033[0;35m{class_name}\033[0m\n{description}\n"
