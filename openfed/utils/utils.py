import random
from datetime import datetime

import numpy as np
import torch


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def seed_everything(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


openfed_class_fmt = "\033[0;34m<OpenFed>\033[0m \033[0;35m{class_name}\033[0m\n{description}\n"


def convert_to_list(x):
    return x if isinstance(x, (list, tuple)) or x is None else [x]
