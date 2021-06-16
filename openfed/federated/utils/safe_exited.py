import sys
from datetime import datetime

import openfed
import openfed.utils as utils


def get_head_info() -> str:
    try:
        raise Exception
    except:
        f = sys.exc_info()[2].tb_frame.f_back
    return 'Time: %s, File: %s, Func: %s, Line: %s' % (str(datetime.now()), f.f_code.co_filename, f.f_code.co_name, str(f.f_lineno))


def safe_exited(msg: str):
    """用于在后台程序退出时，打印一些额外的信息。
    """
    if openfed.VERBOSE or openfed.DEBUG:
        print(utils.red_color("Exited"), f"{msg}")
