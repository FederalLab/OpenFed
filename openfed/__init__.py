# Refer [here](https://semver.org/) to learn more about Semantic Versioning Specification.
__version__ = "0.0.0"

# 如果DEBUG=True，那相关的程序会输出部分调试信息
# 会以更严格的方式，执行程序
DEBUG: bool = False


def debug():
    global DEBUG
    DEBUG = True


# 如果VERBOSE=True, 相关程序会输出一些日志
VERBOSE: bool = False


def verbose():
    global VERBOSE
    VERBOSE = True
