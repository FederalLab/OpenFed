# 如果DEBUG=True，那相关的程序会输出部分调试信息
# 会以更严格的方式，执行程序
DEBUG: bool = False


def debug():
    global DEBUG
    DEBUG = True


# 如果VERBOSE=True, 相关程序会输出一些日志
VERBOSE: bool = True


def verbose():
    global VERBOSE
    VERBOSE = True


def silence():
    global VERBOSE
    VERBOSE = False


DYNAMIC_ADDRESS_LOADING = True

# FIXME 修复from ..common import XXX 的时候，XXX并不是全局共享的问题。

# 是否开启动态地址加载功能。
# 如果开启动态地址加载功能，那么程序可以在运行过程中创建新的连接
# 这使得你的程序更加的灵活，但是你需要维护maintainer_lock的关系，
# 避免发生抢占造成的其他问题。
# 如果你选择关闭这个功能，那么你需要在程序的一开始就启动所有的机器
# 否则的话，程序将会一直等待


def disable_dynamic_address_loading():
    global DYNAMIC_ADDRESS_LOADING
    DYNAMIC_ADDRESS_LOADING = False


def is_dynamic_address_loading():
    return DYNAMIC_ADDRESS_LOADING
