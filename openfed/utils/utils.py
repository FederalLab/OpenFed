def green_color(msg: str) -> str:
    # 一般表示正确的状态
    return f"\033[0;32m+++ {msg} +++ \033[0m"


def red_color(msg: str) -> str:
    # 一般用于打印错误
    return f"\033[0;31mxxx {msg} xxx \033[0m"


def yellow_color(msg: str) -> str:
    # 一般用于表示准备要做
    return f"\033[0;33m>>> {msg} <<< \033[0m"
