from typing import List


class Wrapper(object):
    # 用来明确的指出当前的类当中，有哪些数据是用于打包的。
    package_key_list: List = None
    unpackage_key_list: List = None
