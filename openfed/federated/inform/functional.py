from typing import Any
import json
from abc import abstractmethod


class Collector(object):
    # 定义一些函数，用于计算一些状态信息，并且同步在客户端之间。
    @abstractmethod
    def collect(self) -> Any:
        """这个函数会自动被调用来收集相关的参数，它不接受任何参数输入。
        返回值应当是可以被序列化的基础数据类型。
        """
        raise NotImplementedError

    def __call__(self) -> str:
        output = self.collect()

        return json.dumps(output)

