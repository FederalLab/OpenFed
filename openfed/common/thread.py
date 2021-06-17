from abc import abstractmethod
from threading import Thread
from typing import Dict, final

import openfed
import openfed.utils as utils


class SafeTread(Thread):
    # 用来控制整个线程是否退出。
    stopped: bool

    def __init__(self):
        super().__init__(name="OpenFed SafeTread")
        self.stopped = False

        # 为了不影响主程序退出，设置为True
        # 因此，任何关键的动作，都不允许在Thread里面实现
        # 比如保存模型等。
        # Thread只是拿来做一些轻量级的辅助任务！
        self.setDaemon(True)

        # 自动注册到全局的pool中
        _thread_pool[self] = utils.time_string()

        if openfed.VERBOSE or openfed.DEBUG:
            print(utils.yellow_color("New Thread"), self)

    @final
    def run(self):
        """
            子类通过实现safe_run方法来实现相关的函数功能。
        """
        self.safe_run()

        self.safe_exit()

    def safe_exit(self):
        if openfed.VERBOSE or openfed.DEBUG:
            time_string = _thread_pool[self]
            print(utils.red_color("Exited"), self, f"Created at {time_string}")
        del _thread_pool[self]

    def __repr__(self) -> str:
        return "<OpenFed> SafeTread"

    @abstractmethod
    def safe_run(self):
        """实现你的相关代码
        """
        ...

    def manual_stop(self):
        self.stopped = True


# 记录了这个线程创建的时间。
_thread_pool:  Dict[SafeTread, str] = {}