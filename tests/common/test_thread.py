from openfed.common.base import peeper
from openfed.common.thread import SafeTread


class MyThread(SafeTread):
    def safe_run(self):
        return None


def test_safe_thread():
    thread_pool = peeper.get_from_peeper('thread_pool')
    thread = MyThread()

    assert thread in thread_pool
    print(thread)

    thread.start()
    thread.join()

    assert thread not in thread_pool

    print(thread)
