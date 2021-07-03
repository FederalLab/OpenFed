from openfed.common.thread import SafeTread
from openfed.common.base import peeper


class MyThread(SafeTread):
    def safe_run(self):
        return None


def test_safe_thread():
    thread_pool = peeper.get_from_peeper('thread_pool')
    thread = MyThread()

    assert len(thread_pool) == 1
    print(thread)

    thread.start()
    thread.join()

    assert len(thread_pool) == 0

    print(thread)