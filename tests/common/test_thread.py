from openfed.common.thread import SafeTread, _thread_pool


class MyThread(SafeTread):
    def safe_run(self):
        return None


def test_safe_thread():
    thread = MyThread()

    assert len(_thread_pool) == 1
    print(thread)

    thread.start()
    thread.join()

    assert len(_thread_pool) == 0

    print(thread)