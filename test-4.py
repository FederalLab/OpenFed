
from threading import Thread
import time

cnt = {}
cmt = {}
def A():
    cmt[time.time()] = time.time()
    print("A")

class AClass(Thread):
    def __init__(self):
        super().__init__()
        self.start()
    
    def run(self):
        A()

class Monitor(Thread):
    def __init__(self, cnt):
        super().__init__()
        self.cnt = cnt
        self.stopped = False
        self.start()

    def run(self):
        while not self.stopped:
            self.cnt[time.time()] = time.time()
            AClass()
            time.sleep(0.5)

    def manual_stop(self):
        """Provide a function to end it manually.
        """
        self.stopped = True

monitor = Monitor(cnt=cnt)

print(monitor.cnt)

monitor.manual_stop()

print(monitor.cnt)

print(cmt)