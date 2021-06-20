

class ConnectTimeout(Exception):
    """When we try to connect the other hand but the time has been timeout.
    """

    def __init__(self, msg=""):
        self.msg = msg

    def __str__(self):
        return self.msg
