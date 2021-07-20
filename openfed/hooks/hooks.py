from openfed.common import peeper

class Hooks(object):
    def __init__(self):
        if peeper.api is not None:
            peeper.api.register_everything(self)