from openfed.common import peeper

class Hooks(object):
    def __init__(self):
        if peeper.api is not None: # type: ignore
            peeper.api.register_everything(self) # type: ignore