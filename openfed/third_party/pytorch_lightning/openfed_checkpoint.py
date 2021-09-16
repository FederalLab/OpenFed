from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


class OpenFedCheckpoint(ModelCheckpoint):
    r"""OpenFedCheckpoint.

    OpenFedCheckpoint work flow:
        1. download model from server before train epoch start.
        2. compute the acg operation before training.
        3. call optim.round() after train epoch finished.
        4. upload model to server after finished.
    
    Users can register this callback to enable OpenFed training without necessary 
    to modify any source code.
    """
    def __init__(self):
        pass
