import sys

from openfed.common.logging import logger
from openfed.federated.destroy import Destroy


def keyboard_interrupt_handle():
    """A handle to better deal with keyboard interrupt.
    """
    old_excepthook = sys.excepthook

    def new_hook(type, value, traceback):
        if type != KeyboardInterrupt:
            old_excepthook(type, value, traceback)
        else:
            logger.info("Keyboard interrupt captured by OpenFed.")
            logger.warning("Force Quit!")
            Destroy.destroy_all_in_all_world()
            logger.info("Exited.")
            old_excepthook(type, value, traceback)
    sys.excepthook = new_hook
