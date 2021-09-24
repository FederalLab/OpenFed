import os


def _test_simulator():
    os.system('python -m openfed.tools.simulator --nproc 3 run.py')
