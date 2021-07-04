from openfed.data.vision.emnist import EMNIST

def test_emnist():
    try:
        EMNIST(root='/tmp/', download=False)
    except FileNotFoundError as e:
        pass