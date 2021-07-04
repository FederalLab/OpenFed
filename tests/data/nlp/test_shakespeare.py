from openfed.data.nlp.shakespeare import ShakespeareNCP, ShakespeareNWP


def test_shakespeare_ncp():
    try:
        ShakespeareNCP(root='/tmp/', download=False)
    except FileNotFoundError as e:
        pass

def test_shakespeare_nwp():
    try:
        ShakespeareNWP(root='/tmp/', download=False)
    except FileNotFoundError as e:
        pass
