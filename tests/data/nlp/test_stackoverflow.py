from openfed.data.nlp.stackoverflow import StackOverFlowNWP, StackOverFlowTP


def test_stack_overflow_nwp():
    try:
        StackOverFlowNWP(root='/tmp/', download=False)
    except FileNotFoundError as e:
        pass


def test_stack_overflow_tp():
    try:
        StackOverFlowTP(root='/tmp/', download=False)
    except FileNotFoundError as e:
        pass
