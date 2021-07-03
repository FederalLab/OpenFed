from openfed.common.wrapper import Wrapper


def test_wrapper():
    wrapper = Wrapper()
    wrapper.add_pack_key(['a', 'b', 'c'])
    wrapper.add_pack_key(['d'])

    wrapper.add_unpack_key(['1', '2', '3'])

    wrapper.add_unpack_key('4')
