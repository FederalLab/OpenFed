from openfed.common.address import *
from openfed.common.address import _address_pool
from openfed.common.parser import parser


def test_address():
    # Build Address from parser
    args = parser.parse_args()
    address = Address(args=args)

    # Dump to file
    dump_to_file('/tmp/address.json', address)

    # Load from file
    load_address = load_from_file('/tmp/address.json')[0]

    assert address == load_address

    print(repr(address))
    print(str(address))

    assert address in _address_pool
    assert load_address in _address_pool
    assert id(address) == id(load_address)
