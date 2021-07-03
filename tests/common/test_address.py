from openfed.common.address import *
from openfed.common.base import peeper
from openfed.common.parser import parser


def test_address():
    address_pool = peeper.get_from_peeper('address_pool')
    # Build Address from parser
    args = parser.parse_args()
    address = Address(args=args)

    # Dump to file
    dump_address_to_file('/tmp/address.json', address)

    # Load from file
    load_address = load_address_from_file('/tmp/address.json')[0]

    assert address == load_address

    print(repr(address))
    print(str(address))

    assert address in address_pool
    assert load_address in address_pool
    assert id(address) == id(load_address)

    remove_address_from_pool(address)
    assert address not in address_pool
