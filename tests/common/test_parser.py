from openfed.common.parser import parser


def test_parser():
    args = parser.parse_args()

    print(args.backend)
    print(args.init_method)
    print(args.port)
    print(args.world_size)
    print(args.rank)
    print(args.group_name)