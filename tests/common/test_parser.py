from openfed.common.parser import parser


def test_parser():
    args = parser.parse_args()

    print(args.fed_backend)
    print(args.fed_init_method)
    print(args.fed_world_size)
    print(args.fed_rank)
    print(args.fed_group_name)