import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--props')

args = parser.parse_args()

print(args.props)
