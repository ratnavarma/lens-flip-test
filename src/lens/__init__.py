import argparse
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from lens.flip import cli as flip_cli

    parser = argparse.ArgumentParser(prog="lens")
    sub = parser.add_subparsers(dest="command", required=True)
    flip_cli.register(sub)
    args = parser.parse_args(argv)
    return args.func(args)
