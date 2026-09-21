from argparse import ArgumentParser, SUPPRESS
from argparse import _SubParsersAction


def create_parser() -> ArgumentParser:

    shared = ArgumentParser(add_help=False, argument_default=SUPPRESS)
    shared.add_argument("--symbol", "-s", help="Product code or contract ticker, e.g. MES or MESU6")
    shared.add_argument("--resolution", "-r", type=str, help=(
            "The size of each aggregate candle, specified as a number "
            "followed by a unit: sec, min, hour, session, week, month, quarter, or year."
            )
    )

    parser = ArgumentParser(parents=[shared], argument_default=SUPPRESS)

    subparsers = parser.add_subparsers(dest="command", required=True)

    create_fetch_parser(subparsers, shared)
    create_preprocess_parser(subparsers, shared)
    create_model_parser(subparsers)
    create_fees_parser(subparsers)

    return parser
def create_fetch_parser(subparsers: _SubParsersAction, common_args: ArgumentParser | None = None) -> ArgumentParser:
    if common_args is None:
        common_args = ArgumentParser(add_help=False)

    fetch_parser: ArgumentParser = subparsers.add_parser("fetch", argument_default=SUPPRESS)

    fetch_subparsers: _SubParsersAction = fetch_parser.add_subparsers(dest="fetch_command", required=True)

    latest_parser: ArgumentParser = fetch_subparsers.add_parser("latest", parents=[common_args], argument_default=SUPPRESS)

    lookback_parser: ArgumentParser = fetch_subparsers.add_parser("lookback", parents=[common_args], argument_default=SUPPRESS)
    lookback_parser.add_argument("--period", choices=["days","weeks","months","years"], required=True)
    lookback_parser.add_argument("--depth", required=True, type=int)

    range_parser: ArgumentParser = fetch_subparsers.add_parser("range", parents=[common_args], argument_default=SUPPRESS)
    range_parser.add_argument("--begin", required=True)
    range_parser.add_argument("--end", required=True)

    train_parser: ArgumentParser = fetch_subparsers.add_parser("train", parents=[common_args], argument_default=SUPPRESS)
    train_parser.add_argument('--years', '-y', type=int, help="Number of years to from the current year to collect.")
    train_parser.add_argument('--contract', '-c', type=str)
    train_parser.add_argument('--hold-out', '-H', help="Number of months to hold out for model context.")
    
    return fetch_parser

def create_preprocess_parser(subparsers: _SubParsersAction, common_args: ArgumentParser | None = None) -> ArgumentParser:
    if common_args is None:
        common_args = ArgumentParser(add_help=False)

    preproc_parser: ArgumentParser = subparsers.add_parser("preprocess", parents=[common_args], argument_default=SUPPRESS)
    preproc_parser.add_argument("--train", "-T", action="store_true", help="Preprocess training data.")

    return preproc_parser

def create_model_parser(subparsers: _SubParsersAction) -> None:
    parser: ArgumentParser = subparsers.add_parser("model", argument_default=SUPPRESS)
    parser.add_argument("--ticker", '-t', type=str)
    parser.add_argument("--pred_length", "-p", type=int)
    parser.add_argument("--store_weights", "-s", action="store_true")
    parser.add_argument("--eval", "-e", action="store_true")


def create_fees_parser(subparsers: _SubParsersAction) -> None:
    parser: ArgumentParser = subparsers.add_parser("fees")

    cost_group  = parser.add_argument_group("Trading costs", argument_default=SUPPRESS)
    cost_group.add_argument("--entry-fee", type=float, help="Entry fee per contract")
    cost_group.add_argument("--exit-fee",  type=float, help="Exit fee per contract")
    cost_group.add_argument("--entry-commission", type=float, help="Entry commission per contract")
    cost_group.add_argument("--exit-commission", type=float, help="Exit commission per contract")
    cost_group.add_argument("--round-trip-fee", "-F", type=float, help="Round trip fee per contract")
    cost_group.add_argument("--round-trip-commission", '-C', type=float, help="Round trip commission per contract")
    cost_group.add_argument("--all-in-round-trip-cost", '-A', type=float, help="Round trip fees + commission per contract")
    cost_group.add_argument("--reset", '-r', action="store_true", help="Restore configuration file to defaults")
