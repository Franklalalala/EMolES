import argparse
import logging
from pathlib import Path
from typing import List, Optional

from emoles import __version__
from emoles.entrypoints.config import config
from emoles.entrypoints.data import data
from emoles.entrypoints.train import train
from emoles.utils.config_check import check_config_train
from emoles.utils.loggers import set_log_handles


def get_ll(log_level: str) -> int:
    if log_level.isdigit():
        return (4 - int(log_level)) * 10
    return getattr(logging, log_level)


def _log_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-ll",
        "--log-level",
        choices=["DEBUG", "3", "INFO", "2", "WARNING", "1", "ERROR", "0"],
        default="INFO",
        help="set verbosity level by string or number",
    )
    parser.add_argument(
        "-lp",
        "--log-path",
        type=str,
        default=None,
        help="write logs to this file as well as the console",
    )
    return parser


def main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "EMolES trains density-matrix electronic-structure models for "
            "electrolyte descriptors."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="show the EMolES version and exit",
    )

    parser_log = _log_parser()
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    parser_config = subparsers.add_parser(
        "config",
        parents=[parser_log],
        help="write a training config template",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_config.add_argument("PATH", type=str, help="output config path")
    parser_config.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="checkpoint used to update the template",
    )
    parser_config.add_argument(
        "-tr",
        "--train",
        action="store_true",
        help="generate a training template",
    )
    parser_config.add_argument(
        "-e3",
        "--e3tb",
        action="store_true",
        help="generate the equivariant electronic-structure template",
    )

    parser_train = subparsers.add_parser(
        "train",
        parents=[parser_log],
        help="train an EMolES model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_train.add_argument("INPUT", type=str, help="input JSON/YAML config")
    parser_train.add_argument(
        "-i",
        "--init-model",
        type=str,
        default=None,
        help="checkpoint used to initialize model weights",
    )
    parser_train.add_argument(
        "-r",
        "--restart",
        type=str,
        default=None,
        help="checkpoint used to restart optimizer/training state",
    )
    parser_train.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="training output directory",
    )

    parser_data = subparsers.add_parser(
        "data",
        parents=[parser_log],
        help="prepare or split training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_data.add_argument("INPUT", type=str, help="input JSON/YAML config")
    parser_data.add_argument("-p", "--parse", action="store_true")
    parser_data.add_argument("-s", "--split", action="store_true")
    parser_data.add_argument("-c", "--collect", action="store_true")

    return parser


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = main_parser()
    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()
    else:
        parsed_args.log_level = get_ll(parsed_args.log_level)
    return parsed_args


def main(args: Optional[List[str]] = None):
    parsed_args = parse_args(args)
    if parsed_args.command is None:
        return None

    set_log_handles(
        parsed_args.log_level,
        Path(parsed_args.log_path) if parsed_args.log_path else None,
    )

    dict_args = vars(parsed_args)
    if parsed_args.command == "config":
        return config(**dict_args)
    if parsed_args.command == "train":
        check_config_train(**dict_args)
        return train(**dict_args)
    if parsed_args.command == "data":
        return data(**dict_args)
    raise ValueError(f"Unsupported command: {parsed_args.command}")
