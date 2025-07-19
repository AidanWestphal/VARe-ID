import argparse

from VAREID.libraries.io.format_funcs import load_config
from VAREID.libraries.io.logging import setup_logging, log_subprocess
from VAREID.libraries.io.workflow_funcs import build_config, decode_config

def main(args):
    # SELECT THE CORRECT CONFIG
    if args.config:
        config = decode_config(args.config)
    else:
        config = build_config(load_config(args.config_path))

    command = f'python -u -m VAREID.algo.import.import_images {config["data_dir_in"]} {config["image_out_path"]}'

    logger = setup_logging(config["import_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the import component of the pipeline. Reads in a recursive directory of images or videos and parses them into a json data file. It will also split a video into frames."
    )
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--config",
        type=str,
        default=None,
        help="The built config file as a base64 encoded string. Config file MUST be structured like config.yaml!",
    )
    group.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="A path to the config file to load. Config file MUST be structured like config.yaml!",
    )
    args = parser.parse_args()

    main(args)
    