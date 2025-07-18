import argparse
import subprocess

from VAREID.util.io.format_funcs import load_config
from VAREID.util.io.logging import log_subprocess, setup_logging
from VAREID.util.io.workflow_funcs import build_config, decode_config


def get_inputs(config):
    return [config["fs_out_path"]] if config["data_video"] else [config["ia_filtered_out_path"]]


def main(args):
    # SELECT THE CORRECT CONFIG
    if args.config:
        config = decode_config(args.config)
    else:
        config = build_config(load_config(args.config_path))
    
    input = config["fs_out_path"] if config["data_video"] else config["ia_filtered_out_path"]

    command = f'python -u -m VAREID.algo.miew_id.miew_id {input} {config["mid_model"]} {config["mid_out_path"]}'

    logger = setup_logging(config["mid_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the miew-id embedding component of the pipeline. Generates embeddings for annotations."
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
    