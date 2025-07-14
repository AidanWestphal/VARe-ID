import argparse
import subprocess

from GGR.util.io.logging import log_subprocess, setup_logging
from GGR.util.io.workflow_funcs import decode_config


def get_inputs(config):
    return [config["fs_out_path"]] if config["data_video"] else [config["ia_filtered_out_path"]]


def main(args):
    config = decode_config(args.config)
    
    input = config["fs_out_path"] if config["data_video"] else config["ia_filtered_out_path"]

    command = f'python -m miew_id {input} {config["mid_model"]} {config["mid_out_path"]}'

    logger = setup_logging(config["mid_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the miew-id embedding component of the pipeline. Generates embeddings for annotations."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
