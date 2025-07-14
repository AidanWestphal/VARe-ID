import argparse

from GGR.util.io.logging import log_subprocess, setup_logging
from GGR.util.io.workflow_funcs import decode_config

def main(args):
    config = decode_config(args.config)

    command = f'python -m GGR.algo.import.import_videos {config["data_dir_in"]} {config["video_out_path"]}'

    logger = setup_logging(config["import_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the import component of the pipeline. Reads in a recursive directory of images or videos and parses them into a json data file. It will also split a video into frames."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
