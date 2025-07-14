import argparse

from GGR.util.io.logging import log_subprocess, setup_logging
from GGR.util.io.workflow_funcs import decode_config

def main(args):
    config = decode_config(args.config)

    flag = "--video" if config["data_video"] else ""

    command = f'python -m GGR.algo.ia_classification.IA_filtering {config["ia_out_path"]} {config["ia_filtered_out_path"]} {flag}'

    logger = setup_logging(config["iaf_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the identifiable annotation filtering component of the pipeline. Filters out all annotations not suitable for identification and simplifies viewpoint."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
